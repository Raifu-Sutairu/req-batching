"""
PPO Agent (Proximal Policy Optimization)

Algorithm:
    1. Collect a rollout of T timesteps using current policy π_old
    2. Compute GAE advantages: A_t = Σ (γλ)^l * δ_{t+l}
       where δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
    3. For K epochs, sample mini-batches from the rollout:
       a. Compute probability ratio: r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
       b. Clipped surrogate objective:
          L^CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
       c. Value loss: MSE(V(s), R_t)
       d. Entropy bonus: H(π)
    4. Update θ to maximize L^CLIP - c1*L_value + c2*H

Why PPO for request batching:
    - Most stable and sample-efficient of all three policy methods
    - Clipping prevents destructive large policy updates
    - GAE (λ=0.95) gives excellent bias-variance tradeoff
    - Multiple epochs per rollout = better sample efficiency
    - State-of-the-art for many RL applications
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from .networks import ActorCriticNetwork


class RolloutBuffer:
    """
    Buffer to store rollout experience for PPO updates.
    
    Unlike DQN's replay buffer (which stores individual transitions
    and samples randomly), PPO's rollout buffer stores a contiguous
    sequence of experience and processes it all together with GAE.
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def store(self, state, action, reward, log_prob, value, done):
        """Store a single transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)
    
    def get_batches(self, batch_size: int):
        """
        Generate random mini-batches from the rollout.
        
        PPO processes the same rollout data multiple times (K epochs),
        each time with random mini-batches. This improves sample
        efficiency compared to single-pass methods like A2C.
        
        Args:
            batch_size: Size of each mini-batch
            
        Yields:
            Indices for each mini-batch
        """
        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield indices[start:end]


class PPOAgent:
    """Proximal Policy Optimization agent for request batching."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        mini_batch_size: int = 64,
        rollout_length: int = 128,
        device: str = None
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            learning_rate: Learning rate (PPO typically uses lower LR)
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter ε
            entropy_coeff: Entropy bonus weight
            value_coeff: Value loss weight
            max_grad_norm: Gradient clipping norm
            n_epochs: Number of optimization epochs per rollout
            mini_batch_size: Mini-batch size within each epoch
            rollout_length: Steps to collect before each update
            device: Torch device
        """
        # Device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Network
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.rollout_length = rollout_length
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Counters
        self.episodes_trained = 0
        self.steps_trained = 0
        self.updates_done = 0
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> tuple:
        """
        Select action and store transition data.
        
        PPO needs to remember the log probability under the OLD policy
        (π_old) to compute the importance sampling ratio later.
        
        Args:
            state: Current state
            explore: Sample vs argmax
            
        Returns:
            action (int)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
        
        dist = Categorical(action_probs)
        
        if explore:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Store in buffer (will be completed when reward arrives)
            self._pending_log_prob = log_prob.item()
            self._pending_value = value.squeeze().item()
            self._pending_state = state
            self._pending_action = action.item()
        else:
            action = action_probs.argmax(dim=-1)
        
        return action.item()
    
    def store_transition(self, reward: float, done: bool):
        """
        Complete the pending transition with observed reward and done flag.
        
        Args:
            reward: Reward from environment
            done: Whether episode ended
        """
        self.buffer.store(
            state=self._pending_state,
            action=self._pending_action,
            reward=reward,
            log_prob=self._pending_log_prob,
            value=self._pending_value,
            done=done
        )
    
    def should_update(self) -> bool:
        """Check if rollout buffer is full enough for an update."""
        return len(self.buffer) >= self.rollout_length
    
    def compute_gae(self, next_value: float) -> tuple:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE interpolates between:
          - λ=0: TD(0) advantage (high bias, low variance)
          - λ=1: Monte Carlo advantage (low bias, high variance)
        
        With λ=0.95, we get most of the variance reduction of TD
        while keeping the bias small.
        
        Formula:
            δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            A_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}
        
        Args:
            next_value: V(s_{T+1}) for bootstrapping
            
        Returns:
            (advantages, returns) as numpy arrays
        """
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])
            
            # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            
            # GAE: A_t = δ_t + (γλ)*A_{t+1}
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            last_gae = advantages[t]
        
        # Returns = advantages + values (for value function target)
        returns = advantages + np.array(values, dtype=np.float32)
        
        return advantages, returns
    
    def update(self, next_state: np.ndarray = None) -> dict:
        """
        Perform PPO update with clipped surrogate objective.
        
        This is the core of PPO:
        1. Compute GAE advantages from the collected rollout
        2. For K epochs, take mini-batch gradient steps:
           - Recompute π_θ(a|s) under the CURRENT policy
           - Compute ratio: r(θ) = π_θ / π_old
           - Clip the ratio to prevent too-large updates
           - Minimize: -L^CLIP + c1*L_value - c2*H(π)
        
        Args:
            next_state: State after the rollout (for bootstrap)
            
        Returns:
            Dictionary with loss statistics
        """
        if len(self.buffer) == 0:
            return {'policy_loss': 0., 'value_loss': 0., 'entropy': 0., 'clip_fraction': 0.}
        
        # ── Step 1: Compute GAE advantages ──
        if next_state is not None:
            with torch.no_grad():
                next_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, next_value = self.network(next_tensor)
                next_val = next_value.squeeze().item()
        else:
            next_val = 0.0
        
        advantages, returns = self.compute_gae(next_val)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # ── Step 2: Multiple epochs of mini-batch updates ──
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        num_updates = 0
        
        for epoch in range(self.n_epochs):
            for batch_indices in self.buffer.get_batches(self.mini_batch_size):
                # Get mini-batch
                mb_states = states[batch_indices]
                mb_actions = actions[batch_indices]
                mb_old_log_probs = old_log_probs[batch_indices]
                mb_advantages = advantages_tensor[batch_indices]
                mb_returns = returns_tensor[batch_indices]
                
                # Recompute action probs under CURRENT policy
                action_probs, values = self.network(mb_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # ── Importance sampling ratio ──
                # r(θ) = π_θ(a|s) / π_old(a|s)  = exp(log π_θ - log π_old)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # ── Clipped surrogate objective ──
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 
                                    1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # ── Value loss ──
                value_loss = nn.MSELoss()(values.squeeze(), mb_returns)
                
                # ── Total loss ──
                loss = (
                    policy_loss 
                    + self.value_coeff * value_loss 
                    - self.entropy_coeff * entropy
                )
                
                # ── Optimize ──
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                
                # Track statistics
                clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_fraction += clip_fraction.item()
                num_updates += 1
        
        self.steps_trained += len(self.buffer)
        self.updates_done += 1
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'clip_fraction': total_clip_fraction / max(num_updates, 1)
        }
    
    def end_episode(self):
        """Increment episode counter."""
        self.episodes_trained += 1
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episodes_trained': self.episodes_trained,
            'steps_trained': self.steps_trained,
            'updates_done': self.updates_done
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episodes_trained = checkpoint['episodes_trained']
        self.steps_trained = checkpoint['steps_trained']
        self.updates_done = checkpoint['updates_done']
