"""
A2C Agent (Advantage Actor-Critic)

Algorithm:
    1. At each step, the Actor outputs π(a|s) and the Critic outputs V(s)
    2. After taking action a, observe reward r and next state s'
    3. Compute TD advantage: A(s,a) = r + γ*V(s') - V(s)
    4. Update Actor: θ ← θ + α * ∇_θ log π(a|s) * A(s,a)
    5. Update Critic: minimize MSE between V(s) and r + γ*V(s')

Key difference from REINFORCE:
    - REINFORCE waits for the FULL episode (Monte Carlo return G_t)
    - A2C updates every N steps using TD estimates (bootstrapping)
    - This gives lower variance but introduces some bias
    - The shared network backbone means the critic's gradients
      improve the feature representations used by the actor

Why A2C for request batching:
    - Online learning: updates while the episode is still running
    - Lower variance than REINFORCE → more stable training
    - Shared trunk means fewer parameters to learn
    - N-step returns balance bias/variance tradeoff
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from .networks import ActorCriticNetwork


class A2CAgent:
    """Advantage Actor-Critic agent for request batching."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        n_steps: int = 5,
        max_grad_norm: float = 0.5,
        device: str = None
    ):
        """
        Initialize A2C agent.
        
        Args:
            state_dim: Dimension of state space (6 for BatchingEnv)
            action_dim: Number of actions (2: WAIT/SKIP)
            learning_rate: Learning rate for combined actor-critic network
            gamma: Discount factor
            entropy_coeff: Entropy bonus coefficient (exploration)
            value_coeff: Weight of value loss relative to policy loss
            n_steps: Number of steps for N-step returns
            max_grad_norm: Maximum gradient norm for clipping
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
        
        # Shared Actor-Critic network
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        
        # Single optimizer for entire network
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.n_steps = n_steps
        self.max_grad_norm = max_grad_norm
        
        # N-step buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        
        # Training counters
        self.episodes_trained = 0
        self.steps_trained = 0
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action from the actor's policy distribution.
        
        The actor outputs π(a|s) and the critic outputs V(s).
        Both are computed in a single forward pass through the
        shared network.
        
        Args:
            state: Current state observation
            explore: If True, sample; if False, argmax
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, value = self.network(state_tensor)
        
        dist = Categorical(action_probs)
        
        if explore:
            action = dist.sample()
            # Store for N-step update
            self.log_probs.append(dist.log_prob(action))
            self.values.append(value.squeeze())
            self.entropies.append(dist.entropy())
            self.states.append(state)
        else:
            action = action_probs.argmax(dim=-1)
        
        return action.item()
    
    def store_reward(self, reward: float):
        """Store step reward."""
        self.rewards.append(reward)
    
    def should_update(self) -> bool:
        """Check if we have enough steps for an N-step update."""
        return len(self.rewards) >= self.n_steps
    
    def update(self, next_state: np.ndarray = None, done: bool = False) -> dict:
        """
        Perform N-step A2C update.
        
        Computes N-step returns:
            R_t = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * V(s_{t+n})
        
        The last term (bootstrap) is only added if the episode hasn't ended.
        
        Args:
            next_state: State after the N steps (for bootstrapping V(s'))
            done: Whether the episode ended
            
        Returns:
            Dictionary with losses
        """
        if len(self.rewards) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # ── Step 1: Compute bootstrap value V(s_{t+n}) ──
        if done or next_state is None:
            R = torch.tensor(0.0).to(self.device)
        else:
            with torch.no_grad():
                next_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, next_value = self.network(next_tensor)
                R = next_value.squeeze()
        
        # ── Step 2: Compute N-step returns (backward) ──
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.stack(returns).to(self.device)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        entropies = torch.stack(self.entropies)
        
        # ── Step 3: Compute advantages ──
        # A(s,a) = R_t (N-step return) - V(s_t)
        advantages = returns - values.detach()
        
        # ── Step 4: Compute losses ──
        # Policy loss: -log π(a|s) * A(s,a)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss: MSE(V(s), R_t)
        value_loss = nn.MSELoss()(values, returns.detach())
        
        # Entropy bonus: encourages exploration
        entropy_loss = -entropies.mean()
        
        # Total loss (combined for single optimizer)
        total_loss = (
            policy_loss 
            + self.value_coeff * value_loss 
            + self.entropy_coeff * entropy_loss
        )
        
        # ── Step 5: Backpropagate and update ──
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.steps_trained += len(self.rewards)
        
        # Clear buffers
        avg_entropy = entropies.mean().item()
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': avg_entropy,
            'total_loss': total_loss.item()
        }
    
    def end_episode(self):
        """Called at end of episode to increment counter."""
        self.episodes_trained += 1
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episodes_trained': self.episodes_trained,
            'steps_trained': self.steps_trained
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episodes_trained = checkpoint['episodes_trained']
        self.steps_trained = checkpoint['steps_trained']
