"""
REINFORCE Agent (Policy Gradient with Learned Baseline)

Algorithm:
    1. Collect full episode trajectory: (s_0, a_0, r_0), ..., (s_T, a_T, r_T)
    2. Compute discounted returns: G_t = Σ γ^k * r_{t+k}
    3. Compute advantage using baseline: A_t = G_t - V(s_t)
    4. Update policy: θ ← θ + α * ∇_θ log π(a_t|s_t) * A_t
    5. Update baseline: minimize MSE(V(s_t), G_t)

Why REINFORCE for this problem:
    - The action space is discrete (WAIT/SKIP), making it straightforward
    - Policy gradient directly optimizes the policy without needing a Q-table
    - The learned baseline reduces variance significantly
    - Entropy bonus encourages exploration of different batching strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from .networks import PolicyNetwork, ValueNetwork


class REINFORCEAgent:
    """REINFORCE agent with learned baseline for request batching."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        baseline_lr: float = 0.001,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        device: str = None
    ):
        """
        Initialize REINFORCE agent.
        
        Args:
            state_dim: Dimension of state space (6 for BatchingEnv)
            action_dim: Number of actions (2: WAIT/SKIP)
            learning_rate: Policy network learning rate
            baseline_lr: Value baseline learning rate
            gamma: Discount factor for future rewards
            entropy_coeff: Entropy bonus coefficient (encourages exploration)
            device: Torch device (cpu/cuda/mps)
        """
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Networks
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=baseline_lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        
        # Episode buffer (collect full episode before update)
        self.saved_log_probs = []
        self.saved_entropies = []
        self.rewards = []
        self.states = []
        
        # Training counters
        self.episodes_trained = 0
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action by sampling from the policy distribution.
        
        Unlike DQN's epsilon-greedy, REINFORCE naturally explores
        by sampling from π(a|s). The stochastic policy outputs
        probabilities, so early on (when policy is near-uniform)
        it explores broadly, and as training progresses it becomes
        more deterministic.
        
        Args:
            state: Current state observation
            explore: If True, sample from distribution; if False, take argmax
            
        Returns:
            Selected action (0=WAIT, 1=SKIP)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_net(state_tensor)
        
        dist = Categorical(action_probs)
        
        if explore:
            action = dist.sample()
            # Store log probability and entropy for training
            self.saved_log_probs.append(dist.log_prob(action))
            self.saved_entropies.append(dist.entropy())
            self.states.append(state)
        else:
            # Greedy action for evaluation
            action = action_probs.argmax(dim=-1)
        
        return action.item()
    
    def store_reward(self, reward: float):
        """Store reward for current step (called after env.step)."""
        self.rewards.append(reward)
    
    def update(self) -> dict:
        """
        Update policy and baseline using the collected episode.
        
        This is called at the end of each episode (REINFORCE is
        a Monte Carlo method — it needs the full trajectory).
        
        Returns:
            Dictionary with policy_loss, value_loss, avg_entropy
        """
        if len(self.rewards) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'avg_entropy': 0.0}
        
        # ── Step 1: Compute discounted returns G_t ──
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # ── Step 2: Compute baseline values V(s) ──
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        values = self.value_net(states_tensor).squeeze()
        
        # ── Step 3: Compute advantages A_t = G_t - V(s_t) ──
        advantages = returns - values.detach()
        
        # ── Step 4: Policy gradient loss ──
        # L_policy = -Σ log π(a_t|s_t) * A_t  (negative because we minimize)
        policy_loss = 0
        entropy_loss = 0
        for log_prob, advantage, entropy in zip(
            self.saved_log_probs, advantages, self.saved_entropies
        ):
            policy_loss += -log_prob * advantage
            entropy_loss += -entropy  # We want to maximize entropy
        
        policy_loss = policy_loss / len(self.rewards)
        entropy_loss = entropy_loss / len(self.rewards)
        
        total_policy_loss = policy_loss + self.entropy_coeff * entropy_loss
        
        # ── Step 5: Value baseline loss ──
        value_loss = nn.MSELoss()(values, returns)
        
        # ── Step 6: Update networks ──
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()
        
        # Compute stats
        avg_entropy = torch.stack(self.saved_entropies).mean().item()
        
        # Clear episode buffer
        self.saved_log_probs = []
        self.saved_entropies = []
        self.rewards = []
        self.states = []
        
        self.episodes_trained += 1
        
        return {
            'policy_loss': total_policy_loss.item(),
            'value_loss': value_loss.item(),
            'avg_entropy': avg_entropy
        }
    
    def save(self, filepath: str):
        """Save agent state to file."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'episodes_trained': self.episodes_trained
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.episodes_trained = checkpoint['episodes_trained']
