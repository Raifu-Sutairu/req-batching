"""
Neural Network Architectures for Policy-Based RL

Contains:
- PolicyNetwork: Outputs action probability distribution
- ValueNetwork: Outputs state-value estimate V(s)
- ActorCriticNetwork: Shared backbone with separate policy & value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Policy network that maps states to action probabilities.
    
    Used by REINFORCE agent.
    Output: π(a|s) via Softmax over discrete actions.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 64)):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_dim)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: State tensor [batch_size, state_dim]
            
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class ValueNetwork(nn.Module):
    """
    Value network that estimates V(s).
    
    Used as the baseline in REINFORCE to reduce variance.
    Output: Scalar state-value.
    """
    
    def __init__(self, state_dim: int, hidden_sizes=(128, 64)):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: State tensor [batch_size, state_dim]
            
        Returns:
            State-value estimate [batch_size, 1]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network with shared feature extraction.
    
    Used by A2C and PPO agents.
    
    Architecture:
        State → [Shared Trunk] → Actor Head → π(a|s)
                                → Critic Head → V(s)
    
    Sharing the trunk allows the critic's value gradients to improve
    the feature representations used by the actor.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction trunk
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Actor head: outputs action probabilities
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        
        # Critic head: outputs state-value estimate
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        """
        Forward pass through both heads.
        
        Args:
            x: State tensor [batch_size, state_dim]
            
        Returns:
            action_probs: π(a|s) [batch_size, action_dim]
            state_value: V(s) [batch_size, 1]
        """
        features = self.shared(x)
        action_probs = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_probs, state_value
    
    def get_action_probs(self, x):
        """Get only action probabilities (for inference)."""
        features = self.shared(x)
        return self.actor_head(features)
    
    def get_value(self, x):
        """Get only state value (for advantage computation)."""
        features = self.shared(x)
        return self.critic_head(features)
