"""
Neural Network Architecture for DQN

Simple feedforward network for Q-value approximation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Deep Q-Network for action-value function approximation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 64)):
        """
        Initialize DQN.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_sizes: Tuple of hidden layer sizes
        """
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_dim)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: State tensor [batch_size, state_dim]
            
        Returns:
            Q-values for each action [batch_size, action_dim]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
