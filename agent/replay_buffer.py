"""
Experience Replay Buffer

Stores transitions for off-policy learning and breaks correlation
between consecutive samples.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)
