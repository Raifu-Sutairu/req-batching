"""
Training Configuration

Centralized configuration for hyperparameters and environment settings.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for training the DQN agent."""
    
    # Environment parameters
    max_batch_size: int = 32
    max_wait_time: float = 2.0
    max_queue_length: int = 100
    traffic_pattern: str = 'poisson'  # 'poisson', 'bursty', 'time_varying'
    base_arrival_rate: float = 5.0
    alpha: float = 1.0  # batch efficiency weight
    beta: float = 2.0   # latency penalty weight
    episode_steps: int = 1000
    dt: float = 0.1
    
    # Agent parameters
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_capacity: int = 10000
    batch_size: int = 64
    target_update_freq: int = 10
    
    # Training parameters
    num_episodes: int = 500
    eval_interval: int = 50
    save_interval: int = 100
    
    # Paths
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    # Device
    device: str = None  # None for auto-detect
    
    # Random seed
    seed: int = 42
