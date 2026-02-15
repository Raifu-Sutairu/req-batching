"""
Naive Baseline Policies

Simple rule-based policies for comparison with RL agent.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaselinePolicy(ABC):
    """Base class for baseline policies."""
    
    @abstractmethod
    def select_action(self, state: np.ndarray, info: dict) -> int:
        """
        Select action based on state.
        
        Args:
            state: Current state
            info: Additional info from environment
            
        Returns:
            Action (0=WAIT, 1=SKIP)
        """
        pass


class FixedBatchPolicy(BaselinePolicy):
    """Always SKIP when batch reaches a fixed size threshold."""
    
    def __init__(self, batch_threshold: int = 16):
        """
        Initialize fixed batch policy.
        
        Args:
            batch_threshold: Batch size threshold for SKIP
        """
        self.batch_threshold = batch_threshold
    
    def select_action(self, state: np.ndarray, info: dict) -> int:
        """SKIP if batch size >= threshold, else WAIT."""
        batch_size = info.get('batch_size', 0)
        return 1 if batch_size >= self.batch_threshold else 0


class FixedWaitPolicy(BaselinePolicy):
    """Always SKIP when wait time exceeds a threshold."""
    
    def __init__(self, wait_threshold: float = 1.0):
        """
        Initialize fixed wait policy.
        
        Args:
            wait_threshold: Wait time threshold (seconds) for SKIP
        """
        self.wait_threshold = wait_threshold
    
    def select_action(self, state: np.ndarray, info: dict) -> int:
        """SKIP if wait time >= threshold, else WAIT."""
        wait_time = info.get('wait_time', 0)
        return 1 if wait_time >= self.wait_threshold else 0


class RandomPolicy(BaselinePolicy):
    """Random WAIT/SKIP decisions."""
    
    def __init__(self, skip_prob: float = 0.3, seed: int = None):
        """
        Initialize random policy.
        
        Args:
            skip_prob: Probability of SKIP action
            seed: Random seed
        """
        self.skip_prob = skip_prob
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, state: np.ndarray, info: dict) -> int:
        """Random action with probability skip_prob for SKIP."""
        return 1 if self.rng.random() < self.skip_prob else 0


def evaluate_baseline(policy: BaselinePolicy, env, num_episodes: int = 10):
    """
    Evaluate a baseline policy.
    
    Args:
        policy: Baseline policy to evaluate
        env: Batching environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_metrics = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get action from policy
            # We need the last info to pass to policy
            action = 0  # default
            
            # Take a dummy step first to get info
            # Actually, let's handle this better
            # Policies need current batch size and wait time
            # These are embedded in the state but we need to denormalize
            
            # For simplicity, we'll peek at the environment's internal state
            info = {
                'batch_size': len(env.current_batch),
                'wait_time': env._get_current_wait_time(),
                'queue_length': len(env.wait_queue)
            }
            
            action = policy.select_action(state, info)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            state = next_state
            episode_reward += reward
        
        # Collect metrics
        metrics = env.get_metrics()
        episode_rewards.append(episode_reward)
        episode_metrics.append(metrics)
    
    # Aggregate results
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_batch_size': np.mean([m['avg_batch_size'] for m in episode_metrics]),
        'mean_wait_time': np.mean([m['avg_wait_time'] for m in episode_metrics]),
        'mean_throughput': np.mean([m['throughput'] for m in episode_metrics]),
        'total_requests': np.mean([m['total_requests'] for m in episode_metrics]),
        'total_batches': np.mean([m['total_batches'] for m in episode_metrics])
    }
    
    return results
