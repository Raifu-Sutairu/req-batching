"""
Evaluation Utilities

Functions for evaluating trained agents.
"""

import numpy as np
from env import BatchingEnv
from agent import DQNAgent


def evaluate_agent(
    agent: DQNAgent,
    env: BatchingEnv,
    num_episodes: int = 10,
    render: bool = False
):
    """
    Evaluate agent performance.
    
    Args:
        agent: Trained DQN agent
        env: Batching environment
        num_episodes: Number of evaluation episodes
        render: Whether to render (not implemented)
        
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
            # Select action (no exploration)
            action = agent.select_action(state, explore=False)
            
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


def print_evaluation_results(results: dict, policy_name: str = "DQN"):
    """Pretty print evaluation results."""
    print(f"\n{policy_name} Evaluation Results:")
    print("-" * 50)
    print(f"  Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Mean Batch Size: {results['mean_batch_size']:.2f}")
    print(f"  Mean Wait Time: {results['mean_wait_time']:.3f}s")
    print(f"  Mean Throughput: {results['mean_throughput']:.2f} req/s")
    print(f"  Total Requests: {results['total_requests']:.0f}")
    print(f"  Total Batches: {results['total_batches']:.0f}")
    print("-" * 50)
