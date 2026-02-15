"""
Policy Comparison

Compare RL agent with baseline policies.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from env import BatchingEnv
from agent import DQNAgent
from baselines import FixedBatchPolicy, FixedWaitPolicy, RandomPolicy, evaluate_baseline
from training import evaluate_agent


def compare_policies(
    model_path: str,
    traffic_pattern: str = 'poisson',
    num_episodes: int = 20,
    save_path: str = None
):
    """
    Compare DQN agent with baseline policies.
    
    Args:
        model_path: Path to trained DQN model
        traffic_pattern: Traffic pattern for evaluation
        num_episodes: Number of evaluation episodes
        save_path: Path to save comparison plot
    """
    # Create environment
    env = BatchingEnv(
        traffic_pattern=traffic_pattern,
        max_steps=1000,
        seed=42
    )
    
    # Load DQN agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded DQN model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        return
    
    # Define policies
    policies = {
        'DQN Agent': ('dqn', agent),
        'Fixed Batch (16)': ('baseline', FixedBatchPolicy(batch_threshold=16)),
        'Fixed Batch (8)': ('baseline', FixedBatchPolicy(batch_threshold=8)),
        'Fixed Wait (1.0s)': ('baseline', FixedWaitPolicy(wait_threshold=1.0)),
        'Fixed Wait (0.5s)': ('baseline', FixedWaitPolicy(wait_threshold=0.5)),
        'Random': ('baseline', RandomPolicy(skip_prob=0.3, seed=42))
    }
    
    # Evaluate each policy
    results = {}
    
    print(f"\nEvaluating policies on {traffic_pattern} traffic...")
    print("=" * 60)
    
    for name, (policy_type, policy) in policies.items():
        if policy_type == 'dqn':
            result = evaluate_agent(policy, env, num_episodes=num_episodes)
        else:
            result = evaluate_baseline(policy, env, num_episodes=num_episodes)
        
        results[name] = result
        
        print(f"\n{name}:")
        print(f"  Reward: {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
        print(f"  Batch Size: {result['mean_batch_size']:.2f}")
        print(f"  Wait Time: {result['mean_wait_time']:.3f}s")
        print(f"  Throughput: {result['mean_throughput']:.2f} req/s")
    
    print("\n" + "=" * 60)
    
    # Plot comparison
    _plot_comparison(results, traffic_pattern, save_path)
    
    return results


def _plot_comparison(results: dict, traffic_pattern: str, save_path: str = None):
    """Create comparison visualization."""
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Policy Comparison - {traffic_pattern.capitalize()} Traffic', 
                 fontsize=16, fontweight='bold')
    
    policies = list(results.keys())
    colors = sns.color_palette("husl", len(policies))
    
    # Extract metrics
    rewards = [results[p]['mean_reward'] for p in policies]
    reward_stds = [results[p]['std_reward'] for p in policies]
    batch_sizes = [results[p]['mean_batch_size'] for p in policies]
    wait_times = [results[p]['mean_wait_time'] for p in policies]
    throughputs = [results[p]['mean_throughput'] for p in policies]
    
    # Plot 1: Mean Reward
    axes[0, 0].bar(range(len(policies)), rewards, yerr=reward_stds, 
                   color=colors, alpha=0.7, capsize=5)
    axes[0, 0].set_xticks(range(len(policies)))
    axes[0, 0].set_xticklabels(policies, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].set_title('Average Reward per Episode')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Batch Size
    axes[0, 1].bar(range(len(policies)), batch_sizes, color=colors, alpha=0.7)
    axes[0, 1].set_xticks(range(len(policies)))
    axes[0, 1].set_xticklabels(policies, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Avg Batch Size')
    axes[0, 1].set_title('Average Batch Size')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Wait Time
    axes[1, 0].bar(range(len(policies)), wait_times, color=colors, alpha=0.7)
    axes[1, 0].set_xticks(range(len(policies)))
    axes[1, 0].set_xticklabels(policies, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Avg Wait Time (s)')
    axes[1, 0].set_title('Average Wait Time')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Throughput
    axes[1, 1].bar(range(len(policies)), throughputs, color=colors, alpha=0.7)
    axes[1, 1].set_xticks(range(len(policies)))
    axes[1, 1].set_xticklabels(policies, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Throughput (req/s)')
    axes[1, 1].set_title('Request Throughput')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to {save_path}")
    
    plt.show()


def compare_traffic_patterns(model_path: str, save_dir: str = 'results'):
    """
    Evaluate DQN on different traffic patterns.
    
    Args:
        model_path: Path to trained DQN model
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    patterns = ['poisson', 'bursty', 'time_varying']
    
    for pattern in patterns:
        print(f"\n{'='*60}")
        print(f"Evaluating on {pattern} traffic pattern")
        print('='*60)
        
        save_path = os.path.join(save_dir, f'comparison_{pattern}.png')
        compare_policies(model_path, traffic_pattern=pattern, 
                        num_episodes=20, save_path=save_path)
