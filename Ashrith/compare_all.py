"""
Compare All RL Agents

Evaluates REINFORCE, A2C, PPO alongside the existing DQN and baselines.
Generates side-by-side comparison plots.

Usage:
    python -m Ashrith.compare_all
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import BatchingEnv
from Ashrith.reinforce_agent import REINFORCEAgent
from Ashrith.a2c_agent import A2CAgent
from Ashrith.ppo_agent import PPOAgent

# Try to import DQN agent from the existing project
try:
    from agent import DQNAgent
    HAS_DQN = True
except ImportError:
    HAS_DQN = False

from baselines.naive_policies import (
    FixedBatchPolicy, FixedWaitPolicy, RandomPolicy, evaluate_baseline
)


def evaluate_policy_agent(agent, env, num_episodes=20):
    """
    Evaluate a policy-based agent (REINFORCE/A2C/PPO).
    
    All three share the same interface: select_action(state, explore=False).
    """
    episode_rewards = []
    episode_metrics = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state, explore=False)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
        
        metrics = env.get_metrics()
        episode_rewards.append(episode_reward)
        episode_metrics.append(metrics)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_batch_size': np.mean([m['avg_batch_size'] for m in episode_metrics]),
        'mean_wait_time': np.mean([m['avg_wait_time'] for m in episode_metrics]),
        'mean_throughput': np.mean([m['throughput'] for m in episode_metrics]),
    }


def plot_comparison(results: dict, save_path: str):
    """
    Generate comparison bar charts for all agents.
    
    Creates a 2x2 figure comparing:
    - Mean Reward
    - Avg Batch Size
    - Avg Wait Time
    - Throughput
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RL Agent Comparison — Request Batching', fontsize=16, fontweight='bold')
    
    names = list(results.keys())
    
    # Color palette: RL agents in blue/purple tones, baselines in gray/orange
    colors = []
    for name in names:
        if name in ('REINFORCE', 'A2C', 'PPO'):
            colors.append({'REINFORCE': '#3498db', 'A2C': '#9b59b6', 'PPO': '#e74c3c'}[name])
        elif name == 'DQN':
            colors.append('#2ecc71')
        else:
            colors.append('#95a5a6')
    
    # Mean Reward
    ax = axes[0, 0]
    rewards = [results[n]['mean_reward'] for n in names]
    errors = [results[n]['std_reward'] for n in names]
    bars = ax.bar(names, rewards, color=colors, edgecolor='white', linewidth=1.5)
    ax.errorbar(names, rewards, yerr=errors, fmt='none', color='black', capsize=5)
    ax.set_title('Mean Episode Reward', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reward')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Avg Batch Size
    ax = axes[0, 1]
    batch_sizes = [results[n]['mean_batch_size'] for n in names]
    ax.bar(names, batch_sizes, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title('Avg Batch Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Requests / Batch')
    ax.grid(axis='y', alpha=0.3)
    
    # Avg Wait Time
    ax = axes[1, 0]
    wait_times = [results[n]['mean_wait_time'] for n in names]
    ax.bar(names, wait_times, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title('Avg Wait Time', fontsize=13, fontweight='bold')
    ax.set_ylabel('Seconds')
    ax.grid(axis='y', alpha=0.3)
    
    # Throughput
    ax = axes[1, 1]
    throughputs = [results[n]['mean_throughput'] for n in names]
    ax.bar(names, throughputs, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title('Throughput', fontsize=13, fontweight='bold')
    ax.set_ylabel('Requests / Second')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison plot saved to: {save_path}")


def plot_training_curves(save_path: str):
    """
    Plot training reward curves for all 3 policy-based agents on one figure.
    Reads from Ashrith/logs/*_logs.json.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    agent_colors = {
        'reinforce': ('#3498db', 'REINFORCE'),
        'a2c': ('#9b59b6', 'A2C'),
        'ppo': ('#e74c3c', 'PPO')
    }
    
    found_any = False
    for agent_key, (color, label) in agent_colors.items():
        log_path = os.path.join('Ashrith', 'logs', f'{agent_key}_logs.json')
        if os.path.exists(log_path):
            with open(log_path) as f:
                logs = json.load(f)
            rewards = logs['rewards']
            
            # Plot raw rewards (faint) and smoothed (bold)
            ax.plot(rewards, alpha=0.15, color=color)
            
            # Moving average (window=30)
            window = min(30, len(rewards) // 3) if len(rewards) > 10 else 1
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, window-1+len(smoothed)), smoothed, 
                    color=color, linewidth=2.5, label=label)
            found_any = True
    
    if not found_any:
        print("  No training logs found in Ashrith/logs/")
        plt.close()
        return
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Training Curves — Policy-Based Agents', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved to: {save_path}")


def compare_all(traffic_pattern='poisson', num_eval_episodes=20, seed=42):
    """
    Main comparison: evaluate all agents and baselines.
    
    Returns:
        Dictionary of results per agent/baseline
    """
    env = BatchingEnv(
        traffic_pattern=traffic_pattern,
        seed=seed,
        max_steps=1000
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    results = {}
    
    print("\n" + "="*60)
    print("Evaluating All Agents & Baselines")
    print("="*60)
    
    # ─── Policy-Based Agents ───
    agents_to_eval = [
        ('reinforce', REINFORCEAgent, 'reinforce_best.pth'),
        ('a2c', A2CAgent, 'a2c_best.pth'),
        ('ppo', PPOAgent, 'ppo_best.pth'),
    ]
    
    for agent_key, AgentClass, ckpt_name in agents_to_eval:
        ckpt_path = os.path.join('Ashrith', 'checkpoints', ckpt_name)
        if os.path.exists(ckpt_path):
            agent = AgentClass(state_dim=state_dim, action_dim=action_dim)
            agent.load(ckpt_path)
            r = evaluate_policy_agent(agent, env, num_eval_episodes)
            results[agent_key.upper()] = r
            print(f"  {agent_key.upper():>12}: Reward={r['mean_reward']:.3f} ± {r['std_reward']:.3f} | "
                  f"Batch={r['mean_batch_size']:.1f} | Wait={r['mean_wait_time']:.3f}s")
        else:
            print(f"  {agent_key.upper():>12}: (no checkpoint found at {ckpt_path})")
    
    # ─── DQN Agent ───
    if HAS_DQN:
        dqn_path = os.path.join('checkpoints', 'dqn_best.pth')
        if os.path.exists(dqn_path):
            dqn = DQNAgent(state_dim=state_dim, action_dim=action_dim)
            dqn.load(dqn_path)
            r = evaluate_policy_agent(dqn, env, num_eval_episodes)
            results['DQN'] = r
            print(f"  {'DQN':>12}: Reward={r['mean_reward']:.3f} ± {r['std_reward']:.3f} | "
                  f"Batch={r['mean_batch_size']:.1f} | Wait={r['mean_wait_time']:.3f}s")
    
    # ─── Baselines ───
    baselines = {
        'Fixed Batch': FixedBatchPolicy(batch_threshold=16),
        'Fixed Wait': FixedWaitPolicy(wait_threshold=1.0),
        'Random': RandomPolicy(skip_prob=0.3, seed=seed),
    }
    
    for name, policy in baselines.items():
        r = evaluate_baseline(policy, env, num_eval_episodes)
        results[name] = r
        print(f"  {name:>12}: Reward={r['mean_reward']:.3f} ± {r['std_reward']:.3f} | "
              f"Batch={r['mean_batch_size']:.1f} | Wait={r['mean_wait_time']:.3f}s")
    
    print("="*60)
    
    return results


if __name__ == '__main__':
    results = compare_all()
    
    os.makedirs(os.path.join('Ashrith', 'results'), exist_ok=True)
    
    if results:
        plot_comparison(results, os.path.join('Ashrith', 'results', 'comparison.png'))
    
    plot_training_curves(os.path.join('Ashrith', 'results', 'training_curves.png'))
    
    # Save results as JSON
    json_results = {}
    for k, v in results.items():
        json_results[k] = {key: float(val) for key, val in v.items()}
    
    with open(os.path.join('Ashrith', 'results', 'comparison_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  Results JSON saved to: Ashrith/results/comparison_results.json")
