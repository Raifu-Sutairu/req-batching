"""
Compare All RL Agents
Evaluates REINFORCE, A2C, PPO, Predictive Dyna-Q alongside baselines.
Generates a comprehensive final evaluation table.

Usage:
    python -m Ashrith.compare_all
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ashrith.env import BatchingEnv
from Ashrith.legacy.reinforce_agent import REINFORCEAgent
from Ashrith.legacy.a2c_agent import A2CAgent
from Ashrith.legacy.ppo_agent import PPOAgent
from Ashrith.predictive_dynaq_agent import PredictiveDynaQAgent

from Ashrith.baselines.naive_policies import (
    FixedBatchPolicy, FixedWaitPolicy, RandomPolicy, evaluate_baseline
)


def evaluate_policy_agent(agent, env, num_episodes=20):
    episode_rewards = []
    episode_metrics = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        if hasattr(agent, 'start_episode'):
            agent.start_episode()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state, explore=False)
            next_state, reward, done, truncated, info = env.step(action)
            if hasattr(agent, 'observe'):
                agent.observe(
                    state=state, action=action, reward=reward,
                    next_state=next_state, done=(done or truncated), train=False
                )
            state = next_state
            episode_reward += reward
        
        metrics = env.get_metrics()
        episode_rewards.append(episode_reward)
        episode_metrics.append(metrics)
    
    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_batch_size': float(np.mean([m['avg_batch_size'] for m in episode_metrics])),
        'mean_wait_time': float(np.mean([m['avg_wait_time'] for m in episode_metrics])),
        'p95_wait_time': float(np.mean([m.get('p95_wait_time', 0.0) for m in episode_metrics])),
        'mean_throughput': float(np.mean([m['throughput'] for m in episode_metrics])),
        'avg_queue': float(np.mean([m.get('avg_queue_length', 0.0) for m in episode_metrics])),
        'slo_violations': float(np.mean([m.get('slo_violation_rate', 0.0) for m in episode_metrics])),
    }


def compare_all():
    print("\n" + "="*80)
    print("Evaluating All Agents across Traffic Patterns and Seeds")
    print("="*80)

    traffic_patterns = ['poisson', 'bursty', 'time_varying']
    seeds = [42, 100, 256, 1024, 2048]
    num_eval_episodes = 5  # Evaluate for 5 episodes per seed
    
    # Needs a mock env to get dimensions
    dummy_env = BatchingEnv()
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n

    agents_config = [
        ('REINFORCE', REINFORCEAgent, os.path.join('Ashrith', 'legacy', 'checkpoints', 'reinforce_best.pth'), True),
        ('A2C', A2CAgent, os.path.join('Ashrith', 'legacy', 'checkpoints', 'a2c_best.pth'), True),
        ('PPO', PPOAgent, os.path.join('Ashrith', 'legacy', 'checkpoints', 'ppo_best.pth'), True),
        ('Predictive Dyna-Q', PredictiveDynaQAgent, os.path.join('Ashrith', 'checkpoints', 'predictive_dynaq_best.npy'), False),
    ]

    baselines = {
        'Fixed Batch (16)': FixedBatchPolicy(batch_threshold=16),
        'Fixed Wait (1.0s)': FixedWaitPolicy(wait_threshold=1.0),
        'Random': RandomPolicy(skip_prob=0.3),
    }

    # Store results: agent -> dict of combined stats
    compiled_runs = defaultdict(list)
    
    for traffic in traffic_patterns:
        print(f"\n--- Testing Traffic Pattern: {traffic.upper()} ---")
        for seed in seeds:
            env = BatchingEnv(traffic_pattern=traffic, seed=seed, max_steps=1000)
            
            # Evaluate RL Agents
            for name, AgentClass, ckpt_path, uses_state_dim in agents_config:
                if os.path.exists(ckpt_path):
                    if uses_state_dim:
                        agent = AgentClass(state_dim=state_dim, action_dim=action_dim)
                    else:
                        agent = AgentClass(action_dim=action_dim)
                    agent.load(ckpt_path)
                    res = evaluate_policy_agent(agent, env, num_episodes=num_eval_episodes)
                    compiled_runs[name].append(res)
            
            # Evaluate Baselines
            for b_name, policy in baselines.items():
                if hasattr(policy, 'rng'):
                    policy.rng = np.random.RandomState(seed)
                res = evaluate_baseline(policy, env, num_episodes=num_eval_episodes)
                compiled_runs[b_name].append(res)

    print("\n" + "="*120)
    print(f"{'Final Comparison Table':^120}")
    print("="*120)
    print(f"| {'Agent':<20} | {'Mean Reward':<13} | {'Avg Wait':<10} | {'P95 Wait':<10} | {'Avg Batch':<10} | {'Throughput':<10} | {'Avg Queue':<10} | {'SLO Violations':<15} |")
    print("|" + "-"*22 + "|" + "-"*15 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*17 + "|")

    final_results = {}
    
    for agent_name, runs in compiled_runs.items():
        # Average across all traffics and seeds combined
        avg_reward = np.mean([r['mean_reward'] for r in runs])
        std_reward = np.std([r['mean_reward'] for r in runs])
        avg_wait = np.mean([r['mean_wait_time'] for r in runs])
        p95_wait = np.mean([r['p95_wait_time'] for r in runs])
        avg_batch = np.mean([r['mean_batch_size'] for r in runs])
        avg_throughput = np.mean([r['mean_throughput'] for r in runs])
        avg_queue = np.mean([r['avg_queue'] for r in runs])
        slo_viol = np.mean([r['slo_violations'] for r in runs])
        
        final_results[agent_name] = {
            'mean_reward': avg_reward, 'std_reward': std_reward,
            'mean_wait': avg_wait, 'p95_wait': p95_wait,
            'mean_batch': avg_batch, 'throughput': avg_throughput,
            'avg_queue': avg_queue, 'slo_violations': slo_viol
        }

        print(f"| {agent_name:<20} | {avg_reward:>6.2f} ± {std_reward:>4.2f} | {avg_wait:>8.3f}s | {p95_wait:>8.3f}s | {avg_batch:>10.1f} | {avg_throughput:>8.1f}/s | {avg_queue:>10.1f} | {slo_viol:>14.1f}% |")

    print("="*120)
    
    return final_results

if __name__ == '__main__':
    results = compare_all()
    
    os.makedirs(os.path.join('Ashrith', 'results'), exist_ok=True)
    with open(os.path.join('Ashrith', 'results', 'final_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON saved to: Ashrith/results/final_evaluation_results.json")
