"""
Quick Example: Train and Evaluate DQN Agent

This script demonstrates how to train and evaluate the DQN agent
with minimal code.
"""

from env import BatchingEnv
from agent import DQNAgent
from training import TrainingConfig, train_agent, evaluate_agent
from baselines import FixedBatchPolicy, FixedWaitPolicy, evaluate_baseline
import os


def quick_demo():
    """Run a quick demonstration of the system."""
    
    print("\n" + "="*60)
    print("Quick Demo: RL Request Batching System")
    print("="*60)
    
    # ===== STEP 1: Train Agent =====
    print("\n[1/4] Training DQN Agent...")
    print("-" * 60)
    
    config = TrainingConfig(
        num_episodes=200,  # Reduced for quick demo
        traffic_pattern='poisson',
        alpha=1.0,
        beta=2.0,
        seed=42
    )
    
    agent, logs = train_agent(config)
    
    # ===== STEP 2: Evaluate Agent =====
    print("\n[2/4] Evaluating DQN Agent...")
    print("-" * 60)
    
    env = BatchingEnv(traffic_pattern='poisson', seed=42)
    dqn_results = evaluate_agent(agent, env, num_episodes=10)
    
    print(f"\nDQN Results:")
    print(f"  Mean Reward: {dqn_results['mean_reward']:.3f}")
    print(f"  Mean Batch Size: {dqn_results['mean_batch_size']:.2f}")
    print(f"  Mean Wait Time: {dqn_results['mean_wait_time']:.3f}s")
    print(f"  Mean Throughput: {dqn_results['mean_throughput']:.2f} req/s")
    
    # ===== STEP 3: Evaluate Baselines =====
    print("\n[3/4] Evaluating Baseline Policies...")
    print("-" * 60)
    
    baselines = {
        'Fixed Batch (16)': FixedBatchPolicy(batch_threshold=16),
        'Fixed Wait (1.0s)': FixedWaitPolicy(wait_threshold=1.0)
    }
    
    baseline_results = {}
    for name, policy in baselines.items():
        results = evaluate_baseline(policy, env, num_episodes=10)
        baseline_results[name] = results
        
        print(f"\n{name}:")
        print(f"  Mean Reward: {results['mean_reward']:.3f}")
        print(f"  Mean Batch Size: {results['mean_batch_size']:.2f}")
        print(f"  Mean Wait Time: {results['mean_wait_time']:.3f}s")
    
    # ===== STEP 4: Comparison =====
    print("\n[4/4] Comparison Summary")
    print("=" * 60)
    
    print(f"\nReward Comparison:")
    print(f"  DQN Agent:        {dqn_results['mean_reward']:>8.3f} ⭐")
    for name, results in baseline_results.items():
        print(f"  {name:20s} {results['mean_reward']:>8.3f}")
    
    print(f"\nBatch Size Comparison:")
    print(f"  DQN Agent:        {dqn_results['mean_batch_size']:>8.2f}")
    for name, results in baseline_results.items():
        print(f"  {name:20s} {results['mean_batch_size']:>8.2f}")
    
    print(f"\nWait Time Comparison:")
    print(f"  DQN Agent:        {dqn_results['mean_wait_time']:>8.3f}s")
    for name, results in baseline_results.items():
        print(f"  {name:20s} {results['mean_wait_time']:>8.3f}s")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. View training curves: python main.py --mode plot")
    print("  2. Full comparison: python main.py --mode compare")
    print("  3. Try different traffic: python main.py --mode train --traffic bursty")
    print("  4. Tune hyperparameters: python main.py --mode train --alpha 1.5 --beta 2.5")
    print()


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    quick_demo()
