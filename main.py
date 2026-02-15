"""
RL-Based Intelligent Request Batching System

Main entry point for training and evaluation.
"""

import argparse
import os
from training import TrainingConfig, train_agent, evaluate_agent
from env import BatchingEnv
from agent import DQNAgent
from visualization import plot_training_curves, compare_policies


def main():
    parser = argparse.ArgumentParser(
        description='RL-Based Request Batching System'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'eval', 'compare', 'plot'],
        help='Mode: train, eval, compare, or plot'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/dqn_best.pth',
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=500,
        help='Number of training episodes'
    )
    
    parser.add_argument(
        '--traffic',
        type=str,
        default='poisson',
        choices=['poisson', 'bursty', 'time_varying'],
        help='Traffic pattern'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Batch efficiency weight in reward'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=2.0,
        help='Latency penalty weight in reward'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("\n" + "="*60)
        print("Training DQN Agent for Request Batching")
        print("="*60)
        
        config = TrainingConfig(
            num_episodes=args.episodes,
            traffic_pattern=args.traffic,
            alpha=args.alpha,
            beta=args.beta,
            learning_rate=args.lr,
            gamma=args.gamma,
            seed=args.seed
        )
        
        agent, logs = train_agent(config)
        
        # Plot training curves
        print("\nGenerating training plots...")
        plot_training_curves('logs/training_logs.json', 
                            save_path='results/training_curves.png')
        
    elif args.mode == 'eval':
        print("\n" + "="*60)
        print("Evaluating Trained DQN Agent")
        print("="*60)
        
        # Create environment
        env = BatchingEnv(traffic_pattern=args.traffic, seed=args.seed)
        
        # Load agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        
        if os.path.exists(args.model):
            agent.load(args.model)
            print(f"Loaded model from {args.model}\n")
        else:
            print(f"Error: Model not found at {args.model}")
            return
        
        # Evaluate
        results = evaluate_agent(agent, env, num_episodes=20)
        
        print("\nEvaluation Results:")
        print("-" * 50)
        print(f"  Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"  Mean Batch Size: {results['mean_batch_size']:.2f}")
        print(f"  Mean Wait Time: {results['mean_wait_time']:.3f}s")
        print(f"  Mean Throughput: {results['mean_throughput']:.2f} req/s")
        print("-" * 50)
        
    elif args.mode == 'compare':
        print("\n" + "="*60)
        print("Comparing DQN with Baseline Policies")
        print("="*60)
        
        os.makedirs('results', exist_ok=True)
        
        compare_policies(
            model_path=args.model,
            traffic_pattern=args.traffic,
            num_episodes=20,
            save_path=f'results/comparison_{args.traffic}.png'
        )
        
    elif args.mode == 'plot':
        print("\n" + "="*60)
        print("Plotting Training Curves")
        print("="*60)
        
        if os.path.exists('logs/training_logs.json'):
            os.makedirs('results', exist_ok=True)
            plot_training_curves('logs/training_logs.json',
                               save_path='results/training_curves.png')
        else:
            print("Error: No training logs found at logs/training_logs.json")


if __name__ == '__main__':
    main()
