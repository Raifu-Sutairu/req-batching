"""
Training Loop for DQN Agent

Main training loop with logging and checkpointing.
"""

import os
import json
import numpy as np
from tqdm import tqdm
from env import BatchingEnv
from agent import DQNAgent
from .config import TrainingConfig


def train_agent(config: TrainingConfig):
    """
    Train DQN agent on batching environment.
    
    Args:
        config: Training configuration
    """
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Create environment
    env = BatchingEnv(
        max_batch_size=config.max_batch_size,
        max_wait_time=config.max_wait_time,
        max_queue_length=config.max_queue_length,
        traffic_pattern=config.traffic_pattern,
        base_arrival_rate=config.base_arrival_rate,
        alpha=config.alpha,
        beta=config.beta,
        max_steps=config.episode_steps,
        dt=config.dt,
        seed=config.seed
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        buffer_capacity=config.buffer_capacity,
        batch_size=config.batch_size,
        target_update_freq=config.target_update_freq,
        device=config.device
    )
    
    # Training logs
    episode_rewards = []
    episode_losses = []
    episode_metrics = []
    
    best_reward = -float('inf')
    
    print(f"Training DQN Agent on {agent.device}")
    print(f"Traffic Pattern: {config.traffic_pattern}")
    print(f"Episodes: {config.num_episodes}")
    print("-" * 60)
    
    # Training loop
    for episode in tqdm(range(config.num_episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(state, explore=True)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done or truncated)
            
            # Train
            loss = agent.train_step()
            if loss > 0:
                episode_loss.append(loss)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Update target network
        if (episode + 1) % agent.target_update_freq == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Log metrics
        metrics = env.get_metrics()
        episode_rewards.append(episode_reward)
        episode_losses.append(np.mean(episode_loss) if episode_loss else 0)
        episode_metrics.append(metrics)
        
        # Print progress
        if (episode + 1) % config.eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.eval_interval:])
            avg_loss = np.mean(episode_losses[-config.eval_interval:])
            
            print(f"\nEpisode {episode + 1}/{config.num_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Batch Size: {metrics['avg_batch_size']:.2f}")
            print(f"  Avg Wait Time: {metrics['avg_wait_time']:.3f}s")
            print(f"  Throughput: {metrics['throughput']:.2f} req/s")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(config.checkpoint_dir, 'dqn_best.pth'))
        
        # Save checkpoint
        if (episode + 1) % config.save_interval == 0:
            agent.save(os.path.join(config.checkpoint_dir, f'dqn_ep{episode + 1}.pth'))
    
    # Save final model
    agent.save(os.path.join(config.checkpoint_dir, 'dqn_final.pth'))
    
    # Save training logs
    logs = {
        'rewards': episode_rewards,
        'losses': episode_losses,
        'metrics': episode_metrics,
        'config': config.__dict__
    }
    
    with open(os.path.join(config.log_dir, 'training_logs.json'), 'w') as f:
        json.dump(logs, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Episode Reward: {best_reward:.3f}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print(f"Logs saved to: {config.log_dir}")
    print("=" * 60)
    
    return agent, logs
