"""
Unified Training Script for Policy-Based RL Agents

Trains REINFORCE, A2C, or PPO on the BatchingEnv.
Usage:
    python -m Ashrith.train_agents --agent reinforce
    python -m Ashrith.train_agents --agent a2c
    python -m Ashrith.train_agents --agent ppo
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import BatchingEnv
from Ashrith.reinforce_agent import REINFORCEAgent
from Ashrith.a2c_agent import A2CAgent
from Ashrith.ppo_agent import PPOAgent


def train_reinforce(env, agent, num_episodes, eval_interval=50):
    """
    Train REINFORCE agent.
    
    REINFORCE collects a full episode, computes returns,
    then does a single update. This is Monte Carlo style.
    """
    episode_rewards = []
    episode_losses = []
    best_reward = -float('inf')
    
    for episode in tqdm(range(num_episodes), desc="REINFORCE"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        # Collect full episode
        while not (done or truncated):
            action = agent.select_action(state, explore=True)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_reward(reward)
            state = next_state
            episode_reward += reward
        
        # Update at end of episode (Monte Carlo)
        loss_info = agent.update()
        
        episode_rewards.append(episode_reward)
        episode_losses.append(loss_info['policy_loss'])
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join('Ashrith', 'checkpoints', 'reinforce_best.pth'))
        
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            metrics = env.get_metrics()
            print(f"\n  Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Policy Loss: {loss_info['policy_loss']:.4f} | "
                  f"Entropy: {loss_info['avg_entropy']:.3f} | "
                  f"Batch Size: {metrics['avg_batch_size']:.1f}")
    
    return episode_rewards, episode_losses


def train_a2c(env, agent, num_episodes, eval_interval=50):
    """
    Train A2C agent.
    
    A2C updates every N steps (not waiting for full episode).
    This is a TD-style method with bootstrapping.
    """
    episode_rewards = []
    episode_losses = []
    best_reward = -float('inf')
    
    for episode in tqdm(range(num_episodes), desc="A2C"):
        state, _ = env.reset()
        episode_reward = 0
        ep_losses = []
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state, explore=True)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_reward(reward)
            state = next_state
            episode_reward += reward
            
            # Update every N steps or at end of episode
            if agent.should_update() or done or truncated:
                loss_info = agent.update(
                    next_state=next_state, 
                    done=(done or truncated)
                )
                ep_losses.append(loss_info['total_loss'])
        
        agent.end_episode()
        episode_rewards.append(episode_reward)
        episode_losses.append(np.mean(ep_losses) if ep_losses else 0)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join('Ashrith', 'checkpoints', 'a2c_best.pth'))
        
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            metrics = env.get_metrics()
            print(f"\n  Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Loss: {episode_losses[-1]:.4f} | "
                  f"Batch Size: {metrics['avg_batch_size']:.1f}")
    
    return episode_rewards, episode_losses


def train_ppo(env, agent, num_episodes, eval_interval=50):
    """
    Train PPO agent.
    
    PPO collects a rollout of T steps, then does K epochs
    of mini-batch updates on the collected data.
    """
    episode_rewards = []
    episode_losses = []
    best_reward = -float('inf')
    
    for episode in tqdm(range(num_episodes), desc="PPO"):
        state, _ = env.reset()
        episode_reward = 0
        ep_losses = []
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state, explore=True)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition(reward, done=(done or truncated))
            state = next_state
            episode_reward += reward
            
            # Update when rollout buffer is full
            if agent.should_update():
                loss_info = agent.update(next_state=next_state)
                ep_losses.append(loss_info['policy_loss'])
        
        # Handle remaining transitions at end of episode
        if len(agent.buffer) > 0:
            loss_info = agent.update(next_state=next_state)
            ep_losses.append(loss_info['policy_loss'])
        
        agent.end_episode()
        episode_rewards.append(episode_reward)
        episode_losses.append(np.mean(ep_losses) if ep_losses else 0)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join('Ashrith', 'checkpoints', 'ppo_best.pth'))
        
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            metrics = env.get_metrics()
            print(f"\n  Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Policy Loss: {episode_losses[-1]:.4f} | "
                  f"Batch Size: {metrics['avg_batch_size']:.1f}")
    
    return episode_rewards, episode_losses


def create_agent(agent_type, state_dim, action_dim, device='cpu'):
    """
    Factory function to create the specified agent.
    
    Note: We default to CPU because MPS has too much overhead for
    these small networks (6→128→64→2). CPU is actually faster.
    """
    if agent_type == 'reinforce':
        return REINFORCEAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.001,
            baseline_lr=0.001,
            gamma=0.99,
            entropy_coeff=0.01,
            device=device
        )
    elif agent_type == 'a2c':
        return A2CAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.001,
            gamma=0.99,
            entropy_coeff=0.01,
            value_coeff=0.5,
            n_steps=5,
            device=device
        )
    elif agent_type == 'ppo':
        return PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.0003,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coeff=0.01,
            n_epochs=4,
            mini_batch_size=64,
            rollout_length=128,
            device=device
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train(agent_type='reinforce', num_episodes=300, traffic_pattern='poisson', seed=42):
    """
    Main training function.
    
    Args:
        agent_type: 'reinforce', 'a2c', or 'ppo'
        num_episodes: Number of training episodes
        traffic_pattern: 'poisson', 'bursty', or 'time_varying'
        seed: Random seed
        
    Returns:
        (agent, rewards, losses)
    """
    # Create directories
    os.makedirs(os.path.join('Ashrith', 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join('Ashrith', 'logs'), exist_ok=True)
    
    # Create environment
    env = BatchingEnv(
        max_batch_size=32,
        max_wait_time=2.0,
        max_queue_length=100,
        traffic_pattern=traffic_pattern,
        base_arrival_rate=5.0,
        alpha=1.0,
        beta=2.0,
        max_steps=1000,
        dt=0.1,
        seed=seed
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = create_agent(agent_type, state_dim, action_dim)
    
    print(f"\n{'='*60}")
    print(f"Training {agent_type.upper()} Agent on {agent.device}")
    print(f"Traffic Pattern: {traffic_pattern}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*60}")
    
    # Train
    train_fn = {
        'reinforce': train_reinforce,
        'a2c': train_a2c,
        'ppo': train_ppo,
    }[agent_type]
    
    rewards, losses = train_fn(env, agent, num_episodes)
    
    # Save final model
    agent.save(os.path.join('Ashrith', 'checkpoints', f'{agent_type}_final.pth'))
    
    # Save logs
    logs = {
        'agent_type': agent_type,
        'rewards': rewards,
        'losses': losses,
        'traffic_pattern': traffic_pattern,
        'num_episodes': num_episodes
    }
    with open(os.path.join('Ashrith', 'logs', f'{agent_type}_logs.json'), 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"{agent_type.upper()} Training Complete!")
    print(f"Best Reward: {max(rewards):.3f}")
    print(f"Final Avg (last 50): {np.mean(rewards[-50:]):.3f}")
    print(f"Checkpoints: Ashrith/checkpoints/")
    print(f"{'='*60}")
    
    return agent, rewards, losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Policy-Based RL Agents')
    parser.add_argument('--agent', type=str, default='reinforce',
                        choices=['reinforce', 'a2c', 'ppo'],
                        help='Agent type to train')
    parser.add_argument('--episodes', type=int, default=300,
                        help='Number of training episodes')
    parser.add_argument('--traffic', type=str, default='poisson',
                        choices=['poisson', 'bursty', 'time_varying'],
                        help='Traffic pattern')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    train(args.agent, args.episodes, args.traffic, args.seed)
