"""
Smart Delivery Environment - Custom Grid-based RL Environment
Author: Student Submission
Description: A 6x6 grid world where an agent must pick up a package, 
manage battery, and deliver it to a target location.
"""

import numpy as np
import matplotlib.pyplot as plt


class SmartDeliveryEnv:
    """Custom Environment for Smart Delivery task"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(SmartDeliveryEnv, self).__init__()
        
        # Grid dimensions
        self.grid_size = 6
        
        # Define fixed positions
        self.agent_start = (0, 0)
        self.package_loc = (2, 4)
        self.delivery_loc = (5, 5)
        self.charging_station = (3, 1)
        self.obstacles = [(1, 1), (1, 2), (2, 2), (4, 3), (4, 4)]
        
        # Action space: 7 discrete actions
        self.n_actions = 7
        
        # State space will be encoded as integer
        # Total states = 36 positions × 2 package states × 2 battery states = 144
        self.n_states = self.grid_size * self.grid_size * 2 * 2
        
        # Initialize state
        self.agent_pos = None
        self.package_picked = None
        self.battery_level = None
        self.steps = 0
        self.max_steps = 100
        
        # Battery management
        self.initial_battery = 50
        self.battery_consumption_per_step = 2
        self.battery_threshold = 10  # Low battery threshold
        
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        self.agent_pos = list(self.agent_start)
        self.package_picked = 0  # 0 = not picked, 1 = picked
        self.battery_level = self.initial_battery
        self.steps = 0
        return self._get_state()
    
    def _get_state(self):
        """Encode state as a single integer"""
        # State encoding: position (0-35) + package_status (0-1) + battery_status (0-1)
        row, col = self.agent_pos
        position_idx = row * self.grid_size + col
        battery_status = 1 if self.battery_level > self.battery_threshold else 0
        
        # State = position * 4 + package_picked * 2 + battery_status
        state = position_idx * 4 + self.package_picked * 2 + battery_status
        return state
    
    def _is_valid_position(self, pos):
        """Check if position is valid (within bounds and not obstacle)"""
        row, col = pos
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return False
        if tuple(pos) in self.obstacles:
            return False
        return True
    
    def step(self, action):
        """Execute one step in the environment"""
        self.steps += 1
        reward = -1  # Step penalty
        done = False
        info = {}
        
        # Store old position for collision detection
        old_pos = self.agent_pos.copy()
        
        # Action execution
        if action == 0:  # Move Left
            self.agent_pos[1] -= 1
        elif action == 1:  # Move Right
            self.agent_pos[1] += 1
        elif action == 2:  # Move Up
            self.agent_pos[0] -= 1
        elif action == 3:  # Move Down
            self.agent_pos[0] += 1
        elif action == 4:  # Pick Package
            if tuple(self.agent_pos) == self.package_loc and self.package_picked == 0:
                self.package_picked = 1
                reward = 10  # Successful pickup
                info['event'] = 'package_picked'
            else:
                reward = -1  # Just step penalty, no special penalty
        elif action == 5:  # Recharge Battery
            if tuple(self.agent_pos) == self.charging_station:
                old_battery = self.battery_level
                self.battery_level = self.initial_battery
                # Only reward if battery was actually low
                if old_battery < self.battery_threshold * 2:
                    reward = 5  # Successful recharge when needed
                else:
                    reward = -1  # Penalty for unnecessary recharge
                info['event'] = 'recharged'
            else:
                reward = -1  # Just step penalty
        elif action == 6:  # Drop Package
            if tuple(self.agent_pos) == self.delivery_loc and self.package_picked == 1:
                reward = 50  # Successful delivery
                done = True
                info['event'] = 'delivery_success'
            else:
                reward = -10  # Wrong drop
                info['event'] = 'wrong_drop'
        
        # Check if new position is valid (for movement actions)
        if action in [0, 1, 2, 3]:
            if not self._is_valid_position(self.agent_pos):
                self.agent_pos = old_pos  # Revert to old position
                reward = -5  # Obstacle/wall penalty
                info['event'] = 'collision'
            else:
                # Consume battery for valid movement
                self.battery_level -= self.battery_consumption_per_step
        
        # Check battery exhaustion
        if self.battery_level <= 0:
            reward = -20
            done = True
            info['event'] = 'battery_exhausted'
        
        # Check max steps
        if self.steps >= self.max_steps:
            reward = -20
            done = True
            info['event'] = 'max_steps_reached'
        
        next_state = self._get_state()
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        
        # Place obstacles
        for obs in self.obstacles:
            grid[obs] = 'X'
        
        # Place package if not picked
        if self.package_picked == 0:
            grid[self.package_loc] = 'P'
        
        # Place delivery location
        grid[self.delivery_loc] = 'D'
        
        # Place charging station
        grid[self.charging_station] = 'C'
        
        # Place agent (overrides other symbols)
        grid[tuple(self.agent_pos)] = 'A'
        
        print("\n" + "="*30)
        print(f"Step: {self.steps} | Battery: {self.battery_level} | Package: {'✓' if self.package_picked else '✗'}")
        print("="*30)
        for row in grid:
            print(' '.join(row))
        print("="*30)
    
    def close(self):
        pass


class QLearningAgent:
    """Q-Learning Agent for Smart Delivery Environment"""
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
    
    def get_action(self, state, training=True):
        """Select action using ε-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule"""
        # Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Update Q-value
        self.q_table[state, action] = current_q + self.alpha * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(env, agent, n_episodes=500, verbose=True):
    """Train the Q-learning agent"""
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select and execute action
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track metrics
        episode_rewards.append(total_reward)
        episode_steps.append(env.steps)
        
        if info.get('event') == 'delivery_success':
            success_count += 1
        
        # Print progress
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Success Rate: {success_count}/{episode + 1}")
    
    return episode_rewards, episode_steps


def test_agent(env, agent, n_episodes=10, render=True):
    """Test the trained agent"""
    test_rewards = []
    success_count = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        if render and episode == 0:
            print(f"\n{'='*50}")
            print(f"Testing Episode {episode + 1}")
            print(f"{'='*50}")
            env.render()
        
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            
            if render and episode == 0:
                action_names = ['Left', 'Right', 'Up', 'Down', 'Pick', 'Recharge', 'Drop']
                print(f"\nAction: {action_names[action]}")
                env.render()
        
        test_rewards.append(total_reward)
        if info.get('event') == 'delivery_success':
            success_count += 1
    
    return test_rewards, success_count


def plot_results(episode_rewards, save_path='/mnt/user-data/outputs/training_rewards.png'):
    """Plot training results"""
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Raw Rewards')
    
    # Plot moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, 
                label=f'{window}-Episode Moving Average', linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Training Progress: Episode vs Total Reward', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot reward distribution
    plt.subplot(1, 2, 2)
    plt.hist(episode_rewards, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Total Reward', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Reward Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    """Main training and testing pipeline"""
    print("="*60)
    print("SMART DELIVERY ENVIRONMENT - Q-LEARNING IMPLEMENTATION")
    print("="*60)
    
    # Create environment
    env = SmartDeliveryEnv()
    print("\n✓ Environment created successfully")
    print(f"  Grid size: {env.grid_size}x{env.grid_size}")
    print(f"  Agent start: {env.agent_start}")
    print(f"  Package location: {env.package_loc}")
    print(f"  Delivery location: {env.delivery_loc}")
    print(f"  Charging station: {env.charging_station}")
    print(f"  Obstacles: {env.obstacles}")
    
    # Create agent with specified hyperparameters
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    print("\n✓ Q-Learning agent initialized")
    print(f"  Learning rate (α): {agent.alpha}")
    print(f"  Discount factor (γ): {agent.gamma}")
    print(f"  Initial epsilon (ε): {agent.epsilon}")
    print(f"  Minimum epsilon: {agent.epsilon_min}")
    print(f"  Epsilon decay: {agent.epsilon_decay}")
    
    # Train agent
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    n_episodes = 1000
    episode_rewards, episode_steps = train_agent(env, agent, n_episodes=n_episodes)
    
    # Plot results
    plot_results(episode_rewards)
    
    # Print training statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print(f"Total episodes: {n_episodes}")
    print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best reward: {np.max(episode_rewards):.2f}")
    print(f"Worst reward: {np.min(episode_rewards):.2f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    
    # Test agent
    print("\n" + "="*60)
    print("TESTING PHASE")
    print("="*60)
    n_test_episodes = 10
    test_rewards, success_count = test_agent(env, agent, n_episodes=n_test_episodes, render=True)
    
    # Print test statistics
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Test episodes: {n_test_episodes}")
    print(f"Successful deliveries: {success_count}/{n_test_episodes}")
    print(f"Success rate: {success_count/n_test_episodes*100:.1f}%")
    print(f"Average test reward: {np.mean(test_rewards):.2f}")
    print(f"Test rewards: {test_rewards}")
    
    # Analyze learned policy
    print("\n" + "="*60)
    print("LEARNED POLICY ANALYSIS")
    print("="*60)
    
    # Count non-zero Q-values
    non_zero_q = np.count_nonzero(agent.q_table)
    total_q = agent.q_table.size
    print(f"Explored state-action pairs: {non_zero_q}/{total_q} ({non_zero_q/total_q*100:.1f}%)")
    
    # Find best actions for key states
    print("\nOptimal actions for key states:")
    action_names = ['Left', 'Right', 'Up', 'Down', 'Pick Package', 'Recharge', 'Drop Package']
    
    # Sample important states
    key_positions = [
        (env.agent_start, "Start position"),
        (env.package_loc, "Package location"),
        (env.charging_station, "Charging station"),
        (env.delivery_loc, "Delivery location")
    ]
    
    for pos, desc in key_positions:
        row, col = pos
        position_idx = row * env.grid_size + col
        
        # State: no package, sufficient battery
        state = position_idx * 4 + 0 * 2 + 1
        best_action = np.argmax(agent.q_table[state])
        print(f"  {desc} (no package): {action_names[best_action]}")
        
        # State: package picked, sufficient battery
        state = position_idx * 4 + 1 * 2 + 1
        best_action = np.argmax(agent.q_table[state])
        print(f"  {desc} (with package): {action_names[best_action]}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return env, agent, episode_rewards, test_rewards


if __name__ == "__main__":
    env, agent, train_rewards, test_rewards = main()