"""
Training Curve Plotting

Visualize training progress over episodes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(log_file: str, save_path: str = None):
    """
    Plot training curves from log file.
    
    Args:
        log_file: Path to training_logs.json
        save_path: Path to save figure (optional)
    """
    # Load logs
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    rewards = logs['rewards']
    losses = logs['losses']
    metrics = logs['metrics']
    
    # Extract metric arrays
    batch_sizes = [m['avg_batch_size'] for m in metrics]
    wait_times = [m['avg_wait_time'] for m in metrics]
    throughputs = [m['throughput'] for m in metrics]
    
    # Compute moving averages
    window = min(20, len(rewards) // 10)  # Adaptive window size
    rewards_ma = moving_average(rewards, window)
    losses_ma = moving_average(losses, window)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DQN Training Progress', fontsize=16, fontweight='bold')
    
    episodes = range(len(rewards))
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
    axes[0, 0].plot(episodes, rewards_ma, color='blue', linewidth=2, label=f'MA({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    axes[0, 1].plot(episodes, losses, alpha=0.3, color='red', label='Raw')
    axes[0, 1].plot(episodes, losses_ma, color='red', linewidth=2, label=f'MA({window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average Batch Size
    axes[0, 2].plot(episodes, batch_sizes, color='green', linewidth=1.5)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Avg Batch Size')
    axes[0, 2].set_title('Average Batch Size per Episode')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Average Wait Time
    axes[1, 0].plot(episodes, wait_times, color='orange', linewidth=1.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Avg Wait Time (s)')
    axes[1, 0].set_title('Average Wait Time per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Throughput
    axes[1, 1].plot(episodes, throughputs, color='purple', linewidth=1.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Throughput (req/s)')
    axes[1, 1].set_title('Throughput per Episode')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Latency vs Batch Size Tradeoff
    axes[1, 2].scatter(batch_sizes, wait_times, alpha=0.5, c=episodes, cmap='viridis', s=20)
    axes[1, 2].set_xlabel('Avg Batch Size')
    axes[1, 2].set_ylabel('Avg Wait Time (s)')
    axes[1, 2].set_title('Latency-Batch Tradeoff Evolution')
    cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
    cbar.set_label('Episode')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_metrics(metrics_dict: dict, title: str = "Metrics", save_path: str = None):
    """
    Plot specific metrics over episodes.
    
    Args:
        metrics_dict: Dictionary of metric_name -> list of values
        title: Plot title
        save_path: Path to save figure
    """
    sns.set_style("whitegrid")
    
    num_metrics = len(metrics_dict)
    cols = 2
    rows = (num_metrics + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics_dict.items()):
        episodes = range(len(values))
        axes[idx].plot(episodes, values, linewidth=2)
        axes[idx].set_xlabel('Episode')
        axes[idx].set_ylabel(metric_name)
        axes[idx].set_title(metric_name)
        axes[idx].grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    
    plt.show()


def moving_average(data, window):
    """Compute moving average with same length as input."""
    if len(data) < window:
        return np.array(data)
    
    # Compute moving average
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Pad beginning with NaN to match original length
    pad_size = len(data) - len(ma)
    padded = np.concatenate([np.full(pad_size, np.nan), ma])
    
    return padded
