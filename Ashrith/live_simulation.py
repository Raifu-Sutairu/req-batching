import os
import sys
import time
import collections
import numpy as np
import matplotlib.pyplot as plt

# Ensure that the root directory is accessible so env can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ashrith.env import BatchingEnv
from Ashrith.predictive_dynaq_agent import PredictiveDynaQAgent

def run_live_simulation():
    # Setup environment
    MAX_RATE_MULTIPLIER = 3.0
    BASE_RATE = 5.0
    MAX_RATE = BASE_RATE * MAX_RATE_MULTIPLIER
    
    env = BatchingEnv(
        traffic_pattern='time_varying', # Uses time_varying to make pred vs obs interesting
        base_arrival_rate=BASE_RATE,
        max_steps=int(1e9),  # Run practically forever
        seed=42
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Setup agent
    agent = PredictiveDynaQAgent(action_dim=action_dim)
    ckpt_path = os.path.join('Ashrith', 'checkpoints', 'predictive_dynaq_best.npy')
    
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        print(f"Loaded Predictive DynaQ checkpoint from {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}. Running with untrained agent.")

    agent.start_episode()
    state, _ = env.reset()

    # Enable interactive mode for matplotlib
    plt.ion()
    
    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    try:
        fig.canvas.manager.set_window_title("Live Simulation: Predictive Dyna-Q")
    except Exception:
        pass
        
    ax_wait, ax_p95, ax_batch, ax_pred = axes.flatten()

    # Use deques to store the plotted data over time (sliding window)
    MAX_HISTORY = 300
    steps = collections.deque(maxlen=MAX_HISTORY)
    wait_times = collections.deque(maxlen=MAX_HISTORY)
    p95_waits = collections.deque(maxlen=MAX_HISTORY)
    batch_sizes = collections.deque(maxlen=MAX_HISTORY)
    pred_rates = collections.deque(maxlen=MAX_HISTORY)
    obs_rates = collections.deque(maxlen=MAX_HISTORY)

    # Initialize empty plot lines
    line_wait, = ax_wait.plot([], [], color='#e74c3c', lw=2)
    ax_wait.set_title('Avg Wait Time', fontsize=12)
    ax_wait.set_xlabel('Step')
    ax_wait.set_ylabel('Seconds')

    line_p95, = ax_p95.plot([], [], color='#c0392b', lw=2)
    ax_p95.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, label="SLO (1.0s)")
    ax_p95.set_title('p95 Wait Time', fontsize=12)
    ax_p95.set_xlabel('Step')
    ax_p95.set_ylabel('Seconds')
    ax_p95.legend(loc='upper left')

    line_batch, = ax_batch.plot([], [], color='#3498db', lw=2)
    ax_batch.set_title('Avg Batch Size', fontsize=12)
    ax_batch.set_xlabel('Step')
    ax_batch.set_ylabel('Requests / Batch')

    line_pred, = ax_pred.plot([], [], color='#f39c12', lw=2, label="Predicted Rate")
    line_obs, = ax_pred.plot([], [], color='#2ecc71', lw=1.5, alpha=0.7, label="Observed Rate")
    ax_pred.set_title('Predicted vs Observed Demand', fontsize=12)
    ax_pred.set_xlabel('Step')
    ax_pred.set_ylabel('Req / s')
    ax_pred.legend(loc='upper right')

    for ax in axes.flatten():
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Leave room for suptitle
    plt.show(block=False)

    done = False
    truncated = False
    step = 0

    print("Starting simulation in 2 seconds...")
    plt.pause(2.0)  # brief pause before starting

    while not (done or truncated) and plt.fignum_exists(fig.number):
        # Determine predicted arrival rate
        # state[4] is the normalized observed rate
        normalized_obs_rate = float(state[4])
        normalized_pred_rate = agent.predictor.predict(normalized_obs_rate)
        pred_rate_raw = normalized_pred_rate * MAX_RATE

        # Let agent select an action based on policy (no exploration)
        action = agent.select_action(state, explore=False)
        action_str = "SKIP" if action == 1 else "WAIT"
        
        next_state, reward, done, truncated, info = env.step(action)
        
        # Observe the transition (eval: Predictive Dyna-Q only updates its predictor)
        agent.observe(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=(done or truncated),
            train=False
        )
        
        state = next_state
        step += 1

        # Record metrics
        metrics = env.get_metrics()
        
        steps.append(step)
        wait_times.append(metrics['avg_wait_time'])
        p95_waits.append(metrics.get('p95_wait_time', 0.0))
        batch_sizes.append(metrics['avg_batch_size'])
        pred_rates.append(pred_rate_raw)
        obs_rates.append(info['arrival_rate'])

        # Update plots every N steps to keep drawing smooth
        if step % 2 == 0:
            # Update Title with Live Text Counters
            total_reqs = metrics.get('total_requests', 0)
            total_batches = metrics.get('total_batches', 0)
            throughput = metrics.get('throughput', 0.0)
            slo_viol = metrics.get('slo_violation_rate', 0.0)
            
            title_text = (
                f"LIVE SIMULATION: Predictive Dyna-Q\n"
                f"Reqs Served: {total_reqs}  |  Batches Sent: {total_batches}  |  "
                f"Throughput: {throughput:.1f} r/s  |  "
                f"SLO Violations: {slo_viol:.1f}%  |  "
                f"Action: {action_str}"
            )
            fig.suptitle(title_text, fontsize=14, fontweight='bold', color='black')

            # Update data
            list_steps = list(steps)
            line_wait.set_data(list_steps, list(wait_times))
            ax_wait.relim()
            ax_wait.autoscale_view()

            line_p95.set_data(list_steps, list(p95_waits))
            ax_p95.relim()
            ax_p95.autoscale_view()

            line_batch.set_data(list_steps, list(batch_sizes))
            ax_batch.relim()
            ax_batch.autoscale_view()

            line_pred.set_data(list_steps, list(pred_rates))
            line_obs.set_data(list_steps, list(obs_rates))
            ax_pred.relim()
            ax_pred.autoscale_view()

            # Refresh the canvas
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Short pause to control simulation playback speed (approx 50 steps/second)
            time.sleep(0.02)

    print("Simulation stopped.")
    
    # Disable interactive mode and block the script so user can inspect the final plot
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    run_live_simulation()
