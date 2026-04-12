"""
Request Batching Environment

Custom Gym environment for learning intelligent batching policies.
Balances batch efficiency (larger batches) with latency (waiting time).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any
from .traffic_generator import TrafficGenerator


class BatchingEnv(gym.Env):
    """
    Environment for request batching optimization.
    
    State: [batch_size, wait_time, queue_length, time_since_last_skip, 
            arrival_rate, system_load]
    Actions: 0 = WAIT, 1 = SKIP (send batch)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 2.0,
        max_queue_length: int = 100,
        traffic_pattern: str = 'poisson',
        base_arrival_rate: float = 5.0,
        alpha: float = 1.0,  # batch efficiency weight
        beta: float = 2.0,   # latency penalty weight
        max_steps: int = 1000,
        dt: float = 0.1,     # time step (seconds)
        seed: int = None
    ):
        """
        Initialize batching environment.
        
        Args:
            max_batch_size: Maximum requests per batch
            max_wait_time: Maximum wait threshold (seconds)
            max_queue_length: Maximum queue capacity
            traffic_pattern: Traffic generation pattern
            base_arrival_rate: Base request arrival rate (req/sec)
            alpha: Reward weight for batch efficiency
            beta: Reward weight for latency penalty
            max_steps: Maximum steps per episode
            dt: Time step duration (seconds)
            seed: Random seed
        """
        super().__init__()
        
        # Environment parameters
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_length = max_queue_length
        self.alpha = alpha
        self.beta = beta
        self.max_steps = max_steps
        self.dt = dt
        
        # State space: [batch_size, wait_time, queue_length, 
        #               time_since_last_skip, arrival_rate, system_load]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 0 = WAIT, 1 = SKIP
        self.action_space = spaces.Discrete(2)
        
        # Traffic generator
        self.traffic_gen = TrafficGenerator(
            pattern=traffic_pattern,
            base_rate=base_arrival_rate,
            seed=seed
        )
        
        # Initialize state
        self.rng = np.random.RandomState(seed)
        self.reset()
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset traffic generator
        self.traffic_gen.reset()
        
        # Internal state
        self.current_batch = []
        self.wait_queue = []
        self.batch_start_time = 0
        self.current_time = 0
        self.time_since_last_skip = 0
        self.step_count = 0
        
        # Metrics
        self.total_requests_processed = 0
        self.total_batches_sent = 0
        self.total_wait_time = 0
        self.all_wait_times = []
        self.queue_length_sum = 0
        self.queue_length_samples = 0
        
        # Arrival rate tracking (exponential moving average)
        self.arrival_rate_ema = self.traffic_gen.base_rate
        self.ema_alpha = 0.1
        
        # System load (simulated)
        self.system_load = 0.5
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0 = WAIT, 1 = SKIP
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Generate new arrivals
        num_arrivals = self.traffic_gen.generate_arrivals(self.dt)
        
        # Update arrival rate EMA
        current_rate = num_arrivals / self.dt
        self.arrival_rate_ema = (self.ema_alpha * current_rate + 
                                 (1 - self.ema_alpha) * self.arrival_rate_ema)
        
        # Add new requests to queue
        for _ in range(num_arrivals):
            if len(self.wait_queue) < self.max_queue_length:
                self.wait_queue.append(self.current_time)
        
        # Move requests from queue to current batch
        self._fill_batch()
        
        # Update time
        self.current_time += self.dt
        self.time_since_last_skip += self.dt
        self.step_count += 1
        
        # Execute action and compute reward
        reward = 0
        batch_sent = False
        
        if action == 1 or self._should_force_skip():  # SKIP
            reward = self._send_batch()
            batch_sent = True
        else:  # WAIT
            reward = self._compute_wait_reward()
        
        # Check termination
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        # Update system load (simple simulation)
        self.system_load = min(1.0, len(self.wait_queue) / self.max_queue_length)
        
        # Track queue metrics
        self.queue_length_sum += len(self.wait_queue)
        self.queue_length_samples += 1
        
        info = {
            'batch_size': len(self.current_batch),
            'queue_length': len(self.wait_queue),
            'wait_time': self._get_current_wait_time(),
            'batch_sent': batch_sent,
            'total_batches': self.total_batches_sent,
            'total_requests': self.total_requests_processed,
            'arrival_rate': self.arrival_rate_ema
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _fill_batch(self):
        """Move requests from queue to current batch."""
        while len(self.current_batch) < self.max_batch_size and self.wait_queue:
            arrival_time = self.wait_queue.pop(0)
            if len(self.current_batch) == 0:
                self.batch_start_time = self.current_time
            self.current_batch.append(arrival_time)
    
    def _should_force_skip(self) -> bool:
        """Force skip if batch is full or wait time exceeded."""
        if len(self.current_batch) >= self.max_batch_size:
            return True
        if self._get_current_wait_time() >= self.max_wait_time:
            return True
        return False
    
    def _send_batch(self) -> float:
        """Send current batch and compute reward."""
        if len(self.current_batch) == 0:
            # No batch to send, small penalty
            return -0.1
        
        batch_size = len(self.current_batch)
        wait_times = [self.current_time - t for t in self.current_batch]
        self.all_wait_times.extend(wait_times)
        avg_wait_time = np.mean(wait_times)
        
        # Batch efficiency reward (normalized)
        batch_reward = self.alpha * (batch_size / self.max_batch_size)
        
        # Latency penalty (quadratic to heavily penalize long waits)
        latency_penalty = self.beta * (avg_wait_time / self.max_wait_time) ** 2
        
        # Small action cost to encourage decisiveness
        action_cost = 0.01
        
        reward = batch_reward - latency_penalty - action_cost
        
        # Update metrics
        self.total_requests_processed += batch_size
        self.total_batches_sent += 1
        self.total_wait_time += avg_wait_time * batch_size
        
        # Clear batch
        self.current_batch = []
        self.time_since_last_skip = 0
        self.batch_start_time = self.current_time
        
        return reward
    
    def _compute_wait_reward(self) -> float:
        """Compute reward for WAIT action."""
        if len(self.current_batch) == 0:
            # Waiting with empty batch, small penalty
            return -0.01
        
        # Small penalty for waiting (encourages eventual action)
        wait_penalty = 0.02
        
        # Potential reward for accumulating batch (small incentive)
        accumulation_bonus = 0.01 * (len(self.current_batch) / self.max_batch_size)
        
        return accumulation_bonus - wait_penalty
    
    def _get_current_wait_time(self) -> float:
        """Get current wait time for the batch."""
        if len(self.current_batch) == 0:
            return 0.0
        return self.current_time - self.batch_start_time
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation (normalized to [0, 1]).
        
        Returns:
            [batch_size, wait_time, queue_length, time_since_last_skip,
             arrival_rate, system_load]
        """
        batch_size_norm = len(self.current_batch) / self.max_batch_size
        wait_time_norm = min(1.0, self._get_current_wait_time() / self.max_wait_time)
        queue_norm = len(self.wait_queue) / self.max_queue_length
        time_since_skip_norm = min(1.0, self.time_since_last_skip / (self.max_wait_time * 2))
        
        # Normalize arrival rate (assume max rate is 3x base rate)
        max_rate = self.traffic_gen.base_rate * 3.0
        arrival_rate_norm = min(1.0, self.arrival_rate_ema / max_rate)
        
        system_load_norm = self.system_load
        
        return np.array([
            batch_size_norm,
            wait_time_norm,
            queue_norm,
            time_since_skip_norm,
            arrival_rate_norm,
            system_load_norm
        ], dtype=np.float32)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get environment metrics."""
        avg_batch_size = (self.total_requests_processed / self.total_batches_sent 
                         if self.total_batches_sent > 0 else 0)
        avg_wait_time = (self.total_wait_time / self.total_requests_processed
                        if self.total_requests_processed > 0 else 0)
        
        p95_wait = np.percentile(self.all_wait_times, 95) if self.all_wait_times else 0.0
        slo_violations = sum(1 for w in self.all_wait_times if w > 1.0)
        slo_violation_rate = (slo_violations / len(self.all_wait_times) * 100.0) if self.all_wait_times else 0.0
        avg_queue_length = self.queue_length_sum / max(1, self.queue_length_samples)
        
        return {
            'total_requests': self.total_requests_processed,
            'total_batches': self.total_batches_sent,
            'avg_batch_size': avg_batch_size,
            'avg_wait_time': avg_wait_time,
            'p95_wait_time': p95_wait,
            'slo_violation_rate': slo_violation_rate,
            'avg_queue_length': avg_queue_length,
            'throughput': self.total_requests_processed / (self.current_time + 1e-6)
        }
