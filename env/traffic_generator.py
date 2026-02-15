"""
Traffic Generator for Request Batching Environment

Simulates various traffic patterns including:
- Poisson arrivals (constant rate)
- Bursty traffic (alternating high/low intensity)
- Time-varying patterns (peak vs idle hours)
"""

import numpy as np
from typing import Literal


class TrafficGenerator:
    """Generates request arrival patterns for the batching environment."""
    
    def __init__(
        self,
        pattern: Literal['poisson', 'bursty', 'time_varying'] = 'poisson',
        base_rate: float = 5.0,
        seed: int = None
    ):
        """
        Initialize traffic generator.
        
        Args:
            pattern: Type of traffic pattern to generate
            base_rate: Base arrival rate (requests per second)
            seed: Random seed for reproducibility
        """
        self.pattern = pattern
        self.base_rate = base_rate
        self.rng = np.random.RandomState(seed)
        
        # For bursty traffic
        self.burst_state = 'low'  # 'low' or 'high'
        self.burst_duration = 0
        self.burst_switch_interval = 50  # steps before switching
        
        # For time-varying traffic
        self.time_step = 0
        self.peak_hours = [(300, 600), (1200, 1800)]  # time steps
        
    def reset(self):
        """Reset generator state."""
        self.burst_state = 'low'
        self.burst_duration = 0
        self.time_step = 0
        
    def generate_arrivals(self, dt: float = 1.0) -> int:
        """
        Generate number of requests arriving in this time step.
        
        Args:
            dt: Time delta (seconds)
            
        Returns:
            Number of new requests
        """
        current_rate = self._get_current_rate()
        # Poisson process: number of arrivals in interval dt
        num_arrivals = self.rng.poisson(current_rate * dt)
        
        self.time_step += 1
        return max(0, num_arrivals)
    
    def _get_current_rate(self) -> float:
        """Get current arrival rate based on pattern."""
        if self.pattern == 'poisson':
            return self.base_rate
            
        elif self.pattern == 'bursty':
            # Alternate between high and low traffic bursts
            self.burst_duration += 1
            
            if self.burst_duration >= self.burst_switch_interval:
                self.burst_state = 'high' if self.burst_state == 'low' else 'low'
                self.burst_duration = 0
                
            return self.base_rate * 3.0 if self.burst_state == 'high' else self.base_rate * 0.5
            
        elif self.pattern == 'time_varying':
            # Higher rate during peak hours
            is_peak = any(start <= self.time_step % 2000 < end 
                         for start, end in self.peak_hours)
            return self.base_rate * 2.5 if is_peak else self.base_rate * 0.8
            
        else:
            return self.base_rate
    
    def get_current_rate(self) -> float:
        """Get the current arrival rate (for observation)."""
        return self._get_current_rate()
