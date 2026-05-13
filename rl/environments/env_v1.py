import gymnasium as gym
import numpy as np
import random
import sys
from pathlib import Path
from typing import List

# Add parent directory to path so we can import from rl.*
sys.path.append(str(Path(__file__).parent.parent))

from data.episode import Episode
from reward import compute_reward
from normalise import obs_to_tensor
from config import config

class BatchFlushEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, kafka_episode_buffer: List[Episode]):
        super().__init__()
        # 4D Observation Space normalised to [0, 1]
        self.observation_space = gym.spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        # Action Space: 0 = WAIT, 1 = FLUSH
        self.action_space = gym.spaces.Discrete(2)
        
        self.episodes = kafka_episode_buffer
        self._current_episode = None
        self._step_idx = 0
        self.rng = random.Random()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            
        if not self.episodes:
            raise ValueError("Episode buffer is empty! Make sure Kafka consumer has loaded data.")
            
        self._current_episode = self.rng.choice(self.episodes)
        self._step_idx = 0
        return self._get_obs(), {}

    def step(self, action: int):
        step_data = self._current_episode.steps[self._step_idx]
        terminated = False

        if action == 1:  # FLUSH
            reward = compute_reward(
                action=1, 
                **step_data.to_kwargs(), 
                timeout_ms=config.batch_timeout_ms, 
                was_forced=False
            )
            terminated = True
        else:  # WAIT
            self._step_idx += 1
            if self._step_idx >= len(self._current_episode.steps):
                # The heuristic timeout or size cap fired here in the original trace
                last_step = self._current_episode.steps[-1]
                reward = compute_reward(
                    action=1, 
                    **last_step.to_kwargs(), 
                    timeout_ms=config.batch_timeout_ms, 
                    was_forced=True
                )
                terminated = True
            else:
                reward = compute_reward(
                    action=0, 
                    **step_data.to_kwargs(), 
                    timeout_ms=config.batch_timeout_ms, 
                    was_forced=False
                )

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        # Safely bound the index
        idx = min(self._step_idx, len(self._current_episode.steps) - 1)
        step_data = self._current_episode.steps[idx]
        
        return obs_to_tensor(
            batch_size=step_data.batch_size,
            batch_age_ms=step_data.batch_age_ms,
            upstream_p99_ms=step_data.upstream_p99_ms,
            request_rate=step_data.request_rate,
            max_batch_size=config.max_batch_size,
            batch_timeout_ms=config.batch_timeout_ms
        )
