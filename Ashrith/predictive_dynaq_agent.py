"""
Predictive Dyna-Q Agent for Request Batching

Paper-inspired adaptation of the idea behind:
    "Using Grouped Linear Prediction and Accelerated Reinforcement Learning
     for Online Content Caching"

This agent is intentionally different from PPO/DQN/SAC/IMPALA-style work.
It combines:
1. Grouped linear prediction over recent arrival-rate observations
2. Tabular Q-learning on a discretized batching state
3. Dyna-Q planning updates from a learned one-step model
4. A small optimism bonus for under-explored state-action pairs

That makes it a good fit for this project when we want something that is:
- distinct from the teammates' methods
- still grounded in one of the cited papers
- simple enough to train quickly on the current environment
"""

from __future__ import annotations

from collections import defaultdict, deque
import random
from typing import Deque, Dict, List, Tuple

import numpy as np


class GroupedArrivalPredictor:
    """
    Lightweight online grouped linear predictor.

    We observe the normalized arrival-rate feature from the environment and
    learn an online one-step predictor using grouped moving averages over
    short, medium, and longer windows. This is a compact batching-friendly
    adaptation of the grouped prediction idea from the caching paper.
    """

    def __init__(
        self,
        windows: Tuple[int, ...] = (2, 4, 8),
        learning_rate: float = 0.08,
    ):
        self.windows = windows
        self.learning_rate = learning_rate
        self.max_window = max(windows)
        self.weights = np.zeros(len(windows) + 1, dtype=np.float32)
        self.history: Deque[float] = deque(maxlen=self.max_window + 1)
        self._pending_features = None

    def reset(self):
        """Clear episode-local history while keeping learned weights."""
        self.history.clear()
        self._pending_features = None

    def _build_features(self) -> np.ndarray:
        values = list(self.history)
        grouped_means: List[float] = []
        for window in self.windows:
            chunk = values[-window:] if len(values) >= window else values
            grouped_means.append(float(np.mean(chunk)) if chunk else 0.0)

        return np.array([1.0, *grouped_means], dtype=np.float32)

    def predict(self, fallback_value: float) -> float:
        """
        Predict the next normalized arrival-rate value.

        When we do not have enough history yet, fall back to the observed
        arrival-rate feature already provided by the environment.
        """
        if not self.history:
            return float(np.clip(fallback_value, 0.0, 1.0))

        features = self._build_features()
        prediction = float(np.dot(self.weights, features))
        self._pending_features = features
        return float(np.clip(prediction, 0.0, 1.0))

    def update(self, observed_value: float):
        """Online SGD update using the latest observed normalized arrival rate."""
        observed_value = float(np.clip(observed_value, 0.0, 1.0))

        if self._pending_features is not None:
            prediction = float(np.dot(self.weights, self._pending_features))
            error = observed_value - prediction
            self.weights += self.learning_rate * error * self._pending_features

        self.history.append(observed_value)


class PredictiveDynaQAgent:
    """Forecast-aware Dyna-Q agent for the batching environment."""

    def __init__(
        self,
        action_dim: int,
        alpha: float = 0.18,
        gamma: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.08,
        epsilon_decay: float = 0.992,
        planning_steps: int = 20,
        optimism_coeff: float = 0.08,
        seed: int | None = None,
    ):
        self.action_dim = action_dim
        self.device = "cpu"
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.planning_steps = planning_steps
        self.optimism_coeff = optimism_coeff
        self.rng = random.Random(seed)

        self.predictor = GroupedArrivalPredictor()
        self.q_table = defaultdict(self._zero_q_values)
        self.visit_counts = defaultdict(int)
        self.model: Dict[Tuple[Tuple[int, ...], int], Tuple[float, Tuple[int, ...], bool]] = {}

        self.episodes_trained = 0
        self.steps_trained = 0

        # 7 dimensions: 6 env features + predicted arrival feature.
        self.state_bins = (8, 8, 8, 8, 6, 6, 6)

    def _zero_q_values(self) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)

    def start_episode(self):
        self.predictor.reset()

    def _discretize_scalar(self, value: float, num_bins: int) -> int:
        value = float(np.clip(value, 0.0, 1.0))
        return min(num_bins - 1, int(value * num_bins))

    def _state_key(self, state: np.ndarray, predicted_arrival: float | None = None) -> Tuple[int, ...]:
        if predicted_arrival is None:
            predicted_arrival = self.predictor.predict(float(state[4]))

        augmented_state = list(np.asarray(state, dtype=np.float32)) + [predicted_arrival]
        return tuple(
            self._discretize_scalar(value, bins)
            for value, bins in zip(augmented_state, self.state_bins)
        )

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        state_key = self._state_key(state)
        q_values = self.q_table[state_key]

        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_dim)

        bonuses = np.array(
            [
                self.optimism_coeff / np.sqrt(self.visit_counts[(state_key, action)] + 1)
                for action in range(self.action_dim)
            ],
            dtype=np.float32,
        )
        return int(np.argmax(q_values + bonuses))

    def _q_update(
        self,
        state_key: Tuple[int, ...],
        action: int,
        reward: float,
        next_state_key: Tuple[int, ...],
        done: bool,
    ):
        current_q = self.q_table[state_key][action]
        best_next_q = 0.0 if done else float(np.max(self.q_table[next_state_key]))
        td_target = reward + self.gamma * best_next_q
        self.q_table[state_key][action] += self.alpha * (td_target - current_q)

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        train: bool = True,
    ):
        """
        Record a transition.

        During evaluation we still call this with train=False so the predictor
        can keep updating online without changing the Q-table.
        """
        state_key = self._state_key(state)

        self.predictor.update(float(next_state[4]))
        next_state_key = self._state_key(next_state)

        if not train:
            return

        self._q_update(state_key, action, reward, next_state_key, done)
        self.visit_counts[(state_key, action)] += 1
        self.model[(state_key, action)] = (reward, next_state_key, done)
        self.steps_trained += 1

        if self.model:
            model_keys = list(self.model.keys())
            for _ in range(self.planning_steps):
                sim_state_key, sim_action = self.rng.choice(model_keys)
                sim_reward, sim_next_state_key, sim_done = self.model[(sim_state_key, sim_action)]
                self._q_update(sim_state_key, sim_action, sim_reward, sim_next_state_key, sim_done)

    def end_episode(self):
        self.episodes_trained += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        payload = {
            "q_table": dict(self.q_table),
            "visit_counts": dict(self.visit_counts),
            "model": dict(self.model),
            "predictor_weights": self.predictor.weights,
            "epsilon": self.epsilon,
            "episodes_trained": self.episodes_trained,
            "steps_trained": self.steps_trained,
            "state_bins": self.state_bins,
        }
        np.save(filepath, payload, allow_pickle=True)

    def load(self, filepath: str):
        payload = np.load(filepath, allow_pickle=True).item()
        self.q_table = defaultdict(self._zero_q_values)
        for state_key, q_values in payload["q_table"].items():
            self.q_table[state_key] = np.asarray(q_values, dtype=np.float32)

        self.visit_counts = defaultdict(int, payload["visit_counts"])
        self.model = dict(payload["model"])
        self.predictor.weights = np.asarray(payload["predictor_weights"], dtype=np.float32)
        self.epsilon = float(payload.get("epsilon", self.epsilon_start))
        self.episodes_trained = int(payload.get("episodes_trained", 0))
        self.steps_trained = int(payload.get("steps_trained", 0))
