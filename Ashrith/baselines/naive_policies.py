"""Simple rule-based baselines for request batching."""

from __future__ import annotations

import numpy as np


class FixedBatchPolicy:
    def __init__(self, batch_threshold: int = 16):
        self.batch_threshold = batch_threshold

    def select_action(self, state, env) -> int:
        batch_size = float(state[0]) * env.max_batch_size
        return 1 if batch_size >= self.batch_threshold else 0


class FixedWaitPolicy:
    def __init__(self, wait_threshold: float = 1.0):
        self.wait_threshold = wait_threshold

    def select_action(self, state, env) -> int:
        wait_time = float(state[1]) * env.max_wait_time
        return 1 if wait_time >= self.wait_threshold else 0


class RandomPolicy:
    def __init__(self, skip_prob: float = 0.3, seed: int | None = None):
        self.skip_prob = skip_prob
        self.rng = np.random.RandomState(seed)

    def select_action(self, state, env) -> int:
        return int(self.rng.rand() < self.skip_prob)


def evaluate_baseline(policy, env, num_episodes: int = 20):
    episode_rewards = []
    episode_metrics = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            action = policy.select_action(state, env)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward

        metrics = env.get_metrics()
        episode_rewards.append(episode_reward)
        episode_metrics.append(metrics)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_batch_size": float(np.mean([m["avg_batch_size"] for m in episode_metrics])),
        "mean_wait_time": float(np.mean([m["avg_wait_time"] for m in episode_metrics])),
        "p95_wait_time": float(np.mean([m.get("p95_wait_time", 0.0) for m in episode_metrics])),
        "mean_throughput": float(np.mean([m["throughput"] for m in episode_metrics])),
        "avg_queue": float(np.mean([m.get("avg_queue_length", 0.0) for m in episode_metrics])),
        "slo_violations": float(np.mean([m.get("slo_violation_rate", 0.0) for m in episode_metrics])),
    }
