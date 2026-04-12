"""
deploy/middleware.py

Production-ready middleware class for deploying the trained PPO batching agent.

This module exposes BatchingMiddleware — a self-contained component that can
sit between a request router and a GPU inference backend.  It loads the trained
policy and VecNormalize statistics once at startup, then answers the question
"should I dispatch now?" in microseconds.

──────────────────────────────────────────────────────────────────────────────
Architecture in production
──────────────────────────────────────────────────────────────────────────────

  [clients] ──req──► [BatchingMiddleware] ──batch──► [GPU inference backend]
                           ▲
                    calls should_dispatch()
                    every decision_interval_ms

The middleware runs as a lightweight loop (or is called by your async event
loop).  It tracks its own EMA arrival rate and queue state — the same signals
the agent was trained on.

──────────────────────────────────────────────────────────────────────────────
Deployment checklist
──────────────────────────────────────────────────────────────────────────────
  1. models/ppo_final.zip      — trained policy weights
  2. models/ppo_vecnorm.pkl    — VecNormalize statistics (MUST match the model)

  If you retrain, both files must be regenerated together.
  The VecNorm running mean/variance is fitted to the training distribution;
  using stale stats with a new model will produce incorrect normalisation.

──────────────────────────────────────────────────────────────────────────────
Dynamic parameters learned by the model
──────────────────────────────────────────────────────────────────────────────
The following are NOT hard-coded — the agent adapts them dynamically at
inference time based on the current system state:

  • Effective dispatch threshold  (function of batch_fill_ratio + urgency)
  • Response to arrival rate spikes (via ema_rate in observation)
  • Time-of-day load prediction    (via time_of_day in observation)
  • Urgency adjustment for GPU overhead (via urgency_ratio in observation)

To CHANGE the operating point (e.g. new SLA or new GPU), retrain with updated
config.py rather than patching thresholds — the agent will relearn the optimal
adaptive policy.

──────────────────────────────────────────────────────────────────────────────
Usage examples
──────────────────────────────────────────────────────────────────────────────

    # Startup (once per process)
    from deploy.middleware import BatchingMiddleware
    mw = BatchingMiddleware(
        model_path  = "models/ppo_final",
        vecnorm_path= "models/ppo_vecnorm.pkl",
    )

    # In your request handler loop (every 10 ms)
    import time
    while True:
        mw.record_arrivals(new_request_count)   # called as requests arrive
        if mw.should_dispatch():
            batch = mw.flush()
            inference_backend.process(batch)
        time.sleep(0.010)  # 10 ms decision interval

    # Async variant (FastAPI / asyncio)
    async def batching_loop():
        while True:
            await asyncio.sleep(0.010)
            if mw.should_dispatch():
                batch = mw.flush()
                await inference_backend.process_async(batch)
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

from env.batching_env import gpu_processing_ms
from config import CONFIG


class BatchingMiddleware:
    """RL-powered batching middleware.

    Tracks queue state in-process and answers should_dispatch() queries using
    the trained PPO policy.  Thread-safe for single-writer / single-reader use.

    Parameters
    ----------
    model_path : str
        Path to the trained SB3 PPO model (with or without .zip extension).
    vecnorm_path : str | None
        Path to the saved VecNormalize statistics (.pkl).  Required for
        correct normalisation — the model will produce bad actions without it.
    config : dict | None
        Override config parameters (e.g. different SLA for a specific tenant).
    decision_interval_ms : float
        How often the caller queries should_dispatch() (ms).  Must match
        the training decision_interval_ms for correct EMA rate estimation.
    """

    def __init__(
        self,
        model_path:            str,
        vecnorm_path:          str | None = None,
        config:                dict | None = None,
        decision_interval_ms:  float = 10.0,
    ):
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable_baselines3 is required.  "
                "Install with: pip install stable-baselines3"
            )

        self.cfg = {**CONFIG, **(config or {})}
        self.dt_ms = decision_interval_ms

        # Load policy
        self._model = PPO.load(model_path)
        self._model.policy.set_training_mode(False)

        # Load VecNormalize statistics
        self._vecnorm: VecNormalize | None = None
        if vecnorm_path and os.path.exists(vecnorm_path):
            dummy = DummyVecEnv([lambda: _DummyEnv(self.cfg)])
            self._vecnorm = VecNormalize.load(vecnorm_path, dummy)
            self._vecnorm.training     = False
            self._vecnorm.norm_reward  = False
        else:
            print(
                "[BatchingMiddleware] WARNING: vecnorm_path not provided or not found. "
                "Predictions may be inaccurate without observation normalisation."
            )

        # Queue state
        self._queue:         list[float] = []   # arrival wall-clock times (ms)
        self._last_serve_ms: float = self._now_ms()
        self._sim_time_ms:   float = 0.0
        self._ema_rate:      float = float(self.cfg["arrival_rate"])
        self._prev_ema_rate: float = float(self.cfg["arrival_rate"])
        self._alpha_ema:     float = 0.1

        # Metrics
        self.total_served    = 0
        self.total_dispatches= 0
        self.sla_violations  = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def record_arrivals(self, n: int) -> None:
        """Register *n* new requests arriving at this moment.

        Call this once per incoming request (or in bulk) before
        calling should_dispatch().
        """
        now = self._now_ms()
        for _ in range(n):
            self._queue.append(now)

        # Cap queue at hardware limit
        max_q = self.cfg["max_batch_size"]
        if len(self._queue) > max_q:
            self._queue = self._queue[-max_q:]

    def should_dispatch(self) -> bool:
        """Query the PPO policy: should we dispatch the current queue?

        This advances the internal simulation clock by decision_interval_ms
        and updates the EMA arrival rate.  Returns True if the policy
        recommends dispatching now.

        Call this at a regular cadence (every decision_interval_ms).
        """
        self._sim_time_ms += self.dt_ms

        # Update EMA rate based on queue growth since last call
        current_q     = len(self._queue)
        observed_rate = (current_q / self.dt_ms) * 1000.0 * 0.1  # rough
        self._prev_ema_rate = self._ema_rate
        self._ema_rate = (self._alpha_ema * observed_rate
                         + (1 - self._alpha_ema) * self._ema_rate)

        obs = self._build_obs()
        if self._vecnorm is not None:
            obs = self._vecnorm.normalize_obs(obs)

        action, _ = self._model.predict(obs, deterministic=True)
        return int(action) == 1

    def flush(self) -> list[float]:
        """Dispatch the current queue and return the list of arrival times.

        The caller should pass these to the inference backend.  The arrival
        times can be used to compute end-to-end latency once the result is
        ready.

        Returns
        -------
        list[float]
            Wall-clock arrival times (ms) of dispatched requests.
        """
        batch = list(self._queue)
        self._queue.clear()
        self._last_serve_ms = self._now_ms()

        batch_size = len(batch)
        self.total_served     += batch_size
        self.total_dispatches += 1

        # SLA check for monitoring
        proc_ms = gpu_processing_ms(batch_size, self.cfg)
        now     = self._now_ms()
        for arrival_t in batch:
            total_lat = (now - arrival_t) + proc_ms
            if total_lat > self.cfg["max_latency_ms"]:
                self.sla_violations += 1

        return batch

    def stats(self) -> dict:
        """Return current operating statistics."""
        return {
            "queue_length":     len(self._queue),
            "total_served":     self.total_served,
            "total_dispatches": self.total_dispatches,
            "sla_violations":   self.sla_violations,
            "sla_viol_rate":    (self.sla_violations / max(1, self.total_served)),
            "ema_rate_req_s":   self._ema_rate,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _now_ms(self) -> float:
        return time.monotonic() * 1000.0

    def _oldest_wait_ms(self) -> float:
        if not self._queue:
            return 0.0
        return self._now_ms() - self._queue[0]

    def _build_obs(self) -> np.ndarray:
        pending  = float(len(self._queue))
        oldest   = self._oldest_wait_ms()
        rate     = self._ema_rate
        t_since  = self._now_ms() - self._last_serve_ms
        fill     = pending / max(self.cfg["max_batch_size"], 1)
        tod      = (datetime.now().hour + datetime.now().minute / 60.0)

        est_proc   = gpu_processing_ms(max(1, int(pending)), self.cfg)
        budget     = max(1.0, self.cfg["max_latency_ms"] - est_proc)
        urgency    = oldest / budget
        rate_trend = (self._ema_rate - self._prev_ema_rate) / max(1.0, self._prev_ema_rate)

        obs = np.array([pending, oldest, rate, t_since, fill, tod, urgency, rate_trend],
                       dtype=np.float32)

        # Clip to trained observation bounds
        low  = np.array([0, 0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
        peak = self.cfg["arrival_rate"] * self.cfg["peak_multiplier"]
        high = np.array([
            self.cfg["max_batch_size"],
            self.cfg["max_latency_ms"] * 2.0,
            peak * 2.0,
            self.cfg["max_latency_ms"] * 2.0,
            1.0, 24.0, 3.0, 1.0,
        ], dtype=np.float32)
        return np.clip(obs, low, high)


class _DummyEnv:
    """Minimal shim so VecNormalize.load() has an env to wrap."""

    def __init__(self, cfg: dict):
        import gymnasium as gym
        from gymnasium import spaces
        peak = cfg["arrival_rate"] * cfg["peak_multiplier"]
        self.observation_space = spaces.Box(
            low  = np.array([0, 0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32),
            high = np.array([cfg["max_batch_size"], cfg["max_latency_ms"]*2,
                             peak*2, cfg["max_latency_ms"]*2, 1.0, 24.0, 3.0, 1.0],
                            dtype=np.float32),
        )
        self.action_space = spaces.Discrete(2)

    def reset(self, **kw):
        return np.zeros(7, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(7, dtype=np.float32), 0.0, True, False, {}


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the deployment middleware")
    parser.add_argument("--model",   default=os.path.join(ROOT, "models", "ppo_final"))
    parser.add_argument("--vecnorm", default=os.path.join(ROOT, "models", "ppo_vecnorm.pkl"))
    parser.add_argument("--steps",   type=int, default=200,
                        help="Number of simulated decision steps")
    args = parser.parse_args()

    print("Initialising BatchingMiddleware …")
    mw = BatchingMiddleware(model_path=args.model, vecnorm_path=args.vecnorm)
    print("  Model loaded.\n")

    dispatches = 0
    total_in   = 0

    for step in range(args.steps):
        # Simulate 20 arrivals per 10ms tick (≈ 2000 req/s)
        n_arrive = np.random.poisson(20)
        mw.record_arrivals(n_arrive)
        total_in += n_arrive

        if mw.should_dispatch():
            batch = mw.flush()
            dispatches += 1
            if dispatches <= 5:
                print(f"  Step {step:4d}: DISPATCH  batch_size={len(batch):4d}  "
                      f"queue_after={len(mw._queue)}")

        time.sleep(0.001)  # 1ms for demo speed

    s = mw.stats()
    print(f"\nResults over {args.steps} steps:")
    print(f"  Total arrivals   : {total_in:,}")
    print(f"  Total served     : {s['total_served']:,}")
    print(f"  Total dispatches : {s['total_dispatches']:,}")
    print(f"  SLA violations   : {s['sla_violations']}  ({s['sla_viol_rate']:.2%})")
    print(f"  EMA rate         : {s['ema_rate_req_s']:.0f} req/s")
