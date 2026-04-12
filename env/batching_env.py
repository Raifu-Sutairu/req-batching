"""
env/batching_env.py

Gymnasium environment for Dynamic Request Batching on an inference server.

──────────────────────────────────────────────────────────────────────────────
State space  (8-dim Box, float32)
──────────────────────────────────────────────────────────────────────────────
  [0] pending_requests      – number of requests currently queued
  [1] oldest_wait_ms        – age of the oldest pending request (ms)
  [2] request_rate          – EMA of observed arrival rate (req/s)
  [3] since_serve_ms        – ms since the last Serve action
  [4] batch_fill_ratio      – pending / max_batch_size  ∈ [0, 1]
  [5] time_of_day           – fractional hour ∈ [0, 24)
  [6] urgency_ratio         – oldest_wait / effective_latency_budget ∈ [0, ~3]
                              Accounts for GPU processing time: when this
                              exceeds 1.0 the current batch WILL violate SLA.
  [7] rate_trend            – fractional EMA rate change per step ∈ [−1, 1]
                              Positive = traffic accelerating (spike incoming),
                              Negative = traffic decelerating (lull).
                              Lets the agent act before the EMA fully adapts.

──────────────────────────────────────────────────────────────────────────────
Action space  Discrete(2)
──────────────────────────────────────────────────────────────────────────────
  0 = Wait  – keep accumulating, pay a small urgency cost
  1 = Serve – dispatch all pending requests; GPU starts processing

──────────────────────────────────────────────────────────────────────────────
Reward
──────────────────────────────────────────────────────────────────────────────
  Serve:  alpha * batch_size
        − beta  * oldest_wait_ms
        − dispatch_cost
        − sla_penalty_per_ms * Σ max(0, wait_i + processing_ms − SLA) per req

  Wait:   −beta  * oldest_wait_ms
        − idle_penalty   (only if queue is empty)

──────────────────────────────────────────────────────────────────────────────
Episode
──────────────────────────────────────────────────────────────────────────────
  episode_ms milliseconds of simulated time (default 60 000 = 60 s).
  The agent makes a decision every decision_interval_ms (default 10 ms).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import CONFIG
from env.traffic_generator import TrafficGenerator


def gpu_processing_ms(batch_size: int, cfg: dict) -> float:
    """Return estimated GPU processing time for a batch of `batch_size` requests.

    Uses a linear model fitted to Triton inference benchmarks:
        t(n) = gpu_base_ms + gpu_per_request_ms * n

    Parameters
    ----------
    batch_size : int
        Number of requests in the batch.
    cfg : dict
        Configuration dict containing ``gpu_base_ms`` and ``gpu_per_request_ms``.

    Returns
    -------
    float
        Estimated processing time in milliseconds.
    """
    return cfg["gpu_base_ms"] + cfg["gpu_per_request_ms"] * batch_size


class BatchingEnv(gym.Env):
    """Dynamic Request Batching Gymnasium environment.

    The agent learns *when* to dispatch accumulated requests to a GPU inference
    backend.  The key trade-off: waiting builds larger (more efficient) batches
    but increases request latency; serving too aggressively pays repeated
    dispatch overhead for tiny batches.

    The SLA is on TOTAL client-perceived latency:
        total_latency = queue_wait_time + gpu_processing_time(batch_size)

    Parameters
    ----------
    config : dict | None
        Override selected config keys.  Defaults to the global CONFIG.
    render_mode : str | None
        ``"human"`` prints step info to stdout.
    seed : int | None
        RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: dict | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        self.cfg = {**CONFIG, **(config or {})}
        self.render_mode = render_mode
        self._seed = seed

        # ── Observation space (8-dim) ──────────────────────────────────────
        peak_rate = self.cfg["arrival_rate"] * self.cfg["peak_multiplier"]
        obs_low  = np.array([0, 0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
        obs_high = np.array([
            self.cfg["max_batch_size"],          # [0] pending_requests
            self.cfg["max_latency_ms"] * 2.0,    # [1] oldest_wait_ms (allow overshoot)
            peak_rate * 2.0,                      # [2] request_rate
            self.cfg["max_latency_ms"] * 2.0,    # [3] since_serve_ms
            1.0,                                  # [4] batch_fill_ratio
            24.0,                                 # [5] time_of_day
            3.0,                                  # [6] urgency_ratio (>1 = SLA breach)
            1.0,                                  # [7] rate_trend ∈ [−1, 1]
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ── Action space ───────────────────────────────────────────────────
        self.action_space = spaces.Discrete(2)  # 0=Wait, 1=Serve

        # Internal state (initialised in reset)
        self._rng:        np.random.Generator | None = None
        self._traffic:    TrafficGenerator | None    = None
        self._queue:      list[float]                = []  # arrival times (ms)
        self._sim_time_ms:    float = 0.0
        self._last_serve_ms:  float = 0.0
        self._episode_end_ms: float = 0.0
        self._ema_rate:       float = 0.0
        self._prev_ema_rate:  float = 0.0   # for rate_trend computation
        self._start_hour:     float = 0.0

        # Episode metrics
        self._total_served:      int        = 0
        self._total_batches:     int        = 0
        self._total_dispatches:  int        = 0
        self._latency_samples:   list[float]= []   # TOTAL latency per request
        self._sla_violations:    int        = 0    # requests that breached SLA

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        rng_seed = seed if seed is not None else self._seed
        self._rng = np.random.default_rng(rng_seed)

        self._traffic = TrafficGenerator(
            base_rate=self.cfg["arrival_rate"],
            peak_hours=tuple(self.cfg["peak_hours"]),
            peak_multiplier=self.cfg["peak_multiplier"],
            offpeak_multiplier=self.cfg["offpeak_multiplier"],
            rng=self._rng,
        )

        self._start_hour      = float(self._rng.uniform(0, 24))
        self._sim_time_ms     = 0.0
        self._last_serve_ms   = 0.0
        self._episode_end_ms  = float(self.cfg["episode_ms"])
        self._queue           = []
        self._ema_rate        = float(self.cfg["arrival_rate"])
        self._prev_ema_rate   = float(self.cfg["arrival_rate"])
        self._total_served    = 0
        self._total_batches   = 0
        self._total_dispatches= 0
        self._latency_samples = []
        self._sla_violations  = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        cfg         = self.cfg
        dt_ms       = float(cfg["decision_interval_ms"])
        alpha       = cfg["alpha"]
        beta        = cfg["beta"]
        idle_pen    = cfg["idle_penalty"]
        dispatch_c  = cfg["dispatch_cost"]
        sla_ppm     = cfg["sla_penalty_per_ms"]
        max_lat     = cfg["max_latency_ms"]

        # ── 1. Advance clock + generate arrivals ───────────────────────────
        hour = self._current_hour()
        new_arrivals = self._traffic.arrivals_in_window_ms(dt_ms, hour)

        if new_arrivals > 0:
            # Vectorised: one numpy call instead of new_arrivals scalar calls.
            # Arrival times are spread uniformly within [sim_time, sim_time+dt).
            offsets = self._rng.uniform(0, dt_ms, size=new_arrivals)
            self._queue.extend((self._sim_time_ms + offsets).tolist())

        # Hard cap to prevent unbounded queues under extreme traffic.
        # Keep the OLDEST requests ([:max_q]) — they are closest to the SLA
        # deadline and must not be silently dropped.  Excess new arrivals are
        # shed (back-pressure: caller retries or drops).
        max_q = cfg["max_batch_size"]
        if len(self._queue) > max_q:
            self._queue = self._queue[:max_q]

        self._sim_time_ms += dt_ms

        # ── 2. Update EMA arrival rate ─────────────────────────────────────
        observed_rate = (new_arrivals / dt_ms) * 1000.0   # req/s
        alpha_ema     = 0.1
        self._ema_rate = alpha_ema * observed_rate + (1 - alpha_ema) * self._ema_rate

        # ── 3. Compute oldest wait ─────────────────────────────────────────
        oldest_wait = self._oldest_wait_ms()

        # ── 4. Compute reward ──────────────────────────────────────────────
        reward = 0.0

        if action == 1:  # Serve
            batch_size = len(self._queue)

            if batch_size > 0:
                proc_ms = gpu_processing_ms(batch_size, cfg)

                # Record total (client-perceived) latency per request
                sla_violation_cost = 0.0
                for arrival_t in self._queue:
                    wait_ms     = self._sim_time_ms - arrival_t
                    total_lat   = wait_ms + proc_ms
                    self._latency_samples.append(total_lat)
                    if total_lat > max_lat:
                        self._sla_violations += 1
                        sla_violation_cost += sla_ppm * (total_lat - max_lat)

                self._queue.clear()
                self._last_serve_ms  = self._sim_time_ms
                self._total_served  += batch_size
                self._total_batches += 1

                reward = (alpha * batch_size
                          - beta  * oldest_wait
                          - dispatch_c
                          - sla_violation_cost)
            else:
                # Empty serve: only pay dispatch cost
                reward = -dispatch_c

        else:  # Wait
            idle_pen_val = idle_pen if len(self._queue) == 0 else 0.0
            reward = -beta * oldest_wait - idle_pen_val

        # ── 5. Termination ─────────────────────────────────────────────────
        terminated = self._sim_time_ms >= self._episode_end_ms
        truncated  = False

        obs  = self._get_obs()
        info = self._get_info()

        # Update prev EMA AFTER building obs so rate_trend reflects this step's change
        self._prev_ema_rate = self._ema_rate

        if self.render_mode == "human":
            self._render_step(action, reward, len(self._queue), oldest_wait)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(
                f"t={self._sim_time_ms:.0f}ms  queue={len(self._queue)}  "
                f"oldest={self._oldest_wait_ms():.1f}ms  "
                f"rate={self._ema_rate:.0f} req/s"
            )

    def close(self):
        pass

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        pending   = float(len(self._queue))
        oldest    = self._oldest_wait_ms()
        rate      = self._ema_rate
        t_since   = self._sim_time_ms - self._last_serve_ms
        fill      = pending / max(self.cfg["max_batch_size"], 1)
        tod       = self._current_hour()

        # urgency_ratio: how close is the oldest request to a SLA breach,
        # accounting for the GPU processing overhead if we serve now.
        est_proc  = gpu_processing_ms(max(1, int(pending)), self.cfg)
        budget    = max(1.0, self.cfg["max_latency_ms"] - est_proc)
        urgency   = oldest / budget  # > 1.0 → SLA breach if served now

        # rate_trend: fractional change in EMA rate since last step.
        # Positive = traffic accelerating, Negative = decelerating.
        rate_trend = (self._ema_rate - self._prev_ema_rate) / max(1.0, self._prev_ema_rate)

        obs = np.array([pending, oldest, rate, t_since, fill, tod, urgency, rate_trend],
                       dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self) -> dict:
        # Percentiles are expensive (O(n log n)) on large latency sample arrays.
        # Only compute them at episode end or when the episode has enough data
        # to be meaningful.  During training this function is called every step,
        # so we must keep it O(1) for the hot path.
        terminated = self._sim_time_ms >= self._episode_end_ms
        p50 = p95 = p99 = 0.0
        if terminated and self._latency_samples:
            s   = self._latency_samples
            p50 = float(np.percentile(s, 50))
            p95 = float(np.percentile(s, 95))
            p99 = float(np.percentile(s, 99))

        sla_rate = self._sla_violations / max(1, self._total_served)

        return {
            "sim_time_ms":        self._sim_time_ms,
            "total_served":       self._total_served,
            "total_batches":      self._total_batches,
            "sla_violations":     self._sla_violations,
            "sla_violation_rate": sla_rate,
            "p50_latency_ms":     p50,
            "p95_latency_ms":     p95,
            "p99_latency_ms":     p99,
            "queue_length":       len(self._queue),
            "ema_rate":           self._ema_rate,
        }

    def _oldest_wait_ms(self) -> float:
        if not self._queue:
            return 0.0
        return float(self._sim_time_ms - self._queue[0])

    def _current_hour(self) -> float:
        elapsed_hours = self._sim_time_ms / (1_000.0 * 3_600.0)
        return (self._start_hour + elapsed_hours) % 24.0

    def _render_step(self, action: int, reward: float, queue_len: int, oldest: float):
        label = "SERVE" if action == 1 else "WAIT "
        print(
            f"[{self._sim_time_ms:8.0f} ms] {label}  "
            f"queue={queue_len:4d}  oldest={oldest:6.1f} ms  "
            f"rate={self._ema_rate:6.0f} req/s  reward={reward:+.3f}"
        )
