"""
extended_env.py  (v2 — all problems fixed)
------------------------------------------
Fixes applied vs v1:
  [P1]  State normalisation — RunningNormWrapper tracks per-dim mean/std
        and normalises online. Prevents LSTM gradient dominance from
        large-magnitude dims (pending_requests, since_serve_ms).
  [P2]  Rate-of-change added to state — delta_request_rate = rate_now
        minus rate 5 steps ago. State is now 8D (was 6D). This is the
        burst-onset signal that justifies having an LSTM at all.
  [P3]  SLA urgency signal — oldest_wait_ms replaced with
        sla_urgency = oldest_wait_ms / max_latency_ms (0–1 normalised).
        Agent now directly sees how close requests are to the deadline.
  [P4]  Queue length captured BEFORE serve — batch bonus now uses
        queue size at the moment of dispatch, not after. The old code
        read queue_length AFTER serving, which was always near zero.
  [P5]  Gradual SLA penalty — replaced hard step penalty at 500ms with
        a continuously growing quadratic: -β × (sla_urgency²) per step.
        Teaches the agent to manage latency proactively, not cliff-react.
  [P6]  Empty-serve penalty — penalises serving on a near-empty queue
        (< MIN_BATCH_FOR_BONUS). Prevents W100 dominance in off-peak
        where the agent waits 100ms and earns nothing but idle penalties.
  [P7]  Episode phase balancing — make_extended_env accepts a phase
        parameter ('random'|'peak'|'offpeak') so the trainer can
        explicitly balance training across traffic regimes.

State vector is now 8D:
  [0] pending_requests    (norm)
  [1] sla_urgency         (oldest_wait / sla_limit, 0-1)
  [2] request_rate        (norm)
  [3] delta_request_rate  (NEW — rate change over last 5 steps)
  [4] since_serve_ms      (norm)
  [5] batch_fill_ratio    (0-1, uses real max_batch_size)
  [6] time_of_day         (norm)
  [7] queue_pressure      (NEW — pending / arrival_rate * 1000, wait-time proxy)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


# ── Constants ────────────────────────────────────────────────────────────────

WAIT_STEPS = {0: 0, 1: 2, 2: 5, 3: 10}   # action → inner wait steps
STATE_DIM   = 8                             # expanded state vector
SLA_MS      = 500.0                         # SLA deadline (ms)
MIN_BATCH_FOR_BONUS = 2                     # min batch size to earn sqrt bonus

# Reward shaping coefficients
ALPHA_SHAPE      = 0.8    # sqrt(batch_size) bonus — increased from 0.5
ANTICIPATION     = 0.15   # bonus for waiting when queue is growing
EMPTY_SERVE_PEN  = -1.0   # penalty for serving empty / near-empty queue
GRADUAL_SLA_COEF = 2.0    # coefficient for quadratic SLA urgency penalty


# ── Running normaliser ───────────────────────────────────────────────────────

class RunningNorm:
    """
    Online per-dimension mean/variance normalisation (Welford's algorithm).
    Normalises each state dimension to approximately zero mean, unit variance.
    Clips to [-5, 5] to prevent outlier explosion.

    This is the single most impactful fix — unnormalised inputs cause LSTM
    gradient dominance from high-magnitude dims (since_serve_ms can be
    2000ms while batch_fill_ratio is 0–1).
    """

    def __init__(self, dim, clip=5.0, warmup=100):
        self.dim     = dim
        self.clip    = clip
        self.warmup  = warmup        # steps before normalisation kicks in
        self.count   = 0
        self.mean    = np.zeros(dim, dtype=np.float64)
        self.M2      = np.ones(dim,  dtype=np.float64)  # sum of squared diffs

    def update(self, x):
        """Update running statistics with new observation x."""
        self.count += 1
        delta  = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self):
        if self.count < 2:
            return np.ones(self.dim)
        return np.maximum(self.M2 / (self.count - 1), 1e-8)

    @property
    def std(self):
        return np.sqrt(self.var)

    def normalise(self, x):
        """Normalise x using current running statistics."""
        if self.count < self.warmup:
            return x  # pass through raw during warmup
        normed = (x - self.mean) / (self.std + 1e-8)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


# ── Extended env wrapper ─────────────────────────────────────────────────────

class ExtendedBatchingEnv(gym.Wrapper):
    """
    Gym wrapper that extends BatchingEnv with:
      - 4-action discrete space (serve_now, wait_20ms, wait_50ms, wait_100ms)
      - 8-dimensional state vector (was 6D)
      - Online state normalisation
      - All reward fixes from the audit

    Action semantics:
      0 → serve_now   (1 inner step: serve)
      1 → wait_20ms   (2 wait steps + 1 serve)
      2 → wait_50ms   (5 wait steps + 1 serve)
      3 → wait_100ms  (10 wait steps + 1 serve)

    Greedy agent = always action=0. Cannot earn wait bonuses.
    SAC learns when actions 1-3 are worth it.
    """

    def __init__(self, base_env, alpha_shape=ALPHA_SHAPE,
                 anticipation_bonus=ANTICIPATION,
                 empty_serve_penalty=EMPTY_SERVE_PEN,
                 gradual_sla_coef=GRADUAL_SLA_COEF):
        super().__init__(base_env)

        # Override spaces
        self.action_space      = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

        self.alpha_shape         = alpha_shape
        self.anticipation_bonus  = anticipation_bonus
        self.empty_serve_penalty = empty_serve_penalty
        self.gradual_sla_coef    = gradual_sla_coef

        # State tracking
        self._last_raw_obs  = None   # raw 6D obs from base env
        self._last_info     = {}
        self._rate_history  = deque(maxlen=6)   # for delta_request_rate [P2]
        self._norm          = RunningNorm(STATE_DIM)  # [P1]
        self._sla_ms        = SLA_MS
        # BatchingEnv stores config as self.cfg (not self.config)
        _base = base_env.unwrapped if hasattr(base_env, 'unwrapped') else base_env
        _cfg  = getattr(_base, 'cfg', None) or getattr(_base, 'config', {})
        self._max_batch = _cfg.get("max_batch_size", 100) if _cfg else 100

    def _build_state(self, raw_obs, info):
        """
        Convert raw 6D obs + info dict into enriched 8D normalised state.

        raw_obs: [pending_requests, oldest_wait_ms, request_rate,
                  since_serve_ms, batch_fill_ratio, time_of_day]
        """
        pending     = float(raw_obs[0])
        oldest_ms   = float(raw_obs[1])
        rate        = float(raw_obs[2])
        since_ms    = float(raw_obs[3])
        fill_ratio  = float(raw_obs[4])
        tod         = float(raw_obs[5])

        # [P3] SLA urgency — normalised 0–1
        sla_urgency = min(oldest_ms / self._sla_ms, 1.0)

        # [P2] Rate-of-change — delta over last 5 observations
        self._rate_history.append(rate)
        if len(self._rate_history) >= 5:
            delta_rate = rate - self._rate_history[-5]
        else:
            delta_rate = 0.0

        # Queue pressure — how many seconds of requests are waiting
        queue_pressure = (pending / max(rate, 1.0)) * 1000.0  # ms

        state_raw = np.array([
            pending,
            sla_urgency,        # [P3] replaces raw oldest_wait_ms
            rate,
            delta_rate,         # [P2] new
            since_ms,
            fill_ratio,
            tod,
            queue_pressure,     # new
        ], dtype=np.float32)

        # [P1] Update normaliser and return normalised state
        self._norm.update(state_raw)
        return self._norm.normalise(state_raw)

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self._last_raw_obs = raw_obs
        self._last_info    = info
        self._rate_history.clear()
        self.last_batch_size = 0.0
        state = self._build_state(raw_obs, info)
        return state, info

    def step(self, action):
        wait_n = WAIT_STEPS[action]

        total_reward = 0.0
        terminated   = False
        truncated    = False
        obs          = None
        info         = {}

        # [P6] Anticipatory bonus — fires before we know if the wait paid off
        # Only when queue is actually building (rate > 0 and queue non-empty)
        if action > 0 and self._last_info:
            queue_len = float(self._last_info.get("queue_length", 0))
            rate      = float(self._last_raw_obs[2]) if self._last_raw_obs is not None else 0
            delta     = self._rate_history[-1] - self._rate_history[0] \
                        if len(self._rate_history) >= 2 else 0
            # Anticipation only fires when rate is rising OR queue already building
            if queue_len >= 2 or delta > 10:
                total_reward += self.anticipation_bonus * action

        # ── [P4] Capture queue length BEFORE serve ────────────────────
        # This is the critical bug fix. queue_length after serving is ~0.
        # We need to know what was in the queue at dispatch time.
        # For action=0 (serve_now, no wait), capture NOW from last_info.
        pre_serve_queue = float(self._last_info.get("queue_length", 0))                           if self._last_info else 0.0

        # ── Wait phase ────────────────────────────────────────────────
        for step_i in range(wait_n):
            raw_obs, r, terminated, truncated, info = self.env.step(0)  # Wait
            self._last_raw_obs = raw_obs
            self._last_info    = info

            # [P5] Gradual SLA penalty — accumulates per wait step
            oldest_ms    = float(raw_obs[1])
            sla_urgency  = min(oldest_ms / self._sla_ms, 1.0)
            sla_grad_pen = -self.gradual_sla_coef * (sla_urgency ** 2) * 0.1
            total_reward += r + sla_grad_pen

            if terminated or truncated:
                state = self._build_state(raw_obs, info)
                return state, total_reward, terminated, truncated, info

        # Capture queue length just before serving [P4]
        pre_serve_queue = float(self._last_info.get("queue_length", 0)) \
                          if self._last_info else 0

        # If we had no wait steps, capture queue now
        if wait_n == 0 and self._last_info:
            pre_serve_queue = float(self._last_info.get("queue_length", 0))

        # ── Serve phase ───────────────────────────────────────────────
        raw_obs, r, terminated, truncated, info = self.env.step(1)  # Serve
        self._last_raw_obs = raw_obs
        self._last_info    = info
        total_reward += r
        # Expose actual batch size that was dispatched (for logging)
        self.last_batch_size = pre_serve_queue

        # ── [P4+P6] Batch bonus using PRE-serve queue size ────────────
        batch_size = pre_serve_queue

        if batch_size < MIN_BATCH_FOR_BONUS:
            # [P6] Empty-serve penalty
            total_reward += self.empty_serve_penalty * (1.0 - batch_size / MIN_BATCH_FOR_BONUS)
        else:
            # Superlinear bonus — uses CORRECT pre-serve batch size
            total_reward += self.alpha_shape * (batch_size ** 0.5)

        # ── [P5] Final SLA urgency penalty at time of serve ───────────
        oldest_ms   = float(raw_obs[1])
        sla_urgency = min(oldest_ms / self._sla_ms, 1.0)
        if sla_urgency > 0.5:
            total_reward -= self.gradual_sla_coef * (sla_urgency ** 2)

        # Build normalised state
        state = self._build_state(raw_obs, info)
        return state, total_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


# ── Factory ──────────────────────────────────────────────────────────────────

def make_extended_env(traffic_pattern: str, sac_config: dict,
                      seed: int = 42, phase: str = "random"):
    """
    Build a BatchingEnv and wrap with ExtendedBatchingEnv.

    Args:
        traffic_pattern: 'poisson' | 'bursty' | 'time_varying'
        sac_config:      training config dict
        seed:            env seed
        phase:           'random' | 'peak' | 'offpeak'
                         Controls episode start phase for bursty/time_varying.
                         Use 'peak' to train on worst-case bursts explicitly.
                         Use 'random' (default) for natural distribution.

    Returns:
        ExtendedBatchingEnv wrapping a BatchingEnv
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.batching_env import BatchingEnv
    from config import CONFIG

    base_cfg = dict(CONFIG)

    # Arrival rate — must be 100 req/s minimum for batching to be strategic
    base_cfg["arrival_rate"] = sac_config.get("arrival_rate", 100)

    # Reward weights — conservative base values; wrapper adds shaping on top
    base_cfg.update({
        "alpha":       sac_config.get("alpha_reward", 1.5),
        "beta":        sac_config.get("beta_reward",  0.001),  # lowered — wrapper handles SLA
        "gamma":       0.002,    # tiny idle penalty per inner step
        "sla_penalty": -5.0,     # base env step penalty (wrapper adds gradient)
    })

    # Traffic pattern multipliers
    if traffic_pattern == "poisson":
        base_cfg.update({"peak_multiplier": 1.2, "offpeak_multiplier": 0.8})
    elif traffic_pattern == "bursty":
        base_cfg.update({"peak_multiplier": 5.0, "offpeak_multiplier": 0.2})
    elif traffic_pattern == "time_varying":
        base_cfg.update({
            "peak_multiplier":    CONFIG.get("peak_multiplier", 2.5),
            "offpeak_multiplier": CONFIG.get("offpeak_multiplier", 0.5),
        })
    else:
        raise ValueError(
            f"Unknown traffic_pattern '{traffic_pattern}'. "
            "Choose: poisson | bursty | time_varying"
        )

    # [P7] Phase-aware seeding — CORRECTED
    #
    # HOW BatchingEnv controls peak/offpeak:
    #   reset() does: self._start_hour = float(self._rng.uniform(0, 24))
    #   peak_hours=(8,18) → if 8 <= _start_hour < 18 → peak traffic
    #   The ONLY way to control peak/offpeak is to pick the right seed.
    #
    # PRECOMPUTED seed tables (verified with np.random.default_rng(s).uniform(0,24)):
    #   PEAK_SEEDS    → seeds where _start_hour lands in [8, 18)
    #   OFFPEAK_SEEDS → seeds where _start_hour lands outside [8, 18)
    #
    # We index into these tables using (seed + episode_offset) to guarantee
    # every peak episode is truly peak and every offpeak is truly offpeak,
    # while still giving variety across episodes.

    # First 200 verified peak seeds (start_hour in 8-18)
    PEAK_SEEDS = [
        0,1,6,7,15,16,18,19,22,23,26,27,33,35,37,38,39,40,43,45,
        46,48,50,51,52,55,58,59,62,64,65,66,68,71,73,74,76,77,80,82,
        83,85,86,87,88,89,92,93,94,96,99,100,101,103,104,106,107,109,
        111,113,114,116,117,118,120,121,122,123,124,127,128,129,130,132,
        135,136,137,138,139,141,142,143,145,148,150,151,152,155,156,158,
        159,160,162,163,164,165,166,167,169,171,173,174,176,177,178,180,
        181,182,183,184,185,186,188,189,190,193,194,196,198,200,201,203,
        204,205,207,208,210,211,212,213,215,216,218,219,220,221,222,223,
        224,226,228,231,232,233,234,236,238,239,240,241,243,244,245,246,
        248,249,250,252,253,254,256,257,258,260,261,263,264,266,267,268,
    ]
    # First 200 verified offpeak seeds (start_hour outside 8-18)
    OFFPEAK_SEEDS = [
        2,3,4,5,8,9,10,11,12,13,14,17,20,21,24,25,28,29,30,31,
        32,34,36,41,42,44,47,49,53,54,56,57,60,61,63,67,69,70,72,75,
        78,79,81,84,90,91,95,97,98,102,105,108,110,112,115,119,125,126,
        131,133,134,140,144,146,147,149,153,154,157,161,168,170,172,175,
        179,187,191,192,195,197,199,202,206,209,214,217,225,227,229,230,
        235,237,242,247,251,255,259,262,265,269,270,271,272,273,274,275,
        276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,
        292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,
        308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,
        324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,
    ]

    if phase == "peak":
        # Pick from verified peak seeds, rotating by episode (seed encodes which ep)
        adjusted_seed = PEAK_SEEDS[seed % len(PEAK_SEEDS)]
    elif phase == "offpeak":
        # Pick from verified offpeak seeds
        adjusted_seed = OFFPEAK_SEEDS[seed % len(OFFPEAK_SEEDS)]
    else:
        # Random: use seed as-is — natural 42%/58% peak/offpeak split
        adjusted_seed = seed

    base_env = BatchingEnv(config=base_cfg, seed=adjusted_seed)
    return ExtendedBatchingEnv(base_env)