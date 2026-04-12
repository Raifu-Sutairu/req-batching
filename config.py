# config.py
# Central configuration for the Dynamic Request Batching RL project.
#
# Problem: An inference-serving node (e.g. GPU running a transformer model)
# receives a continuous stream of requests. A batching layer decides every
# decision_interval_ms whether to DISPATCH the queued requests now or WAIT
# for the queue to grow (larger batches = better GPU utilisation).
#
# The SLA constraint: TOTAL latency (queue wait + GPU processing) must stay
# under max_latency_ms for every request. The trade-off the agent must learn:
#   too aggressive → many tiny batches → high dispatch overhead, bad efficiency
#   too conservative → large batches → high latency, SLA violations

CONFIG = {
    # ── Environment ───────────────────────────────────────────────────────────
    "max_batch_size":          512,    # Hard cap on dispatch size (GPU memory)
    "min_batch_size":          8,      # Below this batch size GPU is inefficient
    "arrival_rate":            2_000,  # Base Poisson rate (req/s) — realistic for
                                       # one inference node serving a social platform
    "episode_ms":              60_000, # 60 s of simulated time per episode
    "decision_interval_ms":    10,     # Agent decides at 100 Hz

    # ── SLA ──────────────────────────────────────────────────────────────────
    # SLA is on TOTAL latency: time in queue + GPU processing time.
    # This is what real production contracts specify (client-perceived latency).
    "max_latency_ms":          500,    # P99 total latency budget (ms)

    # ── GPU Processing Model ──────────────────────────────────────────────────
    # Approximates a mid-size transformer on a single A10G/A100 GPU.
    # processing_time(n) = gpu_base_ms + gpu_per_request_ms * n   (linear regime)
    # Typical numbers from NVIDIA Triton benchmarks:
    #   batch=1:   ~9 ms   (kernel launch dominates)
    #   batch=32:  ~12 ms  (good utilisation)
    #   batch=128: ~24 ms  (memory-bandwidth bound)
    #   batch=256: ~40 ms  (saturation)
    "gpu_base_ms":             8.0,    # Fixed overhead: kernel launch + DMA copy
    "gpu_per_request_ms":      0.12,   # Marginal compute per request (linear fit)

    # ── Traffic Variation (Time-of-Day) ──────────────────────────────────────
    # The agent is trained across the full 24-h rate distribution. It must learn
    # to be aggressive (serve quickly) at peak and patient (accumulate) off-peak.
    "peak_hours":              (9, 23),  # Extended peak window
    "peak_multiplier":         2.5,      # → 5 000 req/s at peak
    "offpeak_multiplier":      0.2,      # → 400 req/s at night

    # ── Reward Shaping ────────────────────────────────────────────────────────
    # Business objective: maximise requests served, minimise latency violations.
    #
    # SERVE action reward:
    #   r = alpha * batch_size
    #     - beta  * oldest_wait_ms          ← urgency while accumulating
    #     - dispatch_cost                   ← fixed GPU launch overhead
    #     - sla_penalty_per_ms * Σ(violation_ms per request that breaches SLA)
    #
    # WAIT action reward:
    #   r = -beta * oldest_wait_ms          ← growing impatience cost
    #     - idle_penalty    (if queue empty) ← lost capacity
    #
    # Key tension: dispatch_cost forces the agent to batch ≥ 5 requests before
    # serving is profitable. beta forces it not to sit on old requests too long.
    "alpha":               1.0,     # Revenue per request served
    "beta":                0.003,   # Per-ms urgency penalty while waiting / serving
    "dispatch_cost":       5.0,     # Fixed cost per dispatch (~5 requests worth)
    "idle_penalty":        0.5,     # Per-step penalty when queue is empty
    "sla_penalty_per_ms":  0.3,     # Penalty per ms of total-latency SLA violation
                                    # (per request). 100ms over SLA = -30 per request.
}

# ── SAC Hyperparameters ───────────────────────────────────────────────────────
SAC_CONFIG = {
    "learning_rate":          3e-4,
    "buffer_size":            300_000,   # replay buffer capacity
    "batch_size":             256,
    "gamma":                  0.99,
    "tau":                    0.005,     # soft target update coefficient
    "learning_starts":        5_000,     # random policy warm-up steps
    "train_freq":             1,         # gradient update every N env steps
    "gradient_steps":         1,         # gradient updates per train_freq steps
    "net_arch":               [128, 128],
    "target_entropy_ratio":   0.5,       # H_target = log(n_actions) * ratio
                                        # 0.5 ≈ 0.347 nats for binary — keeps
                                        # policy at ~75/25 at the decision boundary
    "reward_scale":           1e-3,     # scale stored rewards so Q-values ≈ 3–5
                                        # α_max=1.0 → entropy = 0.347 = ~10% of Q
                                        # keeps entropy regularisation numerically balanced
}

# ── PPO Hyperparameters ────────────────────────────────────────────────────────
PPO_CONFIG = {
    "learning_rate":   3e-4,
    "n_steps":         2048,    # Rollout steps per env before each gradient update
    "batch_size":      256,     # Minibatch size (larger = more stable gradients)
    "n_epochs":        10,      # Gradient epochs per rollout
    "gamma":           0.99,    # Discount factor
    "gae_lambda":      0.95,    # GAE lambda
    "clip_range":      0.2,     # PPO clip parameter
    "ent_coef":        0.005,   # Entropy bonus (small: prevent degenerate policy)
    "vf_coef":         0.5,     # Value function loss weight
    "max_grad_norm":   0.5,
    "net_arch":        [128, 128],  # Policy and value network hidden layers
}

# ── D3QN Hyperparameters ──────────────────────────────────────────────────────
# Dueling Double DQN + Prioritized Experience Replay + n-step returns.
D3QN_CONFIG = {
    "learning_rate":           3e-4,
    "buffer_size":             500_000,  # larger than SAC: PER needs diversity
    "batch_size":              256,
    "gamma":                   0.99,
    "tau":                     0.005,    # soft target update coefficient
    "learning_starts":         5_000,    # random warm-up steps
    "train_freq":              4,        # update every 4 env steps
    "gradient_steps":          1,
    "net_arch":                [128, 128],
    "reward_scale":            1e-3,     # same scaling as SAC for Q-value stability
    # ε-greedy exploration schedule
    "exploration_initial_eps": 1.0,
    "exploration_final_eps":   0.02,     # 2% noise at convergence
    "exploration_fraction":    0.15,     # anneal over first 15% of timesteps
    # Prioritized Experience Replay
    "per_alpha":               0.6,      # prioritization exponent: 0=uniform, 1=full
    "per_beta_start":          0.4,      # IS-weight correction start (low bias early)
    "per_beta_end":            1.0,      # IS-weight correction end  (fully unbiased)
    "per_eps":                 1e-6,     # min priority to prevent zero sampling
    # n-step returns
    "n_step":                  3,        # 3-step: covers the wait→accumulate→serve cycle
}

# ── Training Settings ──────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "total_timesteps":   1_000_000,  # 1M steps for strong convergence
    "n_envs":            4,          # Parallel environments
    "eval_freq":         20_000,     # Steps between evaluations
    "checkpoint_freq":   100_000,    # Steps between model checkpoints
    "n_eval_episodes":   10,
}

# ── Experiment Presets ─────────────────────────────────────────────────────────
# Vary traffic regime to evaluate policy robustness. The agent is trained on
# "standard" (with full time-of-day variation). These presets are eval-only.
EXPERIMENT_CONFIGS = {
    "off_peak":   {**CONFIG, "arrival_rate": 400,    "peak_multiplier": 1.5},
    "standard":   CONFIG,
    "peak_load":  {**CONFIG, "arrival_rate": 5_000,  "peak_multiplier": 3.0},
    "viral_spike":{**CONFIG, "arrival_rate": 20_000, "max_batch_size":  1024},
}
