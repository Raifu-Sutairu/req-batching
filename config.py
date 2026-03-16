# config.py
# Central configuration for the Dynamic Request Batching RL project

CONFIG = {
    # Environment
    "max_batch_size": 100,       # Maximum number of requests in a batch
    "max_latency_ms": 500,       # SLA: oldest request must not exceed this (ms)
    "arrival_rate": 10,          # Base Poisson arrival rate (requests/second)
    "episode_ms": 60_000,        # Episode length in milliseconds (60 seconds)
    "decision_interval_ms": 10,  # How often the agent makes a decision (ms)

    # Reward shaping
    "alpha": 1.0,    # Reward multiplier for batch size (efficiency)
    "beta": 0.01,    # Penalty multiplier for oldest_wait_ms (latency)
    "gamma": 0.1,    # Penalty for idle (no requests, action=Wait)

    # SLA
    "sla_penalty": -5.0,  # Penalty when oldest request exceeds max_latency_ms

    # Traffic variation (time-of-day)
    "peak_hours": (8, 18),       # 08:00–18:00 → peak traffic window
    "peak_multiplier": 2.5,      # Lambda multiplier during peak hours
    "offpeak_multiplier": 0.5,   # Lambda multiplier during off-peak hours
}

# ───────────────────────────────────────────────────────────────────────────
# DQN Hyperparameters
# Compare directly against PPO_KWARGS in agent/train.py
# ───────────────────────────────────────────────────────────────────────────

DQN_CONFIG = {
    # Learning
    "learning_rate": 1e-4,         # DQN is more sensitive to LR than PPO
    "gamma": 0.99,                 # Discount factor — same as PPO for fair comparison
    "batch_size": 64,              # Minibatch size for Q-network updates — same as PPO

    # Replay buffer
    "buffer_size": 100_000,        # How many past (s,a,r,s') transitions to store
                                   # Key DQN innovation — breaks temporal correlations
    "learning_starts": 10_000,     # Steps of random action BEFORE learning begins
                                   # Fills the replay buffer with diverse data first

    # Target network
    "target_update_interval": 1000, # Every N steps, copy Q-network → Target Q-network
                                    # Prevents the Q-value target from "chasing itself"
                                    # (main stability trick of DQN vs vanilla Q-learning)
    "tau": 1.0,                    # 1.0 = hard update (copy weights completely)
                                   # <1.0 = soft/polyak update (blend weights gradually)

    # Exploration (ε-greedy)
    "exploration_fraction": 0.2,   # Fraction of training spent decaying ε
    "exploration_initial_eps": 1.0, # Start fully random (100% explore)
    "exploration_final_eps": 0.05,  # End at 5% random exploration

    # Network
    "net_arch": [64, 64],          # Same hidden layer size as PPO for fair comparison

    # Training
    "train_freq": 4,               # Train the Q-network every N environment steps
    "gradient_steps": 1,           # Gradient update steps per training call

    # Logging
    "verbose": 1,
    "tensorboard_log": "tensorboard_logs/",
}


RPPO_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    "lstm_hidden_size": 64,
    "n_lstm_layers": 1,
    "enable_critic_lstm": True,
    "n_steps": 128,
    "batch_size": 64,
    "n_epochs": 10,
    "net_arch": dict(pi=[64], vf=[64]),
}

# ───────────────────────────────────────────────────────────────────────────
# SAC + LSTM + PER Hyperparameters
# ───────────────────────────────────────────────────────────────────────────
#
# Design rationale:
#   - SAC is off-policy + entropy-regularised. It reuses experience from
#     the replay buffer (unlike PPO which discards after each update).
#     This makes it significantly more sample-efficient than PPO.
#
#   - LSTM seq_len=20: at decision_interval_ms=10ms, this covers
#     200ms of traffic history — enough to detect burst onsets and
#     regime changes without excessive memory overhead.
#
#   - PER alpha=0.6: moderate prioritisation. Pure uniform (0.0) ignores
#     transition importance; full priority (1.0) can destabilise training
#     by over-focusing on a small set of high-error transitions.
#
#   - target_entropy=-1.0: standard SAC recommendation of -dim(action_space).
#     With 2 actions (Wait/Serve), this is -1.0. The agent will maintain
#     ~nats of entropy — enough to stay exploratory without being random.
#
#   - tau=0.005: slow polyak averaging for target network updates.
#     Much slower than DQN_CONFIG's hard update (tau=1.0), which is
#     appropriate because SAC updates every step rather than every 1000.
#
# Tuning guide:
#   alpha_reward ↑  →  agent prefers larger batches (higher throughput)
#   beta_reward  ↑  →  agent more aggressively avoids long waits (lower latency)
#   seq_len      ↑  →  more temporal context but slower training
#   per_alpha    ↑  →  stronger prioritisation of surprising transitions

SAC_CONFIG = {
    # ── Environment interface ──────────────────────────────────────────
    # These must match CONFIG above so SAC and PPO use the same environment
    "state_dim":   6,       # [pending_requests, oldest_wait_ms, request_rate,
                            #  since_serve_ms, batch_fill_ratio, time_of_day]
    "action_dim":  2,       # 0 = Wait, 1 = Serve

    # ── Reward weights ─────────────────────────────────────────────────
    # Mirror CONFIG["alpha"] and CONFIG["beta"] for a fair comparison.
    # You can override these independently here to tune SAC separately.
    "alpha_reward": CONFIG["alpha"],     # batch efficiency weight  (= 1.0)
    "beta_reward":  CONFIG["beta"],      # latency penalty weight   (= 0.01)

    # ── SAC core ───────────────────────────────────────────────────────
    "learning_rate":    3e-4,   # Adam LR — same as RPPO for fair comparison
    "gamma":            0.99,   # discount factor — same as PPO/DQN
    "tau":              0.005,  # soft target update rate (polyak averaging)
                                # θ_target ← τ·θ + (1-τ)·θ_target each step
    "alpha_init":       0.2,    # initial entropy temperature
                                # quickly overridden by automatic tuning
    "target_entropy":  -1.0,   # desired policy entropy = -dim(action_space)
                                # auto-tuning keeps H(π) ≈ this value

    # ── LSTM architecture ──────────────────────────────────────────────
    "seq_len":       20,    # timesteps of history fed to LSTM
                            # 20 steps × 10ms = 200ms of traffic context
    "lstm_hidden":   256,   # LSTM hidden state size
    "fc_hidden":     128,   # FC layer size after LSTM output

    # ── Replay buffer (PER) ────────────────────────────────────────────
    "buffer_capacity":  100_000,  # max transitions stored
                                  # larger than DQN (64k) — SAC is off-policy
                                  # and benefits from diverse historical data
    "batch_size":       256,      # training minibatch size
                                  # larger than DQN/PPO — SAC is more stable
    "warm_up_steps":    1_000,    # random actions before training starts
                                  # fills buffer with diverse initial data
    "updates_per_step": 1,        # gradient updates per environment step

    # PER-specific
    "per_alpha":        0.6,      # prioritisation exponent
                                  # 0 = uniform sampling (standard replay)
                                  # 1 = fully prioritised (only high-error)
    "per_beta_start":   0.4,      # IS weight exponent — anneals to 1.0
                                  # corrects sampling bias from prioritisation
    "per_beta_frames":  100_000,  # steps over which beta anneals to 1.0

    # ── Training schedule ──────────────────────────────────────────────
    "num_episodes":   400,    # total training episodes
                              # fewer than PPO (500k steps) because SAC
                              # is more sample-efficient
    "max_steps":      1_000,  # max steps per episode
                              # at 10ms/step → 10 seconds per episode

    # ── Logging & checkpoints ──────────────────────────────────────────
    "log_interval":    10,    # print progress every N episodes
    "save_interval":   50,    # save checkpoint every N episodes
    "checkpoint_dir": "checkpoints_sac",
    "log_dir":        "logs_sac",

    # ── Traffic patterns to train on ───────────────────────────────────
    # Run one training job per pattern for the ablation study
    "traffic_patterns": ["poisson", "bursty", "time_varying"],

    # ── Reproducibility ────────────────────────────────────────────────
    # Use multiple seeds for statistically rigorous results in the report
    "seeds": [42, 123, 777],
    "seed":  42,   # default seed for single runs
}

# ───────────────────────────────────────────────────────────────────────────
# Experiment presets — change one key to explore different traffic regimes.
#
# Usage example:
#   from config import EXPERIMENT_CONFIGS
#   env = BatchingEnv(config=EXPERIMENT_CONFIGS["high_load"])
# ───────────────────────────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = {
    # Low-traffic scenario: sparse arrivals, agent must decide whether to wait
    # for a bigger batch or serve immediately to avoid idle-wait penalties.
    # Expected behaviour: agent learns to accumulate more before serving.
    "low_load": {
        **CONFIG,
        "arrival_rate": 5,           # half the default traffic
    },

    # Standard scenario: matches CONFIG exactly — the default training regime.
    "standard": {
        **CONFIG,
        "arrival_rate": 10,
    },

    # High-traffic scenario: backpressure stress-test.  The queue fills fast;
    # the agent must serve frequently to prevent SLA violations while still
    # batching enough to get meaningful efficiency rewards.
    "high_load": {
        **CONFIG,
        "arrival_rate": 50,          # 5× the default; queue saturates quickly
    },
}

# ───────────────────────────────────────────────────────────────────────────
# SAC Experiment presets
# Mirror EXPERIMENT_CONFIGS above so SAC is tested under identical conditions
# as PPO/DQN — required for a fair comparison in the report.
#
# Usage:
#   from config import SAC_EXPERIMENT_CONFIGS
#   config = SAC_EXPERIMENT_CONFIGS["high_load_bursty"]
# ───────────────────────────────────────────────────────────────────────────

SAC_EXPERIMENT_CONFIGS = {
    # Standard training — matches PPO's default environment
    "standard": {
        **SAC_CONFIG,
        "traffic_pattern": "poisson",
        "arrival_rate":    10,
    },

    # Bursty traffic — where LSTM memory provides the biggest advantage.
    # The agent must detect burst onsets and accumulate larger batches
    # during peaks, then serve quickly when bursts end.
    "bursty": {
        **SAC_CONFIG,
        "traffic_pattern": "bursty",
        "arrival_rate":    10,
    },

    # Time-varying traffic — tests whether LSTM can learn peak/idle cycles.
    # Peak hours (08:00–18:00) use peak_multiplier=2.5× arrival rate.
    "time_varying": {
        **SAC_CONFIG,
        "traffic_pattern": "time_varying",
        "arrival_rate":    10,
    },

    # High-load stress test — same as EXPERIMENT_CONFIGS["high_load"]
    # but run through SAC to compare against PPO under identical pressure.
    "high_load": {
        **SAC_CONFIG,
        "traffic_pattern": "poisson",
        "arrival_rate":    50,
    },

    # Throughput-focused tuning: higher alpha_reward → agent prefers
    # larger batches even at the cost of some extra latency.
    "throughput_focused": {
        **SAC_CONFIG,
        "traffic_pattern": "bursty",
        "alpha_reward":    2.0,   # 2× efficiency weight
        "beta_reward":     0.005, # halved latency penalty
    },

    # Latency-focused tuning: higher beta_reward → agent aggressively
    # avoids long waits, producing smaller but faster batches.
    "latency_focused": {
        **SAC_CONFIG,
        "traffic_pattern": "bursty",
        "alpha_reward":    0.5,   # halved efficiency weight
        "beta_reward":     0.02,  # 2× latency penalty
    },
}