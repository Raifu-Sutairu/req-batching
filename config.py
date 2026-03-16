# config.py
# Central configuration for the Dynamic Request Batching RL project

CONFIG = {
    # ── Environment ─────────────────────────────────────────────────────────
    # Scaled to a realistic single GPU inference server node.
    # At 2 000 req/s base rate with a 10 ms decision tick, ~20 requests
    # arrive per tick on average (up to ~50 during peak hours).
    # This creates real batching pressure — the agent cannot just flush every
    # tick or it will pay the dispatch cost 100 times per second.
    "max_batch_size": 500,       # Static fallback (or max bounds limit)
    "dynamic_batching_enabled": True, # Enables capacity to scale with traffic
    "min_batch_size": 10,        # Minimum functional batch limit
    "max_possible_batch_size": 1000, # Absolute hardware capacity
    "target_service_time_ms": 200, # Desired base accumulation window for scaling
    "max_latency_ms": 500,       # SLA: oldest request must not exceed 500 ms
    "arrival_rate": 2_000,       # Base Poisson rate (req/s) — realistic for one
                                  # inference node serving a social media backend
                                  # (Instagram: ~10k-100k QPS globally, ~2k per node)
    "episode_ms": 60_000,        # 60 seconds of simulated time per episode
    "decision_interval_ms": 10,  # Agent decides every 10 ms (100 Hz)

    # ── Reward shaping ───────────────────────────────────────────────────────
    # alpha: reward per request served in a batch → encourages efficiency
    # beta:  penalty per ms of oldest_wait → encourages low latency
    # gamma: idle penalty when agent Waits with an empty queue
    # serve_dispatch_cost: FIXED cost every time the agent hits Serve,
    #   regardless of batch size. Represents GPU kernel launch overhead,
    #   memory copy, network round-trip setup, etc.
    #   This is the key parameter that makes the agent WANT to batch:
    #     Serve(batch=1):  1.0 - latency - 3.0  ← likely negative (bad)
    #     Serve(batch=5):  5.0 - latency - 3.0  ← break-even
    #     Serve(batch=20): 20.0 - latency - 3.0 ← clearly positive (good)
    "alpha": 1.0,
    "beta":  0.002,              # Smaller beta now because wait_ms values will
                                  # be larger at higher traffic — avoid dominating
    "gamma": 0.5,                # Stronger idle penalty — empty queue should be rare
    "serve_dispatch_cost": 3.0,  # Fixed overhead per Serve action.
                                  # Forces agent to accumulate ~3+ requests before serving.

    # ── SLA ─────────────────────────────────────────────────────────────────
    "sla_penalty": -50.0,        # Harsher SLA violation penalty at scale

    # ── Traffic variation (time-of-day) ──────────────────────────────────────
    "peak_hours": (9, 23),       # Extended peak window (morning scroll → late night)
    "peak_multiplier": 2.5,      # 5 000 req/s at peak — GPU near saturation
    "offpeak_multiplier": 0.2,   # 400 req/s at night — agent learns patience
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
    "enable_critic_lstm":True,
    "n_steps": 128,
    "batch_size": 64,
    "n_epochs":10,
    "net_arch": dict(pi=[64], vf=[64]),
}

# ───────────────────────────────────────────────────────────────────────────
# Experiment presets — change one key to explore different traffic regimes.
#
# Usage example:
#   from config import EXPERIMENT_CONFIGS
#   env = BatchingEnv(config=EXPERIMENT_CONFIGS["high_load"])
# ───────────────────────────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = {
    # ── Off-peak night traffic (e.g. 2 AM social media) ─────────────────────
    # ~400 req/s — very sparse. ~4 requests arrive per 10 ms tick.
    # Agent should accumulate longer to fill batches efficiently.
    "off_peak": {
        **CONFIG,
        "arrival_rate": 400,
        "peak_multiplier": 1.5,
    },

    # ── Standard daytime load ────────────────────────────────────────────────
    # 2 000 req/s — default CONFIG. ~20 requests per tick.
    # This is the primary training regime.
    "standard": CONFIG,

    # ── Peak hour stress test (e.g. Instagram at 7 PM) ───────────────────────
    # 5 000 req/s — GPU is under pressure. ~50 requests per tick.
    # Agent must balance batch efficiency vs SLA violations.
    "peak_load": {
        **CONFIG,
        "arrival_rate": 5_000,
        "peak_multiplier": 3.0,       # up to 15 000 req/s burst
        "sla_penalty": -100.0,        # harsher SLA at high load
    },

    # ── Viral event / traffic spike (e.g. Super Bowl moment) ─────────────────
    # 20 000 req/s — extreme backpressure test. Queue fills in milliseconds.
    # Agent must serve aggressively to avoid SLA violations.
    "viral_spike": {
        **CONFIG,
        "arrival_rate": 20_000,
        "max_batch_size": 1_000,      # GPU can handle larger batches
        "peak_multiplier": 4.0,
        "sla_penalty": -200.0,
    },
}
