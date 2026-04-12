"""
agent/train_sac.py

Discrete SAC training script for the Dynamic Request Batching agent.

Why SAC alongside PPO
─────────────────────
  • Off-policy: replay buffer means every transition is reused many times,
    giving dramatically better sample efficiency than PPO's on-policy rollouts.
  • Maximum-entropy objective: the temperature α is auto-tuned so the agent
    maintains a target entropy level — no manual ent_coef search needed.
  • Same deployment interface: predict(obs) → int, stateless, same VecNormalize
    wrapper, same evaluate.py comparison pipeline.

Usage
─────
    python agent/train_sac.py

Outputs
───────
    models/sac_final.pt         – actor + critic weights + temperature
    models/sac_vecnorm.pkl      – VecNormalize statistics (required for deploy)
"""

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env.batching_env import BatchingEnv
from agent.discrete_sac import DiscreteSAC
from config import CONFIG, SAC_CONFIG

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR      = os.path.join(ROOT, "models")
FINAL_MODEL     = os.path.join(MODELS_DIR, "sac_final")
VECNORM_PATH    = os.path.join(MODELS_DIR, "sac_vecnorm.pkl")
os.makedirs(MODELS_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 1_000_000


def make_env(seed: int = 0):
    def _init():
        return Monitor(BatchingEnv(seed=seed))
    return _init


def train():
    print("=" * 64)
    print("  Dynamic Request Batching — Discrete SAC Training")
    print("=" * 64)
    print(f"  Timesteps      : {TOTAL_TIMESTEPS:,}")
    print(f"  Obs dims       : 8  (added rate_trend)")
    print(f"  Buffer size    : {SAC_CONFIG['buffer_size']:,}")
    print(f"  Network        : {SAC_CONFIG['net_arch']}")
    print(f"  Learning starts: {SAC_CONFIG['learning_starts']:,}")
    print(f"  VecNormalize   : enabled  (norm_obs=True, norm_reward=False)")
    print("=" * 64 + "\n")

    # ── Normalised environment ─────────────────────────────────────────────
    # norm_reward=False: SAC with a replay buffer is sensitive to reward
    # non-stationarity that arises when the running reward stats evolve while
    # old transitions are replayed.  Observation normalisation is safe and
    # critical for stable learning across the 8 differently-scaled features.
    vec_env = DummyVecEnv([make_env(seed=0)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = True,
        norm_reward = False,   # keep raw rewards in replay buffer
        clip_obs    = 10.0,
    )

    # ── Build SAC agent ────────────────────────────────────────────────────
    obs_dim   = vec_env.observation_space.shape[0]
    n_actions = vec_env.action_space.n

    sac = DiscreteSAC(obs_dim=obs_dim, n_actions=n_actions, cfg=SAC_CONFIG)

    print(f"Actor network  : {sac.actor}")
    print(f"Critic network : {sac.critic}\n")

    # ── Train ──────────────────────────────────────────────────────────────
    t0 = time.time()
    sac.learn(vec_env, total_timesteps=TOTAL_TIMESTEPS, log_freq=20_000)
    elapsed = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────────────
    sac.save(FINAL_MODEL)
    vec_env.save(VECNORM_PATH)

    print("\n" + "=" * 64)
    print("  Training complete")
    print("=" * 64)
    print(f"  Wall time     : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    print(f"  Throughput    : {TOTAL_TIMESTEPS / elapsed:,.0f} steps/s")
    print(f"  Gradient steps: {sac._n_updates:,}")
    print(f"  Final alpha   : {sac.alpha.item():.4f}")
    print(f"  Model         : {FINAL_MODEL}.pt")
    print(f"  VecNorm stats : {VECNORM_PATH}")
    print("=" * 64)

    vec_env.close()
    return sac


if __name__ == "__main__":
    train()
