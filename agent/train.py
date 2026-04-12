"""
agent/train.py

PPO training script for the Dynamic Request Batching agent.

Design choices
──────────────
* VecNormalize   – normalises observations and rewards online.  Critical for
                   stable PPO convergence when state dimensions have very
                   different scales (pending ∈ [0,512] vs fill_ratio ∈ [0,1]).
                   Statistics are saved alongside the model for deployment.

* 4 parallel envs – gives 4× data throughput; each env starts a random hour
                    so the agent trains across the full time-of-day distribution
                    in every rollout batch.

* [128, 128] MLP  – larger than the baseline [64, 64] to capture the non-linear
                    interaction between arrival rate, urgency, and fill ratio.

* 1 000 000 steps – enough for the agent to see all traffic regimes many times
                    and converge to a stable policy.

Usage
─────
    python agent/train.py

Outputs
───────
    models/ppo_final.zip         – final policy weights
    models/ppo_vecnorm.pkl       – VecNormalize statistics (required for deploy)
    models/best/best_model.zip   – best checkpoint by eval reward
    tensorboard_logs/            – training curves
"""

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from env.batching_env import BatchingEnv
from config import CONFIG, PPO_CONFIG, TRAIN_CONFIG

# ── Directories ───────────────────────────────────────────────────────────────
TENSORBOARD_DIR  = os.path.join(ROOT, "tensorboard_logs")
MODELS_DIR       = os.path.join(ROOT, "models")
BEST_MODEL_DIR   = os.path.join(MODELS_DIR, "best")
CHECKPOINT_DIR   = os.path.join(MODELS_DIR, "checkpoints")

FINAL_MODEL_PATH  = os.path.join(MODELS_DIR, "ppo_final")
VECNORM_PATH      = os.path.join(MODELS_DIR, "ppo_vecnorm.pkl")

for d in [TENSORBOARD_DIR, BEST_MODEL_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)


def make_env_fn(seed: int = 0):
    """Factory: returns a Monitor-wrapped BatchingEnv."""
    def _init():
        return Monitor(BatchingEnv(seed=seed))
    return _init


def train():
    n_envs        = TRAIN_CONFIG["n_envs"]
    total_steps   = TRAIN_CONFIG["total_timesteps"]
    eval_freq     = TRAIN_CONFIG["eval_freq"]
    ckpt_freq     = TRAIN_CONFIG["checkpoint_freq"]
    n_eval_ep     = TRAIN_CONFIG["n_eval_episodes"]

    print("=" * 64)
    print("  Dynamic Request Batching — PPO Training")
    print("=" * 64)
    print(f"  Timesteps      : {total_steps:,}")
    print(f"  Parallel envs  : {n_envs}")
    print(f"  Network        : {PPO_CONFIG['net_arch']}")
    print(f"  Obs dims       : 8  (urgency_ratio + rate_trend)")
    print(f"  VecNormalize   : enabled")
    print(f"  TensorBoard    : tensorboard --logdir {TENSORBOARD_DIR}")
    print("=" * 64)

    # ── Training environments ─────────────────────────────────────────────
    train_env = make_vec_env(
        env_id=BatchingEnv,
        n_envs=n_envs,
        seed=0,
        wrapper_class=Monitor,
        wrapper_kwargs={"filename": None},
    )

    # VecNormalize: normalises observations (mean=0, std=1) and clips rewards.
    # norm_reward=True during training only; disabled at eval/inference time.
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=PPO_CONFIG["gamma"],
    )

    # ── Eval environment (same normaliser, rewards NOT normalised) ─────────
    eval_env  = VecNormalize(
        make_vec_env(BatchingEnv, n_envs=1, seed=9999, wrapper_class=Monitor,
                     wrapper_kwargs={"filename": None}),
        norm_obs=True,
        norm_reward=False,   # raw rewards for fair comparison
        clip_obs=10.0,
        training=False,      # do not update running stats during eval
    )

    # ── PPO model ─────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(
            pi=PPO_CONFIG["net_arch"],
            vf=PPO_CONFIG["net_arch"],
        )
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate   = PPO_CONFIG["learning_rate"],
        n_steps         = PPO_CONFIG["n_steps"],
        batch_size      = PPO_CONFIG["batch_size"],
        n_epochs        = PPO_CONFIG["n_epochs"],
        gamma           = PPO_CONFIG["gamma"],
        gae_lambda      = PPO_CONFIG["gae_lambda"],
        clip_range      = PPO_CONFIG["clip_range"],
        ent_coef        = PPO_CONFIG["ent_coef"],
        vf_coef         = PPO_CONFIG["vf_coef"],
        max_grad_norm   = PPO_CONFIG["max_grad_norm"],
        policy_kwargs   = policy_kwargs,
        tensorboard_log = TENSORBOARD_DIR,
        verbose         = 1,
    )

    print(f"\nPolicy network:\n{model.policy}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path  = BEST_MODEL_DIR,
        log_path              = os.path.join(MODELS_DIR, "eval_logs"),
        eval_freq             = max(eval_freq // n_envs, 1),
        n_eval_episodes       = n_eval_ep,
        deterministic         = True,
        render                = False,
        verbose               = 1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = max(ckpt_freq // n_envs, 1),
        save_path   = CHECKPOINT_DIR,
        name_prefix = "ppo_batching",
        verbose     = 1,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps     = total_steps,
        callback            = callbacks,
        reset_num_timesteps = True,
        tb_log_name         = "PPO_batching",
        progress_bar        = True,
    )
    elapsed = time.time() - t0

    # ── Save model + normalisation stats ──────────────────────────────────
    model.save(FINAL_MODEL_PATH)
    train_env.save(VECNORM_PATH)

    print("\n" + "=" * 64)
    print("  Training complete")
    print("=" * 64)
    print(f"  Wall time        : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    print(f"  Throughput       : {total_steps / elapsed:,.0f} steps/s")
    print(f"  Final model      : {FINAL_MODEL_PATH}.zip")
    print(f"  VecNorm stats    : {VECNORM_PATH}")
    print(f"  Best checkpoint  : {BEST_MODEL_DIR}/best_model.zip")
    print(f"  TensorBoard      : tensorboard --logdir {TENSORBOARD_DIR}")
    print("=" * 64)

    train_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train()
