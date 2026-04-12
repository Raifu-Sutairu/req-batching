"""
agent/train_d3qn.py

Train D3QN (Dueling Double DQN + PER + n-step returns) for dynamic request batching.

Usage:
    python agent/train_d3qn.py
"""

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.batching_env import BatchingEnv
from agent.d3qn import D3QN, DuelingQNetwork
from config import D3QN_CONFIG, TRAIN_CONFIG


def make_env(seed: int = 0):
    def _init():
        return BatchingEnv()
    return _init


if __name__ == "__main__":
    total_steps = TRAIN_CONFIG["total_timesteps"]

    env_fn  = make_env(seed=0)
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    sample_env = BatchingEnv()
    obs_dim    = sample_env.observation_space.shape[0]
    n_actions  = sample_env.action_space.n
    sample_env.close()

    agent = D3QN(obs_dim=obs_dim, n_actions=n_actions, cfg=D3QN_CONFIG)

    print("=" * 68)
    print("  Dynamic Request Batching — D3QN Training")
    print("=" * 68)
    print(f"  Timesteps      : {total_steps:,}")
    print(f"  Obs dims       : {obs_dim}  (includes rate_trend)")
    print(f"  Buffer size    : {D3QN_CONFIG['buffer_size']:,}  (PER sum-tree)")
    print(f"  Network        : {D3QN_CONFIG['net_arch']}")
    print(f"  n-step returns : {D3QN_CONFIG['n_step']}")
    print(f"  PER alpha      : {D3QN_CONFIG['per_alpha']}")
    print(f"  PER beta       : {D3QN_CONFIG['per_beta_start']} → {D3QN_CONFIG['per_beta_end']}")
    print(f"  ε schedule     : {D3QN_CONFIG['exploration_initial_eps']} → "
          f"{D3QN_CONFIG['exploration_final_eps']}  "
          f"(over {D3QN_CONFIG['exploration_fraction']*100:.0f}% of steps)")
    print(f"  Learning starts: {D3QN_CONFIG['learning_starts']:,}")
    print(f"  VecNormalize   : enabled  (norm_obs=True, norm_reward=False)")
    print("=" * 68)
    print()
    print("Online (Dueling) network:", agent.online)
    print()

    t0 = time.time()
    agent.learn(vec_env, total_timesteps=total_steps, log_freq=20_000)
    elapsed = time.time() - t0

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    agent.save(os.path.join(ROOT, "models", "d3qn_final"))
    vec_env.save(os.path.join(ROOT, "models", "d3qn_vecnorm.pkl"))

    print("\n" + "=" * 68)
    print("  Training complete")
    print("=" * 68)
    print(f"  Wall time     : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    print(f"  Throughput    : {total_steps/elapsed:.0f} steps/s")
    print(f"  Gradient steps: {agent._n_updates:,}")
    print(f"  Model         : {os.path.join(ROOT, 'models', 'd3qn_final.pt')}")
    print(f"  VecNorm stats : {os.path.join(ROOT, 'models', 'd3qn_vecnorm.pkl')}")
    print("=" * 68)

    vec_env.close()
