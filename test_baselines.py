"""
test_baselines.py

Integration test and baseline sanity check.

1. gymnasium.utils.env_checker.check_env — validates Gymnasium API compliance.
2. Evaluates CloudflareBaseline and GreedyBatchBaseline over 10 episodes each.
3. Sanity assertions: no negative latencies, SLA rate in [0,1], etc.

Usage
─────
    python test_baselines.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gymnasium.utils.env_checker import check_env

from env.batching_env import BatchingEnv
from baselines.cloudflare_formula import (
    CloudflareBaseline,
    GreedyBatchBaseline,
    evaluate_baseline,
)
from config import CONFIG

N_EPISODES  = 10
SEED_OFFSET = 42


def main():
    # ── 1. Gymnasium spec check ──────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Gymnasium check_env(BatchingEnv())")
    print("=" * 60)
    env_check = BatchingEnv()
    check_env(env_check, warn=True, skip_render_check=True)
    env_check.close()
    print("  check_env passed — environment is Gymnasium-compliant.\n")

    # ── 2. Verify 7-dim observation ──────────────────────────────────────
    env_tmp = BatchingEnv()
    obs, _ = env_tmp.reset(seed=0)
    assert obs.shape == (8,), f"Expected obs dim 8, got {obs.shape}"
    print(f"  Observation shape  : {obs.shape}  ✓")
    obs2, r, *_ = env_tmp.step(0)
    assert obs2.shape == (8,), f"Post-step obs dim mismatch: {obs2.shape}"
    print(f"  Post-step obs shape: {obs2.shape}  ✓\n")
    env_tmp.close()

    # ── 3. Evaluate baselines ────────────────────────────────────────────
    print("=" * 60)
    print(f"Step 3: Baseline evaluation ({N_EPISODES} episodes each)")
    print("=" * 60)

    env = BatchingEnv()

    baselines = {
        "Cloudflare":  CloudflareBaseline(),
        "GreedyBatch": GreedyBatchBaseline(),
    }

    results = {}
    for name, agent in baselines.items():
        print(f"  Evaluating {name:<12} …", end=" ", flush=True)
        mean_r, std_r, _, _, sla_viol = evaluate_baseline(
            agent, env, n_episodes=N_EPISODES, seed_offset=SEED_OFFSET
        )
        results[name] = (mean_r, std_r, sla_viol)
        print(f"mean_reward = {mean_r:+.0f}  sla_viol = {sla_viol:.2%}")

    env.close()

    # ── 4. Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Baseline Comparison")
    print("=" * 60)
    hdr = f"{'Agent':<14}  {'Mean Reward':>12}  {'Std':>8}  {'SLA Viol%':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, (mr, sr, sv) in results.items():
        print(f"{name:<14}  {mr:>+12.0f}  {sr:>8.0f}  {sv:>9.2%}")
    print("=" * 60)

    # ── 5. Sanity assertions ─────────────────────────────────────────────
    for name, (mr, sr, sv) in results.items():
        assert isinstance(mr, float),    f"{name}: mean_reward not float"
        assert sr >= 0,                  f"{name}: std_reward < 0"
        assert 0.0 <= sv <= 1.0,         f"{name}: sla_viol_rate out of [0,1]"
        # Cloudflare should be noticeably better than GreedyBatch
        # (has higher avg batch size, lower dispatch overhead)

    # Key assertion: Cloudflare should outscore GreedyBatch
    cf_r  = results["Cloudflare"][0]
    gb_r  = results["GreedyBatch"][0]
    assert cf_r > gb_r, (
        f"Cloudflare ({cf_r:.0f}) should beat GreedyBatch ({gb_r:.0f}). "
        "If this fails, the reward shaping or baseline may need adjustment."
    )
    print(f"\n  Cloudflare > GreedyBatch  ({cf_r:.0f} vs {gb_r:.0f})  ✓")
    print("  All sanity checks passed.\n")


if __name__ == "__main__":
    main()
