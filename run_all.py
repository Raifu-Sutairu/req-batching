"""
run_all.py

Orchestrator script: runs the full SAC + LSTM + PER training pipeline end-to-end.

Steps:
  1. Validate environment (gymnasium check_env)
  2. Train SAC + LSTM + PER agent (for all traffic patterns)
  3. Evaluate SAC agent and baseline heuristics

Run from project root:
    python3 run_all.py
"""

import os
import sys
import traceback
import time
import argparse

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 64

def _header(step: int, msg: str):
    print(f"\n{SEP}")
    print(f"  Step {step}: {msg}")
    print(SEP)

def _ok(msg: str = ""):
    tag = f"  ✓  {msg}" if msg else "  ✓  Done."
    print(tag)

def _err(exc: Exception):
    print(f"\n  ✗  ERROR: {exc}")
    print("  Traceback:")
    for line in traceback.format_exc().splitlines():
        print(f"    {line}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Validate environment
# ─────────────────────────────────────────────────────────────────────────────

def validate_environment():
    _header(1, "Validating environment...")
    try:
        from gymnasium.utils.env_checker import check_env
        from env.batching_env import BatchingEnv

        env = BatchingEnv()
        check_env(env, skip_render_check=True)
        env.close()
        _ok("BatchingEnv passed gymnasium check_env.")
    except Exception as exc:
        _err(exc)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Train SAC + LSTM + PER agent
# ─────────────────────────────────────────────────────────────────────────────

def train_sac_agent():
    _header(2, "Training SAC + LSTM + PER agent...")
    try:
        from config import SAC_CONFIG
        from sac_agent.train_sac import train, DEFAULT_CONFIG

        traffic_patterns = SAC_CONFIG.get("traffic_patterns",
                                          ["poisson", "bursty", "time_varying"])

        for traffic in traffic_patterns:
            print(f"\n  ── Traffic pattern: {traffic} ──")

            # Overlay SAC_CONFIG onto DEFAULT_CONFIG
            run_config = {
                **DEFAULT_CONFIG,

                "gamma":            SAC_CONFIG.get("gamma",            DEFAULT_CONFIG["gamma"]),
                "tau":              SAC_CONFIG.get("tau",              DEFAULT_CONFIG["tau"]),
                "alpha_init":       SAC_CONFIG.get("alpha_init",       DEFAULT_CONFIG["alpha_init"]),
                "target_entropy":   None,   # fixed α — disable auto-tuning
                "seq_len":          SAC_CONFIG.get("seq_len",          DEFAULT_CONFIG["seq_len"]),
                "lstm_hidden":      SAC_CONFIG.get("lstm_hidden",      DEFAULT_CONFIG["lstm_hidden"]),
                "fc_hidden":        SAC_CONFIG.get("fc_hidden",        DEFAULT_CONFIG["fc_hidden"]),
                "buffer_capacity":  SAC_CONFIG.get("buffer_capacity",  DEFAULT_CONFIG["buffer_capacity"]),
                "batch_size":       SAC_CONFIG.get("batch_size",       DEFAULT_CONFIG["batch_size"]),
                "warm_up_steps":    SAC_CONFIG.get("warm_up_steps",    DEFAULT_CONFIG["warm_up_steps"]),
                "per_alpha":        SAC_CONFIG.get("per_alpha",        DEFAULT_CONFIG["per_alpha"]),
                "per_beta_start":   SAC_CONFIG.get("per_beta_start",   DEFAULT_CONFIG["per_beta_start"]),
                "per_beta_frames":  SAC_CONFIG.get("per_beta_frames",  DEFAULT_CONFIG["per_beta_frames"]),
                "updates_per_step": SAC_CONFIG.get("updates_per_step", DEFAULT_CONFIG["updates_per_step"]),
                "num_episodes":     SAC_CONFIG.get("num_episodes",     DEFAULT_CONFIG["num_episodes"]),
                "max_steps":        SAC_CONFIG.get("max_steps",        DEFAULT_CONFIG["max_steps"]),
                "log_interval":     SAC_CONFIG.get("log_interval",     DEFAULT_CONFIG["log_interval"]),
                "save_interval":    SAC_CONFIG.get("save_interval",    DEFAULT_CONFIG["save_interval"]),
                "seed":             SAC_CONFIG.get("seed",             DEFAULT_CONFIG["seed"]),
                "alpha_reward":     SAC_CONFIG.get("alpha_reward",     DEFAULT_CONFIG["alpha_reward"]),
                "beta_reward":      SAC_CONFIG.get("beta_reward",      DEFAULT_CONFIG["beta_reward"]),
                "lr":               SAC_CONFIG.get("learning_rate",    SAC_CONFIG.get("lr", DEFAULT_CONFIG["lr"])),

                "traffic_pattern": traffic,
                "checkpoint_dir":  f"checkpoints_sac/{traffic}",
                "log_dir":         f"logs_sac/{traffic}",
            }

            agent, logs = train(run_config)
            _ok(f"SAC training complete for '{traffic}' traffic.")

        _ok("All SAC training runs complete.")

    except Exception as exc:
        _err(exc)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Evaluate SAC vs In-line Baselines
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sac_agent():
    _header(3, "Evaluating SAC + LSTM + PER vs internal baselines...")
    try:
        from sac_agent.evaluate_sac import main as sac_eval_main
        import argparse

        traffic_patterns = ["poisson", "bursty", "time_varying"]

        for traffic in traffic_patterns:
            print(f"\n  ── Evaluating on '{traffic}' traffic ──")

            sac_model = os.path.join(
                ROOT, "checkpoints_sac", traffic, "sac_best.pth"
            )

            if not os.path.exists(sac_model):
                print(f"  [WARN] SAC model not found at {sac_model} — skipping.")
                continue

            # Build args namespace to reuse evaluate_sac.main()
            eval_args = argparse.Namespace(
                model=sac_model,
                ppo_model="",  # PPO removed
                traffic=traffic,
                peak_episodes=100,
                offpeak_episodes=100,
            )

            sac_eval_main(eval_args)
            _ok(f"SAC evaluation complete for '{traffic}' traffic.")

        _ok("All SAC evaluation runs complete. Check results/ for plots.")

    except Exception as exc:
        _err(exc)

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  Dynamic Request Batching — SAC+LSTM+PER Pipeline")
    print(SEP)

    t_start = time.time()

    validate_environment()
    train_sac_agent()
    evaluate_sac_agent()

    elapsed = time.time() - t_start
    print(f"\n{SEP}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Checkpoints → checkpoints_sac/")
    print(f"  Results     → results/")
    print(f"  Logs        → logs_sac/")
    print(SEP + "\n")

if __name__ == "__main__":
    main()