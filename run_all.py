"""
run_all.py

Orchestrator script: runs the full RL batching pipeline end-to-end.

Steps:
  1. Validate environment (gymnasium check_env)
  2. Evaluate baselines (Random, Greedy, Cloudflare) — 10 episodes each
  3. Train PPO agent (500k timesteps)
  4. Generate comparison plots (PPO vs all baselines, 30 episodes each)
  5. Launch live demo (real-time animation)
  6. Train SAC + LSTM + PER agent (400 episodes, all traffic patterns)
  7. Evaluate SAC and generate comparison plots (SAC vs PPO vs baselines)

Each step is wrapped in a try/except so a failure doesn't stop the pipeline.
Run from project root:
    python3 run_all.py

To run only SAC steps:
    python3 run_all.py --sac-only

To skip the live demo (useful on headless servers):
    python3 run_all.py --no-demo
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

def step1_validate():
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
# Step 2 — Evaluate baselines
# ─────────────────────────────────────────────────────────────────────────────

def step2_baselines():
    _header(2, "Evaluating baselines...")
    try:
        import numpy as np
        from env.batching_env import BatchingEnv
        from baselines.cloudflare_formula import CloudflareBaseline, evaluate_baseline
        from baselines.random_agent import RandomAgent
        from baselines.greedy_agent import GreedyAgent
        from config import CONFIG

        N = 10
        env = BatchingEnv()

        agents = {
            "Cloudflare": CloudflareBaseline(
                max_latency_ms=CONFIG["max_latency_ms"], seed=42),
            "Random":     RandomAgent(seed=42),
            "Greedy":     GreedyAgent(),
        }

        print(f"\n  {'Agent':<14}  {'Mean Reward':>12}  {'Std':>8}  {'Mean Latency (ms)':>18}")
        print(f"  {'-'*14}  {'-'*12}  {'-'*8}  {'-'*18}")

        for name, agent in agents.items():
            mean_r, std_r, mean_lat = evaluate_baseline(agent, env, n_episodes=N)
            print(f"  {name:<14}  {mean_r:>+12.2f}  {std_r:>8.2f}  {mean_lat:>18.2f}")

        env.close()
        _ok("Baseline evaluation complete.")
    except Exception as exc:
        _err(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Train PPO agent
# ─────────────────────────────────────────────────────────────────────────────

def step3_train():
    _header(3, "Training PPO agent...")
    try:
        from agent.train import train
        model = train()
        _ok("Training complete. Model saved.")
    except Exception as exc:
        _err(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Generate comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def step4_evaluate():
    _header(4, "Generating comparison plots...")
    try:
        import os
        from agent.evaluate import run_all_agents, generate_figure, print_summary_table, PLOT_PATH

        best_path  = os.path.join(ROOT, "models", "best", "best_model")
        final_path = os.path.join(ROOT, "models", "ppo_batching_final")

        if os.path.exists(best_path + ".zip") or os.path.exists(best_path):
            model_path = best_path
        elif os.path.exists(final_path + ".zip") or os.path.exists(final_path):
            model_path = final_path
        else:
            raise FileNotFoundError(
                "No trained model found. Run Step 3 (training) first."
            )

        print(f"  Using model: {model_path}\n")
        data = run_all_agents(model_path)
        print_summary_table(data)
        path = generate_figure(data)
        _ok(f"Comparison plot saved → {path}")
    except Exception as exc:
        _err(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Launch live demo
# ─────────────────────────────────────────────────────────────────────────────

def step5_demo():
    _header(5, "Launching live demo...")
    try:
        import os
        import matplotlib.pyplot as plt
        from demo.live_demo import build_demo

        best_path  = os.path.join(ROOT, "models", "best", "best_model")
        final_path = os.path.join(ROOT, "models", "ppo_batching_final")

        if os.path.exists(best_path + ".zip") or os.path.exists(best_path):
            model_path = best_path
        elif os.path.exists(final_path + ".zip") or os.path.exists(final_path):
            model_path = final_path
        else:
            raise FileNotFoundError(
                "No trained model found. Run Step 3 (training) first."
            )

        print(f"  Using model: {model_path}")
        print("  Close the animation window to exit.\n")
        fig, anim = build_demo(model_path, interval_ms=50)
        plt.show()
        _ok("Live demo closed.")
    except Exception as exc:
        _err(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Train SAC + LSTM + PER agent
# ─────────────────────────────────────────────────────────────────────────────

def step6_train_sac():
    _header(6, "Training SAC + LSTM + PER agent...")
    try:
        from config import SAC_CONFIG
        from sac_agent.train_sac import train, DEFAULT_CONFIG

        traffic_patterns = SAC_CONFIG.get("traffic_patterns",
                                          ["poisson", "bursty", "time_varying"])

        for traffic in traffic_patterns:
            print(f"\n  ── Traffic pattern: {traffic} ──")

            # train_sac.py uses its own key names (lr, gamma, seq_len, etc.)
            # SAC_CONFIG uses more descriptive names (learning_rate, etc.)
            # We start from DEFAULT_CONFIG and overlay only what SAC_CONFIG provides,
            # translating key names where they differ.
            run_config = {
                **DEFAULT_CONFIG,

                # ── Keys that match directly ───────────────────────────
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

                # ── Keys that are renamed in SAC_CONFIG ────────────────
                # SAC_CONFIG uses "learning_rate"; train_sac.py expects "lr"
                "lr": SAC_CONFIG.get("learning_rate",
                                     SAC_CONFIG.get("lr", DEFAULT_CONFIG["lr"])),

                # ── Per-run overrides ──────────────────────────────────
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
# Step 7 — Evaluate SAC and compare against PPO + baselines
# ─────────────────────────────────────────────────────────────────────────────

def step7_evaluate_sac():
    _header(7, "Evaluating SAC + LSTM + PER (vs PPO and baselines)...")
    try:
        import os
        from sac_agent.evaluate_sac import main as sac_eval_main
        import argparse

        traffic_patterns = ["poisson", "bursty", "time_varying"]

        # Resolve PPO model path — same logic as step4_evaluate
        best_path  = os.path.join(ROOT, "models", "best", "best_model")
        final_path = os.path.join(ROOT, "models", "ppo_batching_final")

        if os.path.exists(best_path + ".zip") or os.path.exists(best_path):
            ppo_model_path = best_path
        elif os.path.exists(final_path + ".zip") or os.path.exists(final_path):
            ppo_model_path = final_path
        else:
            ppo_model_path = ""
            print("  [WARN] No PPO model found — SAC will be compared against baselines only.")

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
                ppo_model=ppo_model_path,
                traffic=traffic,
                episodes=50,
            )

            sac_eval_main(eval_args)
            _ok(f"SAC evaluation complete for '{traffic}' traffic.")

        _ok("All SAC evaluation runs complete. Check results/ for plots.")

    except Exception as exc:
        _err(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamic Request Batching — Full Pipeline"
    )
    parser.add_argument(
        "--sac-only",
        action="store_true",
        help="Skip PPO steps (1–5) and run only SAC steps (6–7)"
    )
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Skip the live demo step (step 5) — useful on headless servers"
    )
    parser.add_argument(
        "--skip-ppo-train",
        action="store_true",
        help="Skip PPO training (step 3) but still run evaluation and SAC steps"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(SEP)
    print("  Dynamic Request Batching — Full Pipeline")
    print("  PPO (Steps 1–5)  +  SAC + LSTM + PER (Steps 6–7)")
    print(SEP)

    t_start = time.time()

    # ── PPO Pipeline (Steps 1–5) ───────────────────────────────────────
    if not args.sac_only:
        step1_validate()
        step2_baselines()

        if not args.skip_ppo_train:
            step3_train()

        step4_evaluate()

        if not args.no_demo:
            step5_demo()
        else:
            print(f"\n{SEP}")
            print("  Step 5: Live demo skipped (--no-demo flag set)")
            print(SEP)

    # ── SAC Pipeline (Steps 6–7) ───────────────────────────────────────
    step6_train_sac()
    step7_evaluate_sac()

    elapsed = time.time() - t_start
    print(f"\n{SEP}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Checkpoints → checkpoints_sac/")
    print(f"  Results     → results/")
    print(f"  Logs        → logs_sac/")
    print(SEP + "\n")


if __name__ == "__main__":
    main()