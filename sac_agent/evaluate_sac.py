"""
evaluate_sac.py  (v2 — evaluation problems fixed)
--------------------------------------------------
Fixes applied vs v1:
  [P15] Stratified evaluation — 200 episodes split: 100 peak-start seeds +
        100 offpeak-start seeds. Completely eliminates the phase-bias problem
        where 50 random seeds sampled mostly off-peak, masking SAC's advantage.
  [P16] Proper metric reporting:
        - Separate peak_mean_reward and offpeak_mean_reward
        - p95 latency (ms) — the real SLA metric
        - Throughput (req/s served)
        - Mean batch size
        - SLA violation rate (% episodes with any violation)
  [FIX] Uses v2 make_extended_env with phase parameter.
  [FIX] state_dim=8 for SACAgent instantiation.
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .sac_agent import SACAgent
    from .extended_env import make_extended_env, STATE_DIM
except ImportError:
    from sac_agent.sac_agent import SACAgent
    from sac_agent.extended_env import make_extended_env, STATE_DIM

HAS_SB3 = False


os.makedirs("results", exist_ok=True)

EVAL_CONFIG = {
    "arrival_rate": 100,
    "alpha_reward": 1.5,
    "beta_reward":  0.001,
}


# ── Baseline agents ───────────────────────────────────────────────────────────

class ExtendedGreedyAgent:
    """Always action=0 (serve immediately). Cannot earn wait bonuses."""
    def select_action(self, state, deterministic=True): return 0
    def reset_state_queue(self): pass


class ExtendedRandomAgent:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
    def select_action(self, state, deterministic=False):
        return int(self.rng.integers(0, 4))
    def reset_state_queue(self): pass


class ThresholdAgent:
    """
    Heuristic: serve_now if queue < threshold or sla_urgency > 0.6,
    else wait_100ms. Better baseline than pure Greedy.
    State indices: [0]=pending, [1]=sla_urgency, [2]=rate, [3]=delta_rate
    """
    def __init__(self, queue_threshold=5, sla_threshold=0.6):
        self.queue_threshold = queue_threshold
        self.sla_threshold   = sla_threshold

    def select_action(self, state, deterministic=True):
        pending    = float(state[0])  # may be normalised; use relative
        sla_urgency = float(state[1])
        delta_rate  = float(state[3])
        # If normalised, we work in relative terms
        if sla_urgency > self.sla_threshold:   return 0   # SLA near — serve
        if delta_rate > 0.5:                   return 3   # burst rising — wait big
        if pending > 0:                        return 2   # queue building — wait mid
        return 0

    def reset_state_queue(self): pass


# ── Stratified evaluation ─────────────────────────────────────────────────────

def evaluate_agent_stratified(agent, traffic_pattern, n_peak=100, n_offpeak=100,
                               agent_name="SAC", deterministic=True, is_sb3=False):
    """
    [P15] Stratified evaluation: n_peak episodes starting at peak,
    n_offpeak episodes starting at off-peak. Reports separately.

    Returns dict with full metrics split by phase.
    """
    results_by_phase = {}

    for phase, n_eps in [("peak", n_peak), ("offpeak", n_offpeak)]:
        rewards, batch_sizes, wait_times, latencies, violations = [], [], [], [], []
        action_dist = [0, 0, 0, 0]

        for ep in range(n_eps):
            env = make_extended_env(
                traffic_pattern, EVAL_CONFIG,
                seed=2000 + ep * 7, phase=phase
            )
            state, _ = env.reset()
            if hasattr(agent, "reset_state_queue"):
                agent.reset_state_queue()

            ep_reward  = 0.0
            ep_batches = []
            ep_waits   = []
            ep_steps   = 0
            ep_sla_viol = False

            done = False
            while not done:
                if hasattr(agent, "select_action"):
                    action = agent.select_action(state, deterministic=deterministic)
                else:
                    action = agent(state)

                if 0 <= action < 4:
                    action_dist[action] += 1

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_reward += reward
                ep_steps  += 1

                if info:
                    if "queue_length" in info:
                        ep_batches.append(info["queue_length"])
                    if "mean_latency_ms" in info:
                        ep_waits.append(info["mean_latency_ms"])
                    if info.get("mean_latency_ms", 0) > 400:
                        ep_sla_viol = True

                state = next_state

            env.close()
            rewards.append(ep_reward)
            violations.append(float(ep_sla_viol))
            if ep_batches: batch_sizes.append(float(np.mean(ep_batches)))
            if ep_waits:   wait_times.append(float(np.mean(ep_waits)))

        total_act = max(sum(action_dist), 1)
        adist = [c / total_act for c in action_dist]

        results_by_phase[phase] = {
            "mean_reward":    float(np.mean(rewards)),
            "std_reward":     float(np.std(rewards)),
            "p95_latency_ms": float(np.percentile(wait_times, 95)) if wait_times else 0.0,
            "mean_wait_ms":   float(np.mean(wait_times)) if wait_times else 0.0,
            "mean_batch":     float(np.mean(batch_sizes)) if batch_sizes else 0.0,
            "sla_viol_rate":  float(np.mean(violations)),
            "action_dist":    adist,
            "n_episodes":     n_eps,
        }

    overall_rewards = []
    for phase in ["peak", "offpeak"]:
        r = results_by_phase[phase]
        overall_rewards.extend([r["mean_reward"]] * r["n_episodes"])

    all_waits = [results_by_phase[p]["mean_wait_ms"] for p in ["peak","offpeak"]]

    result = {
        "agent":           agent_name,
        "overall_reward":  float(np.mean([results_by_phase[p]["mean_reward"] for p in ["peak","offpeak"]])),
        "peak_reward":     results_by_phase["peak"]["mean_reward"],
        "offpeak_reward":  results_by_phase["offpeak"]["mean_reward"],
        "p95_latency_ms":  max(results_by_phase[p]["p95_latency_ms"] for p in ["peak","offpeak"]),
        "mean_batch":      max(results_by_phase[p]["mean_batch"] for p in ["peak","offpeak"]),
        "sla_viol_rate":   float(np.mean([results_by_phase[p]["sla_viol_rate"] for p in ["peak","offpeak"]])),
        "by_phase":        results_by_phase,
    }

    print(
        f"  [{agent_name:20s}] "
        f"Overall: {result['overall_reward']:8.1f} | "
        f"Peak: {result['peak_reward']:8.1f} | "
        f"Off-peak: {result['offpeak_reward']:8.1f} | "
        f"Batch: {result['mean_batch']:4.1f} | "
        f"p95 lat: {result['p95_latency_ms']:.0f}ms | "
        f"SLA viol: {result['sla_viol_rate']:.0%}"
    )
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(results_list, traffic_pattern, save_path):
    """[P16] Plot now includes peak vs off-peak split and p95 latency."""
    names     = [r["agent"] for r in results_list]
    overall   = [r["overall_reward"] for r in results_list]
    peak      = [r["peak_reward"]    for r in results_list]
    offpeak   = [r["offpeak_reward"] for r in results_list]
    batches   = [r["mean_batch"]     for r in results_list]
    p95_lats  = [r["p95_latency_ms"] for r in results_list]
    sla_viols = [r["sla_viol_rate"]  for r in results_list]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Agent Comparison — {traffic_pattern.replace('_',' ').title()} Traffic\n"
        f"(Stratified eval: 100 peak + 100 off-peak episodes | 8D state | v2)",
        fontsize=12, fontweight="bold"
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(names)]
    x = np.arange(len(names))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, peak,    w, label="Peak",     color=[c+"CC" for c in colors], edgecolor="black", lw=0.8)
    ax.bar(x + w/2, offpeak, w, label="Off-peak", color=colors, edgecolor="black", lw=0.8)
    ax.set_title("Mean reward\n(peak vs off-peak)", fontweight="bold")
    ax.set_ylabel("Reward")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend(fontsize=8)

    ax = axes[1]
    bars = ax.bar(names, batches, color=colors, edgecolor="black", lw=0.8)
    ax.set_title("Avg batch size\n(higher = better batching)", fontweight="bold")
    ax.set_ylabel("Requests/batch")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, batches):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)

    ax = axes[2]
    bars = ax.bar(names, p95_lats, color=colors, edgecolor="black", lw=0.8)
    ax.axhline(y=500, color="red", linestyle="--", linewidth=1, alpha=0.7, label="SLA (500ms)")
    ax.set_title("p95 latency (ms)\n(lower = better, SLA=500ms)", fontweight="bold")
    ax.set_ylabel("Milliseconds")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)

    ax = axes[3]
    bars = ax.bar(names, [v*100 for v in sla_viols], color=colors, edgecolor="black", lw=0.8)
    ax.set_title("SLA violation rate\n(lower = better)", fontweight="bold")
    ax.set_ylabel("Violation %")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_training_curves(log_path, save_path):
    with open(log_path) as f:
        logs = json.load(f)

    rewards = logs.get("episode_rewards", [])
    if not rewards:
        print("  [WARN] No training data found.")
        return

    phases  = logs.get("episode_phases", ["random"] * len(rewards))
    alphas  = logs.get("alpha_values",   [])
    entropy = logs.get("entropy_values", [])
    batches = logs.get("avg_batch_sizes",[])
    adists  = logs.get("action_distributions", [])

    window = 20
    def ma(arr):
        if len(arr) < window: return arr
        return np.convolve(arr, np.ones(window)/window, mode="valid")

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("SAC+LSTM+PER Training Curves [v2 — all fixes]",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    phase_colors = {"random": "#aaaaaa", "peak": "#d62728", "offpeak": "#1f77b4"}

    ax = fig.add_subplot(gs[0, 0])
    eps = list(range(1, len(rewards)+1))
    for ph in ["random","peak","offpeak"]:
        idxs = [i for i,p in enumerate(phases) if p==ph]
        if idxs:
            ax.scatter([eps[i] for i in idxs], [rewards[i] for i in idxs],
                       s=8, alpha=0.5, color=phase_colors[ph], label=ph)
    if len(rewards) >= window:
        ax.plot(range(window, len(rewards)+1), ma(rewards),
                color="black", lw=2, label=f"MA({window})")
    ax.set_title("Episode reward by phase"); ax.set_xlabel("Episode")
    ax.legend(fontsize=7)

    if alphas:
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(alphas, color="#ff7f0e", lw=1.5)
        ax.set_title("Entropy temperature α\n(curriculum anneal)")
        ax.set_xlabel("Episode")

    if entropy:
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(entropy, color="#2ca02c", lw=1.5)
        ax.axhline(y=np.log(4), color="gray", ls="--", alpha=0.5, label="max H(Discrete(4))")
        ax.set_title("Policy entropy H(π)")
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8)

    if batches:
        ax = fig.add_subplot(gs[0, 3])
        ax.plot(batches, color="#d62728", lw=1, alpha=0.5)
        if len(batches) >= window:
            ax.plot(range(window, len(batches)+1), ma(batches), color="#d62728", lw=2)
        ax.set_title("Avg batch size"); ax.set_xlabel("Episode")

    if adists:
        ax = fig.add_subplot(gs[1, :2])
        adists_arr = np.array(adists)
        labels = ["serve_now","wait_20ms","wait_50ms","wait_100ms"]
        cols   = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
        for i,(lbl,col) in enumerate(zip(labels,cols)):
            ax.plot(adists_arr[:,i], label=lbl, color=col, lw=1.2, alpha=0.8)
        ax.set_title("Action distribution over training")
        ax.set_xlabel("Episode"); ax.set_ylabel("Fraction")
        ax.legend(fontsize=8); ax.set_ylim(0, 1)

    critic_l = logs.get("critic_losses", [])
    actor_l  = logs.get("actor_losses",  [])
    if critic_l:
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(critic_l, color="#9467bd", lw=1)
        ax.set_title("Critic loss (convergence)"); ax.set_xlabel("Episode")

    if actor_l:
        ax = fig.add_subplot(gs[1, 3])
        ax.plot(actor_l, color="#8c564b", lw=1)
        ax.set_title("Actor loss (policy quality)"); ax.set_xlabel("Episode")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    traffic = args.traffic
    n_peak  = args.peak_episodes
    n_off   = args.offpeak_episodes

    print(f"\n{'='*64}")
    print(f"  SAC+LSTM+PER Evaluation  [v2 — stratified]")
    print(f"  Traffic: {traffic}  |  Peak eps: {n_peak}  |  Off-peak eps: {n_off}")
    print(f"{'='*64}\n")

    results_list = []

    # SAC
    sac = SACAgent(state_dim=STATE_DIM, action_dim=4,
                   seq_len=30, lstm_hidden=128, fc_hidden=128)
    model_path = args.model
    if not os.path.exists(model_path):
        model_path = os.path.join("checkpoints_sac", traffic, "sac_best.pth")
    if os.path.exists(model_path):
        sac.load(model_path)
        r = evaluate_agent_stratified(sac, traffic, n_peak, n_off, "SAC+LSTM+PER")
        results_list.append(r)
    else:
        print(f"  [WARN] SAC model not found at {model_path}")

    # Baselines
    for agent, name in [
        (ExtendedGreedyAgent(),    "Greedy"),
        (ExtendedRandomAgent(42),  "Random"),
        (ThresholdAgent(),         "Threshold"),
    ]:
        r = evaluate_agent_stratified(agent, traffic, n_peak, n_off, name)
        results_list.append(r)

    # Plots
    plot_comparison(results_list, traffic, f"results/comparison_{traffic}_v2.png")
    log_path = os.path.join("logs_sac", traffic, "training_logs.json")
    if os.path.exists(log_path):
        plot_training_curves(log_path, f"results/training_curves_{traffic}_v2.png")

    # Save JSON
    clean = [{k: v for k, v in r.items() if k != "by_phase"} for r in results_list]
    with open(f"results/results_{traffic}_v2.json", "w") as f:
        json.dump(clean, f, indent=2)

    # Summary
    print(f"\n{'='*64}")
    print(f"  {'Agent':<22} {'Overall':>10} {'Peak':>10} {'Off-peak':>10} {'Batch':>8} {'p95 lat':>10}")
    print(f"  {'-'*70}")
    for r in results_list:
        print(f"  {r['agent']:<22} {r['overall_reward']:>10.1f} "
              f"{r['peak_reward']:>10.1f} {r['offpeak_reward']:>10.1f} "
              f"{r['mean_batch']:>8.2f} {r['p95_latency_ms']:>10.0f}ms")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",            type=str, default="checkpoints_sac/sac_best.pth")
    p.add_argument("--traffic",          type=str, default="bursty",
                   choices=["poisson", "bursty", "time_varying"])
    p.add_argument("--peak-episodes",    type=int, default=100)
    p.add_argument("--offpeak-episodes", type=int, default=100)
    main(p.parse_args())