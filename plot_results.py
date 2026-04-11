"""
plot_results.py
---------------
Standalone plotting script for SAC+LSTM+PER evaluation results.
Run from the project root:
    python plot_results.py

Produces 4 figures in results/:
  1. results/eval_comparison_all.png    — agent comparison across all 3 traffic types
  2. results/training_curves_bursty.png — bursty training diagnostics
  3. results/training_curves_poisson.png
  4. results/training_curves_timevarying.png
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs("results", exist_ok=True)

# ── Hardcoded evaluation results (from the three evaluate_sac.py runs) ────────
# These use the REAL batch sizes from training logs, not the broken post-serve metric

EVAL_RESULTS = {
    "bursty": {
        "traffic_label": "Bursty (5×/0.2×)",
        "agents": {
            "SAC+LSTM+PER": {"overall": 24887, "peak": 47810, "offpeak":  1963,
                             "real_batch_peak": 50.1, "real_batch_offpeak": 2.3,
                             "p95_lat": 57, "sla_viol": 0.0},
            "Greedy":       {"overall": 17385, "peak": 38976, "offpeak": -4207,
                             "real_batch_peak":  1.0, "real_batch_offpeak": 0.0,
                             "p95_lat":  5, "sla_viol": 0.0},
            "Random":       {"overall": 24677, "peak": 48023, "offpeak":  1331,
                             "real_batch_peak": 12.5, "real_batch_offpeak": 0.5,
                             "p95_lat": 42, "sla_viol": 0.0},
            "Threshold":    {"overall": 23376, "peak": 46033, "offpeak":   718,
                             "real_batch_peak": 10.0, "real_batch_offpeak": 0.3,
                             "p95_lat": 47, "sla_viol": 0.0},
        }
    },
    "poisson": {
        "traffic_label": "Poisson (1.2×/0.8×)",
        "agents": {
            "SAC+LSTM+PER": {"overall": 10136, "peak": 12069, "offpeak":  8203,
                             "real_batch_peak": 12.0, "real_batch_offpeak": 8.0,
                             "p95_lat": 56, "sla_viol": 0.0},
            "Greedy":       {"overall":  2989, "peak":  4786, "offpeak":  1192,
                             "real_batch_peak":  1.0, "real_batch_offpeak": 1.0,
                             "p95_lat":  5, "sla_viol": 0.0},
            "Random":       {"overall":  9927, "peak": 11934, "offpeak":  7919,
                             "real_batch_peak":  6.0, "real_batch_offpeak": 4.0,
                             "p95_lat": 41, "sla_viol": 0.0},
            "Threshold":    {"overall":  8815, "peak": 10726, "offpeak":  6905,
                             "real_batch_peak":  5.0, "real_batch_offpeak": 3.5,
                             "p95_lat": 47, "sla_viol": 0.0},
        }
    },
    "time_varying": {
        "traffic_label": "Time-Varying (2.5×/0.5×)",
        "agents": {
            "SAC+LSTM+PER": {"overall": 14821, "peak": 24408, "offpeak":  5234,
                             "real_batch_peak": 25.1, "real_batch_offpeak": 5.1,
                             "p95_lat": 57, "sla_viol": 0.0},
            "Greedy":       {"overall":  7479, "peak": 16466, "offpeak": -1508,
                             "real_batch_peak":  1.0, "real_batch_offpeak": 0.0,
                             "p95_lat":  5, "sla_viol": 0.0},
            "Random":       {"overall": 14623, "peak": 24476, "offpeak":  4771,
                             "real_batch_peak": 12.5, "real_batch_offpeak": 2.5,
                             "p95_lat": 42, "sla_viol": 0.0},
            "Threshold":    {"overall": 13433, "peak": 22914, "offpeak":  3951,
                             "real_batch_peak": 10.0, "real_batch_offpeak": 2.0,
                             "p95_lat": 47, "sla_viol": 0.0},
        }
    },
}

AGENT_COLORS = {
    "SAC+LSTM+PER": "#1f77b4",
    "Greedy":       "#ff7f0e",
    "Random":       "#2ca02c",
    "Threshold":    "#d62728",
}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Full evaluation comparison across all traffic patterns
# ═══════════════════════════════════════════════════════════════════════════════

def plot_full_comparison():
    traffic_keys = ["bursty", "poisson", "time_varying"]
    n_traffic = len(traffic_keys)
    agents = list(EVAL_RESULTS["bursty"]["agents"].keys())

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "SAC+LSTM+PER vs Baselines — Complete Evaluation\n"
        "Stratified: 100 peak + 100 off-peak episodes per traffic pattern",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

    colors = [AGENT_COLORS[a] for a in agents]
    x = np.arange(len(agents))
    w = 0.35

    # Row 0: Overall reward
    for col, tk in enumerate(traffic_keys):
        ax = fig.add_subplot(gs[0, col])
        data = EVAL_RESULTS[tk]["agents"]
        vals = [data[a]["overall"] for a in agents]
        bars = ax.bar(agents, vals, color=colors, edgecolor="black", lw=0.8)
        ax.set_title(f"Overall Reward\n{EVAL_RESULTS[tk]['traffic_label']}", fontweight="bold", fontsize=10)
        ax.set_ylabel("Mean Episode Reward")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        for bar, v in zip(bars, vals):
            ypos = bar.get_height() if v >= 0 else 0
            ax.text(bar.get_x()+bar.get_width()/2, ypos + abs(max(vals))*0.01,
                    f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.axhline(y=0, color="black", lw=0.5, alpha=0.3)

    # Row 1: Peak vs Off-peak reward split
    for col, tk in enumerate(traffic_keys):
        ax = fig.add_subplot(gs[1, col])
        data = EVAL_RESULTS[tk]["agents"]
        peak_vals   = [data[a]["peak"]    for a in agents]
        offpeak_vals = [data[a]["offpeak"] for a in agents]
        ax.bar(x - w/2, peak_vals,    w, label="Peak",     color=[c+"CC" for c in colors], edgecolor="black", lw=0.8)
        ax.bar(x + w/2, offpeak_vals, w, label="Off-peak", color=colors, edgecolor="black", lw=0.8, alpha=0.85)
        ax.set_title(f"Peak vs Off-peak Reward\n{EVAL_RESULTS[tk]['traffic_label']}", fontweight="bold", fontsize=10)
        ax.set_ylabel("Mean Episode Reward")
        ax.set_xticks(x); ax.set_xticklabels(agents, rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.axhline(y=0, color="black", lw=0.5, alpha=0.3)

    # Row 2: Real batch size (peak and offpeak)
    for col, tk in enumerate(traffic_keys):
        ax = fig.add_subplot(gs[2, col])
        data = EVAL_RESULTS[tk]["agents"]
        peak_b   = [data[a]["real_batch_peak"]    for a in agents]
        offpeak_b = [data[a]["real_batch_offpeak"] for a in agents]
        ax.bar(x - w/2, peak_b,    w, label="Peak",     color=[c+"CC" for c in colors], edgecolor="black", lw=0.8)
        ax.bar(x + w/2, offpeak_b, w, label="Off-peak", color=colors, edgecolor="black", lw=0.8, alpha=0.85)
        ax.set_title(f"Avg Batch Size (req/dispatch)\n{EVAL_RESULTS[tk]['traffic_label']}", fontweight="bold", fontsize=10)
        ax.set_ylabel("Requests per batch")
        ax.set_xticks(x); ax.set_xticklabels(agents, rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        for i, (pb, ob) in enumerate(zip(peak_b, offpeak_b)):
            ax.text(i - w/2, pb + 0.3, f"{pb:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Row 3: p95 latency
    for col, tk in enumerate(traffic_keys):
        ax = fig.add_subplot(gs[3, col])
        data = EVAL_RESULTS[tk]["agents"]
        lats = [data[a]["p95_lat"] for a in agents]
        bars = ax.bar(agents, lats, color=colors, edgecolor="black", lw=0.8)
        ax.axhline(y=500, color="red", linestyle="--", lw=1.5, alpha=0.8, label="SLA limit (500ms)")
        ax.set_title(f"p95 Latency (ms)\n{EVAL_RESULTS[tk]['traffic_label']}", fontweight="bold", fontsize=10)
        ax.set_ylabel("Milliseconds")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 120)
        for bar, v in zip(bars, lats):
            ax.text(bar.get_x()+bar.get_width()/2, v + 1, f"{v}ms",
                    ha="center", va="bottom", fontsize=8)

    plt.savefig("results/eval_comparison_all.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → results/eval_comparison_all.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — SAC advantage bar chart (clean summary for report)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sac_advantage():
    traffic_labels = ["Bursty", "Poisson", "Time-varying"]
    traffic_keys   = ["bursty", "poisson", "time_varying"]
    baselines = ["Greedy", "Random", "Threshold"]
    baseline_colors = {"Greedy": "#ff7f0e", "Random": "#2ca02c", "Threshold": "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "SAC+LSTM+PER Percentage Advantage over Baselines\n(Overall mean reward, 200 stratified evaluation episodes)",
        fontsize=13, fontweight="bold"
    )

    for ax, tk, tl in zip(axes, traffic_keys, traffic_labels):
        data = EVAL_RESULTS[tk]["agents"]
        sac_val = data["SAC+LSTM+PER"]["overall"]
        pcts = []
        for b in baselines:
            base_val = data[b]["overall"]
            if base_val <= 0:
                pct = ((sac_val - base_val) / max(abs(base_val), 1)) * 100
            else:
                pct = ((sac_val - base_val) / base_val) * 100
            pcts.append(pct)

        bars = ax.bar(baselines, pcts,
                      color=[baseline_colors[b] for b in baselines],
                      edgecolor="black", lw=0.8)
        ax.axhline(y=0, color="black", lw=1)
        ax.set_title(f"{tl} Traffic", fontweight="bold", fontsize=11)
        ax.set_ylabel("SAC advantage (%)")
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, pcts):
            ypos = v + 2 if v >= 0 else v - 8
            ax.text(bar.get_x()+bar.get_width()/2, ypos, f"+{v:.0f}%" if v >= 0 else f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/sac_advantage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → results/sac_advantage.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Training curves (one per traffic pattern)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(log_path, traffic_name, save_path):
    if not os.path.exists(log_path):
        print(f"  [WARN] Log not found: {log_path}")
        return

    with open(log_path) as f:
        logs = json.load(f)

    rewards  = logs.get("episode_rewards", [])
    phases   = logs.get("episode_phases", ["random"]*len(rewards))
    alphas   = logs.get("alpha_values", [])
    entropy  = logs.get("entropy_values", [])
    batches  = logs.get("avg_batch_sizes", [])
    adists   = logs.get("action_distributions", [])
    c_losses = logs.get("critic_losses", [])
    a_losses = logs.get("actor_losses", [])

    if not rewards:
        print(f"  [WARN] No rewards in {log_path}")
        return

    window = 20
    def ma(arr):
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window)/window, mode="valid")

    phase_colors = {"random": "#aaaaaa", "peak": "#d62728", "offpeak": "#1f77b4"}

    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(
        f"SAC+LSTM+PER Training Diagnostics — {traffic_name.replace('_',' ').title()} Traffic\n"
        f"500 episodes | State dim=8 | Seq len=30 | Buffer=200k | Phase cycle: random→random→peak→offpeak",
        fontsize=12, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    eps = list(range(1, len(rewards)+1))

    # Panel 1: Episode reward coloured by phase
    ax = fig.add_subplot(gs[0, 0])
    for ph, col in phase_colors.items():
        idxs = [i for i, p in enumerate(phases) if p == ph]
        if idxs:
            ax.scatter([eps[i] for i in idxs], [rewards[i] for i in idxs],
                       s=6, alpha=0.5, color=col, label=ph, zorder=2)
    if len(rewards) >= window:
        ax.plot(range(window, len(rewards)+1), ma(rewards),
                color="black", lw=2, label=f"MA({window})", zorder=3)
    ax.set_title("Episode Reward by Phase", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.legend(fontsize=7)

    # Panel 2: Alpha curriculum
    if alphas:
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(alphas, color="#ff7f0e", lw=2)
        ax.set_title("Entropy Temperature α\n(1.0 → 0.1 curriculum)", fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel("α value")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.1, color="gray", ls="--", alpha=0.5, label="Final α=0.1")
        ax.legend(fontsize=8)

    # Panel 3: Avg batch size
    if batches:
        ax = fig.add_subplot(gs[0, 2])
        # Separate peak and offpeak batch sizes
        peak_b = [b for b, p in zip(batches, phases) if p == "peak" and b > 5]
        offb   = [b for b, p in zip(batches, phases) if p == "offpeak" and b > 0]
        rand_b = [b for b, p in zip(batches, phases) if p == "random"]
        if peak_b:
            ax.scatter([i for i, (b,p) in enumerate(zip(batches,phases)) if p=="peak" and b>5],
                       peak_b, s=6, alpha=0.7, color="#d62728", label="peak")
        if offb:
            ax.scatter([i for i, (b,p) in enumerate(zip(batches,phases)) if p=="offpeak" and b>0],
                       offb, s=6, alpha=0.7, color="#1f77b4", label="offpeak")
        if rand_b:
            ax.scatter([i for i, (b,p) in enumerate(zip(batches,phases)) if p=="random"],
                       rand_b, s=4, alpha=0.4, color="#aaaaaa", label="random")
        ax.set_title("Avg Batch Size\n(pre-serve queue)", fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel("Requests/dispatch")
        ax.legend(fontsize=7)

    # Panel 4: Critic loss
    if c_losses:
        ax = fig.add_subplot(gs[0, 3])
        ax.plot(c_losses, color="#9467bd", lw=1.2, alpha=0.8)
        if len(c_losses) >= window:
            ax.plot(range(window, len(c_losses)+1), ma(c_losses),
                    color="#9467bd", lw=2, label=f"MA({window})")
        ax.set_title("Critic Loss\n(convergence indicator)", fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel("Huber Loss")
        ax.legend(fontsize=8)

    # Panel 5: Action distribution over training
    if adists:
        ax = fig.add_subplot(gs[1, :2])
        adists_arr = np.array(adists)
        labels = ["serve_now (W0)", "wait_20ms (W20)", "wait_50ms (W50)", "wait_100ms (W100)"]
        cols   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, (lbl, col) in enumerate(zip(labels, cols)):
            ax.plot(adists_arr[:, i], label=lbl, color=col, lw=1.5, alpha=0.85)
        ax.set_title("Action Distribution over Training", fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel("Fraction of actions")
        ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.1)

    # Panel 6: Actor loss
    if a_losses:
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(a_losses, color="#8c564b", lw=1, alpha=0.7)
        if len(a_losses) >= window:
            ax.plot(range(window, len(a_losses)+1), ma(a_losses),
                    color="#8c564b", lw=2, label=f"MA({window})")
        ax.set_title("Actor Loss\n(policy strength)", fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8)

    # Panel 7: Entropy collapse
    if entropy:
        ax = fig.add_subplot(gs[1, 3])
        ax.semilogy(entropy, color="#17becf", lw=1.5)
        ax.set_title("Policy Entropy (log scale)\n(→ 0 = deterministic policy)", fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel("H(π) [log scale]")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Cross-traffic summary radar / grouped bar
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cross_traffic_summary():
    traffic_keys   = ["bursty", "poisson", "time_varying"]
    traffic_labels = ["Bursty\n(5×/0.2×)", "Poisson\n(1.2×/0.8×)", "Time-varying\n(2.5×/0.5×)"]
    agents = ["SAC+LSTM+PER", "Greedy", "Random", "Threshold"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Cross-Traffic Performance Summary\nSAC+LSTM+PER vs Baselines",
        fontsize=13, fontweight="bold"
    )

    metrics = [
        ("overall", "Overall Reward"),
        ("peak",    "Peak Reward"),
        ("offpeak", "Off-peak Reward"),
    ]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        x = np.arange(len(traffic_labels))
        w = 0.18
        offsets = np.linspace(-0.27, 0.27, len(agents))

        for i, agent in enumerate(agents):
            vals = [EVAL_RESULTS[tk]["agents"][agent][metric_key] for tk in traffic_keys]
            bars = ax.bar(x + offsets[i], vals, w,
                          label=agent, color=AGENT_COLORS[agent],
                          edgecolor="black", lw=0.6)

        ax.set_title(metric_label, fontweight="bold", fontsize=11)
        ax.set_ylabel("Mean Episode Reward")
        ax.set_xticks(x)
        ax.set_xticklabels(traffic_labels, fontsize=9)
        ax.axhline(y=0, color="black", lw=0.8, alpha=0.4)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/cross_traffic_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → results/cross_traffic_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating plots...\n")

    print("1/5  Full evaluation comparison...")
    plot_full_comparison()

    print("2/5  SAC advantage chart...")
    plot_sac_advantage()

    print("3/5  Cross-traffic summary...")
    plot_cross_traffic_summary()

    print("4/5  Training curves...")
    for traffic, logfile in [
        ("bursty",       "logs_sac/bursty/training_logs.json"),
        ("poisson",      "logs_sac/poisson/training_logs.json"),
        ("time_varying", "logs_sac/time_varying/training_logs.json"),
    ]:
        label = traffic.replace("_", " ").title()
        save  = f"results/training_curves_{traffic}.png"
        plot_training_curves(logfile, traffic, save)

    print("\nAll plots saved to results/")
    print("Files:")
    for f in sorted(os.listdir("results")):
        if f.endswith(".png"):
            size_kb = os.path.getsize(f"results/{f}") // 1024
            print(f"  results/{f}  ({size_kb} KB)")