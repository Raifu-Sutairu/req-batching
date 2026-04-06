"""
results/comparison_graphs.py

Comprehensive comparison graph suite for the Dynamic Request Batching RL project.

Loads all trained models (PPO, DQN, RPPO) and evaluates them against three
baselines (Greedy, Cloudflare, Random) over N_EPISODES each, then generates
8 polished comparison charts plus one combined dashboard PNG.

Charts produced (saved to results/graphs/):
    01_reward_bars.png          — Mean ± std episode reward (bar chart)
    02_reward_boxplot.png       — Episode reward distributions (box plot)
    03_batch_distribution.png   — Batch size KDE + histogram per agent
    04_latency_cdf.png          — Cumulative latency distribution curves
    05_latency_boxplot.png      — Latency statistical spread (p50/p95/p99)
    06_sla_violation_rate.png   — % of serve actions that violated SLA
    07_throughput.png           — Total requests served per episode
    08_radar_chart.png          — Multi-metric spider/radar chart
    00_dashboard.png            — All 8 charts in a 2×4 grid (high-res)

Usage:
    python results/comparison_graphs.py
    python results/comparison_graphs.py --episodes 20
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from math import pi

from stable_baselines3 import PPO, DQN
from sb3_contrib import RecurrentPPO

from env.batching_env import BatchingEnv
from baselines.cloudflare_formula import CloudflareBaseline
from baselines.random_agent import RandomAgent
from baselines.greedy_agent import GreedyAgent
from config import CONFIG

# ── Output directory ──────────────────────────────────────────────────────────
GRAPHS_DIR = os.path.join(ROOT, "results", "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ── Evaluation config ─────────────────────────────────────────────────────────
N_EPISODES   = 30
SEED_OFFSET  = 100
SLA_MS       = CONFIG["max_latency_ms"]

# ── Dark-theme design tokens ──────────────────────────────────────────────────
BG_FIG    = "#0a0e1a"    # figure background
BG_AX     = "#141c30"    # axes background
COL_GRID  = "#2a3450"    # grid lines
COL_SPINE = "#2a3450"    # axis spines
COL_TEXT  = "#e8eaf6"    # all text / labels
COL_MUTED = "#7986cb"    # secondary/muted text

AGENT_COLORS = {
    "PPO":        "#00e5ff",   # cyan
    "DQN":        "#76ff03",   # lime green
    "RPPO":       "#e040fb",   # magenta
    "Greedy":     "#7c4dff",   # deep purple
    "Cloudflare": "#ff9100",   # amber
    "Random":     "#f44336",   # red
}
AGENT_ORDER = ["PPO", "DQN", "RPPO", "Greedy", "Cloudflare", "Random"]

# Readable aliases for plot labels
AGENT_LABELS = {
    "PPO":        "PPO",
    "DQN":        "DQN",
    "RPPO":       "RecurrentPPO",
    "Greedy":     "Greedy",
    "Cloudflare": "Cloudflare",
    "Random":     "Random",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "text.color":       COL_TEXT,
    "axes.labelcolor":  COL_TEXT,
    "xtick.color":      COL_TEXT,
    "ytick.color":      COL_TEXT,
    "figure.facecolor": BG_FIG,
    "axes.facecolor":   BG_AX,
})


# ─────────────────────────────────────────────────────────────────────────────
# Agent wrappers (identical to evaluate.py for consistency)
# ─────────────────────────────────────────────────────────────────────────────

class _PPOWrapper:
    def __init__(self, m): self.model = m
    def predict(self, obs): return int(self.model.predict(obs, deterministic=True)[0])

class _DQNWrapper:
    def __init__(self, m): self.model = m
    def predict(self, obs): return int(self.model.predict(obs, deterministic=True)[0])

class _RPPOWrapper:
    def __init__(self, m):
        self.model = m
        self.lstm_states = None
        self.episode_started = True

    def reset(self):
        self.lstm_states = None
        self.episode_started = True

    def predict(self, obs):
        ep_start = np.array([self.episode_started], dtype=bool)
        action, self.lstm_states = self.model.predict(
            obs, state=self.lstm_states,
            episode_start=ep_start, deterministic=True,
        )
        self.episode_started = False
        return int(action)


# ─────────────────────────────────────────────────────────────────────────────
# Data collection — extended beyond evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

def collect_data(agent, env, n_episodes: int, seed_offset: int) -> dict:
    """Run agent for n_episodes and collect rich per-episode statistics.

    Returns
    -------
    dict with keys:
        rewards        : list[float]   — total episode reward
        batch_sizes    : list[int]     — batch size at every Serve action
        latencies      : list[float]   — per-serve mean latency (ms)
        throughputs    : list[int]     — total requests served per episode
        sla_violations : list[int]     — Serve actions that occurred after SLA
                                         deadline was already breached
        n_serve_actions: list[int]     — total Serve actions per episode
    """
    rewards         = []
    batch_sizes     = []
    latencies       = []
    throughputs     = []
    sla_violations  = []
    n_serve_actions = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        total_reward  = 0.0
        terminated = truncated = False
        prev_served   = 0
        ep_sla_viols  = 0
        ep_serve_acts = 0

        if hasattr(agent, "reset"):
            agent.reset()

        while not (terminated or truncated):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if action == 1:   # Serve
                ep_serve_acts += 1
                served_now = info["total_served"] - prev_served
                if served_now > 0:
                    batch_sizes.append(served_now)
                    lat = info["mean_latency_ms"]
                    if lat > 0:
                        latencies.append(lat)
                    # SLA violation: oldest request already past deadline when served
                    oldest_wait = float(obs[1])   # obs[1] = oldest_wait_ms (post-step)
                    if oldest_wait > SLA_MS:
                        ep_sla_viols += 1
                prev_served = info["total_served"]

        rewards.append(total_reward)
        throughputs.append(info["total_served"])
        sla_violations.append(ep_sla_viols)
        n_serve_actions.append(ep_serve_acts)

    return {
        "rewards":         rewards,
        "batch_sizes":     batch_sizes,
        "latencies":       latencies,
        "throughputs":     throughputs,
        "sla_violations":  sla_violations,
        "n_serve_actions": n_serve_actions,
    }


def load_all_agents(env) -> dict:
    """Load all models and wrap them. Returns dict keyed by agent name."""
    agents = {}

    # ── PPO ──────────────────────────────────────────────────────────────────
    for p in [
        os.path.join(ROOT, "models", "best", "best_model"),
        os.path.join(ROOT, "models", "ppo_batching_final"),
    ]:
        if os.path.exists(p + ".zip") or os.path.exists(p):
            agents["PPO"] = _PPOWrapper(PPO.load(p))
            print(f"  [✓] PPO  loaded from {p}")
            break
    if "PPO" not in agents:
        print("  [✗] PPO model not found — using Random fallback")
        agents["PPO"] = RandomAgent(seed=0)

    # ── DQN ──────────────────────────────────────────────────────────────────
    for p in [
        os.path.join(ROOT, "models", "dqn_best", "best_model"),
        os.path.join(ROOT, "models", "dqn_batching_final"),
    ]:
        if os.path.exists(p + ".zip") or os.path.exists(p):
            agents["DQN"] = _DQNWrapper(DQN.load(p))
            print(f"  [✓] DQN  loaded from {p}")
            break
    if "DQN" not in agents:
        print("  [✗] DQN model not found — using Random fallback")
        agents["DQN"] = RandomAgent(seed=1)

    # ── RPPO ─────────────────────────────────────────────────────────────────
    for p in [
        os.path.join(ROOT, "models", "rppo_best", "best_model"),
        os.path.join(ROOT, "models", "rppo_batching_final"),
    ]:
        if os.path.exists(p + ".zip") or os.path.exists(p):
            agents["RPPO"] = _RPPOWrapper(RecurrentPPO.load(p))
            print(f"  [✓] RPPO loaded from {p}")
            break
    if "RPPO" not in agents:
        print("  [✗] RPPO model not found — using Random fallback")
        agents["RPPO"] = RandomAgent(seed=2)

    # ── Baselines ─────────────────────────────────────────────────────────────
    agents["Greedy"]     = GreedyAgent()
    agents["Cloudflare"] = CloudflareBaseline(max_latency_ms=SLA_MS, seed=42)
    agents["Random"]     = RandomAgent(seed=42)
    print("  [✓] Greedy / Cloudflare / Random baselines ready")

    return agents


def run_evaluation(n_episodes: int = N_EPISODES) -> dict:
    """Evaluate all agents and return data dict keyed by agent name."""
    env = BatchingEnv()
    agents = load_all_agents(env)
    all_data = {}

    print()
    for name in AGENT_ORDER:
        print(f"  Evaluating {AGENT_LABELS[name]:<16s} × {n_episodes} episodes ...", end=" ", flush=True)
        data = collect_data(agents[name], env, n_episodes, SEED_OFFSET)
        all_data[name] = data
        mean_r = np.mean(data["rewards"])
        print(f"mean reward = {mean_r:+.1f}")

    env.close()
    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# Shared style helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "",
           grid_axis: str = "both"):
    ax.set_facecolor(BG_AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(COL_SPINE)
    ax.tick_params(colors=COL_TEXT, labelsize=9)
    ax.xaxis.label.set_color(COL_TEXT)
    ax.yaxis.label.set_color(COL_TEXT)
    ax.grid(True, axis=grid_axis, color=COL_GRID, linewidth=0.5, alpha=0.7, zorder=0)
    if title:
        ax.set_title(title, color=COL_TEXT, fontsize=11, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=6)


def _legend(ax, **kwargs):
    leg = ax.legend(
        facecolor=BG_AX, edgecolor=COL_SPINE,
        labelcolor=COL_TEXT, fontsize=8.5,
        **kwargs,
    )
    return leg


def _save(fig, name: str) -> str:
    path = os.path.join(GRAPHS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)
    return path


def _agent_patches():
    return [
        mpatches.Patch(facecolor=AGENT_COLORS[n], label=AGENT_LABELS[n])
        for n in AGENT_ORDER
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Mean Reward Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def chart_reward_bars(data: dict, ax=None) -> str | None:
    """Bar chart: mean ± std episode reward per agent."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)

    means  = [np.mean(data[n]["rewards"]) for n in AGENT_ORDER]
    stds   = [np.std(data[n]["rewards"])  for n in AGENT_ORDER]
    colors = [AGENT_COLORS[n]             for n in AGENT_ORDER]
    x      = np.arange(len(AGENT_ORDER))

    bars = ax.bar(
        x, means, yerr=stds,
        color=colors, width=0.6, alpha=0.88, zorder=3,
        capsize=6, error_kw=dict(color=COL_TEXT, linewidth=1.5, zorder=4),
    )

    for bar, mean in zip(bars, means):
        va  = "bottom" if mean >= 0 else "top"
        off = 6 if mean >= 0 else -6
        ax.annotate(
            f"{mean:+.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, mean),
            xytext=(0, off), textcoords="offset points",
            ha="center", va=va, color=COL_TEXT, fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[n] for n in AGENT_ORDER], color=COL_TEXT, fontsize=9)
    ax.axhline(0, color=COL_TEXT, linewidth=0.8, linestyle="--", alpha=0.4)
    _style(ax, "Mean Episode Reward  ±  Std Dev",
           ylabel="Episode Reward", grid_axis="y")

    if standalone:
        ax.legend(handles=_agent_patches(), facecolor=BG_AX,
                  edgecolor=COL_SPINE, labelcolor=COL_TEXT, fontsize=8)
        return _save(fig, "01_reward_bars.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Reward Box Plot
# ─────────────────────────────────────────────────────────────────────────────

def chart_reward_boxplot(data: dict, ax=None) -> str | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)

    rewards_list = [data[n]["rewards"] for n in AGENT_ORDER]
    colors       = [AGENT_COLORS[n]   for n in AGENT_ORDER]

    bp = ax.boxplot(
        rewards_list,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color=COL_TEXT, linewidth=2),
        whiskerprops=dict(color=COL_TEXT, linewidth=1.2),
        capprops=dict(color=COL_TEXT, linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.6,
                        markeredgecolor="none"),
        zorder=3,
    )
    for patch, col, flier in zip(bp["boxes"], colors, bp["fliers"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
        flier.set_markerfacecolor(col)

    ax.set_xticks(range(1, len(AGENT_ORDER) + 1))
    ax.set_xticklabels([AGENT_LABELS[n] for n in AGENT_ORDER], color=COL_TEXT, fontsize=9)
    ax.axhline(0, color=COL_TEXT, linewidth=0.8, linestyle="--", alpha=0.4)
    _style(ax, "Episode Reward Distribution", ylabel="Episode Reward", grid_axis="y")

    if standalone:
        ax.legend(handles=_agent_patches(), facecolor=BG_AX,
                  edgecolor=COL_SPINE, labelcolor=COL_TEXT, fontsize=8)
        return _save(fig, "02_reward_boxplot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Batch Size Distribution
# ─────────────────────────────────────────────────────────────────────────────

def chart_batch_distribution(data: dict, ax=None) -> str | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)

    max_bs = CONFIG["max_batch_size"]
    plotted = False

    for name in AGENT_ORDER:
        bs = data[name]["batch_sizes"]
        if not bs:
            continue
        plotted = True
        col = AGENT_COLORS[name]
        ax.hist(bs, bins=40, range=(0, max_bs),
                alpha=0.25, color=col, density=True, edgecolor="none", zorder=2)
        try:
            if len(set(bs)) > 3:
                kde = gaussian_kde(bs, bw_method=0.3)
                xs  = np.linspace(0, max_bs, 400)
                ax.plot(xs, kde(xs), color=col, linewidth=2.2,
                        label=AGENT_LABELS[name], zorder=3)
        except Exception:
            # fallback — just show a vertical line at mean
            ax.axvline(np.mean(bs), color=col, linewidth=2,
                       label=f"{AGENT_LABELS[name]} (mean={np.mean(bs):.0f})")

    if not plotted:
        ax.text(0.5, 0.5, "No serve actions recorded",
                transform=ax.transAxes, ha="center", va="center",
                color=COL_TEXT, fontsize=10)

    _style(ax, "Batch Size Distribution  (KDE)",
           xlabel="Batch Size (requests)", ylabel="Density", grid_axis="y")
    _legend(ax, loc="upper right")

    if standalone:
        return _save(fig, "03_batch_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — Latency CDF
# ─────────────────────────────────────────────────────────────────────────────

def chart_latency_cdf(data: dict, ax=None) -> str | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)

    plotted = False
    for name in AGENT_ORDER:
        lat = data[name]["latencies"]
        if not lat:
            continue
        plotted = True
        sl = np.sort(lat)
        cdf = np.arange(1, len(sl) + 1) / len(sl)
        ax.plot(sl, cdf, color=AGENT_COLORS[name], linewidth=2.2,
                label=AGENT_LABELS[name], alpha=0.9, zorder=3)

    ax.axvline(SLA_MS, color="#ffffff", linewidth=1.0, linestyle=":",
               alpha=0.7, label=f"SLA ({SLA_MS} ms)", zorder=4)

    if not plotted:
        ax.text(0.5, 0.5, "No latency data", transform=ax.transAxes,
                ha="center", va="center", color=COL_TEXT, fontsize=10)

    ax.set_ylim(0, 1.06)
    _style(ax, "Serve Latency CDF  (lower-left = better)",
           xlabel="Mean Serve Latency (ms)", ylabel="Cumulative Probability")
    _legend(ax, loc="lower right")

    if standalone:
        return _save(fig, "04_latency_cdf.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 — Latency Box Plot  (p50 / p95 / p99 visible)
# ─────────────────────────────────────────────────────────────────────────────

def chart_latency_boxplot(data: dict, ax=None) -> str | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)

    lat_list = [data[n]["latencies"] or [0] for n in AGENT_ORDER]
    colors   = [AGENT_COLORS[n]             for n in AGENT_ORDER]

    bp = ax.boxplot(
        lat_list,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color=COL_TEXT, linewidth=2),
        whiskerprops=dict(color=COL_TEXT, linewidth=1.2),
        capprops=dict(color=COL_TEXT, linewidth=1.5),
        flierprops=dict(marker="o", markersize=3.5, alpha=0.5,
                        markeredgecolor="none"),
        zorder=3,
    )
    for patch, col, flier in zip(bp["boxes"], colors, bp["fliers"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.72)
        flier.set_markerfacecolor(col)

    ax.axhline(SLA_MS, color="#ff5252", linewidth=1.2, linestyle=":",
               alpha=0.8, label=f"SLA ({SLA_MS} ms)", zorder=4)

    ax.set_xticks(range(1, len(AGENT_ORDER) + 1))
    ax.set_xticklabels([AGENT_LABELS[n] for n in AGENT_ORDER], color=COL_TEXT, fontsize=9)
    _style(ax, "Serve Latency Distribution", ylabel="Latency (ms)", grid_axis="y")
    _legend(ax, loc="upper right")

    if standalone:
        ax.legend(
            handles=[mpatches.Patch(facecolor="#ff5252", label=f"SLA {SLA_MS} ms")]
                    + _agent_patches(),
            facecolor=BG_AX, edgecolor=COL_SPINE, labelcolor=COL_TEXT, fontsize=8,
        )
        return _save(fig, "05_latency_boxplot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 6 — SLA Violation Rate
# ─────────────────────────────────────────────────────────────────────────────

def chart_sla_violation_rate(data: dict, ax=None) -> str | None:
    """% of Serve actions that occurred after an SLA violation per agent."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)

    rates  = []
    for name in AGENT_ORDER:
        viols  = sum(data[name]["sla_violations"])
        serves = sum(data[name]["n_serve_actions"])
        rate   = 100.0 * viols / serves if serves > 0 else 0.0
        rates.append(rate)

    colors = [AGENT_COLORS[n] for n in AGENT_ORDER]
    x      = np.arange(len(AGENT_ORDER))

    bars = ax.bar(x, rates, color=colors, width=0.6, alpha=0.88, zorder=3)

    for bar, rate in zip(bars, rates):
        ax.annotate(
            f"{rate:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, rate),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", color=COL_TEXT, fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[n] for n in AGENT_ORDER], color=COL_TEXT, fontsize=9)
    ax.set_ylim(0, max(rates) * 1.3 + 1)
    _style(ax, "SLA Violation Rate per Agent",
           ylabel="% of Serve Actions with SLA Breach", grid_axis="y")

    # Lower = better annotation
    ax.text(0.98, 0.96, "← lower is better", transform=ax.transAxes,
            ha="right", va="top", color=COL_MUTED, fontsize=8.5, fontstyle="italic")

    if standalone:
        ax.legend(handles=_agent_patches(), facecolor=BG_AX,
                  edgecolor=COL_SPINE, labelcolor=COL_TEXT, fontsize=8)
        return _save(fig, "06_sla_violation_rate.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 7 — Throughput (requests served per episode)
# ─────────────────────────────────────────────────────────────────────────────

def chart_throughput(data: dict, ax=None) -> str | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)

    means  = [np.mean(data[n]["throughputs"]) for n in AGENT_ORDER]
    stds   = [np.std(data[n]["throughputs"])  for n in AGENT_ORDER]
    colors = [AGENT_COLORS[n]                 for n in AGENT_ORDER]
    x      = np.arange(len(AGENT_ORDER))

    bars = ax.bar(
        x, means, yerr=stds, color=colors, width=0.6, alpha=0.88, zorder=3,
        capsize=6, error_kw=dict(color=COL_TEXT, linewidth=1.5, zorder=4),
    )

    for bar, mean in zip(bars, means):
        ax.annotate(
            f"{mean:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, mean),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", color=COL_TEXT, fontsize=8.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[n] for n in AGENT_ORDER], color=COL_TEXT, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _style(ax, "Throughput — Total Requests Served per Episode",
           ylabel="Requests Served", grid_axis="y")

    ax.text(0.98, 0.96, "↑ higher is better", transform=ax.transAxes,
            ha="right", va="top", color=COL_MUTED, fontsize=8.5, fontstyle="italic")

    if standalone:
        ax.legend(handles=_agent_patches(), facecolor=BG_AX,
                  edgecolor=COL_SPINE, labelcolor=COL_TEXT, fontsize=8)
        return _save(fig, "07_throughput.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 8 — Radar / Spider Chart
# ─────────────────────────────────────────────────────────────────────────────

def chart_radar(data: dict, ax=None) -> str | None:
    """Multi-metric radar chart — normalised so larger = better on all axes."""
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(9, 8), facecolor=BG_FIG)
        ax  = fig.add_subplot(111, polar=True)
        fig.subplots_adjust(top=0.88, bottom=0.08)

    metrics = [
        "Reward",
        "Throughput",
        "Batch Size",
        "Low Latency",      # inverted: higher score = lower latency
        "SLA Compliance",   # inverted: higher score = fewer violations
    ]
    N = len(metrics)

    # ── Normalise each metric to [0, 1] with higher = better ────────────────
    raw = {}
    for name in AGENT_ORDER:
        d = data[name]
        mean_reward   = np.mean(d["rewards"])
        mean_tp       = np.mean(d["throughputs"])
        mean_bs       = np.mean(d["batch_sizes"]) if d["batch_sizes"] else 0
        mean_lat      = np.mean(d["latencies"])   if d["latencies"]   else SLA_MS
        viols = sum(d["sla_violations"])
        serves = max(sum(d["n_serve_actions"]), 1)
        sla_viol_rate = viols / serves
        raw[name] = [mean_reward, mean_tp, mean_bs, mean_lat, sla_viol_rate]

    # Per-metric min/max for normalisation
    all_vals = np.array([raw[n] for n in AGENT_ORDER])          # shape (6, 5)
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)

    def _norm(val, mn, mx, invert=False):
        if mx == mn:
            return 0.5
        n = (val - mn) / (mx - mn)
        return 1 - n if invert else n

    normalised = {}
    for name in AGENT_ORDER:
        r, tp, bs, lat, sla = raw[name]
        normalised[name] = [
            _norm(r,   mins[0], maxs[0]),           # reward   ↑
            _norm(tp,  mins[1], maxs[1]),            # throughput ↑
            _norm(bs,  mins[2], maxs[2]),            # batch size ↑
            _norm(lat, mins[3], maxs[3], True),      # latency ↓ (inverted)
            _norm(sla, mins[4], maxs[4], True),      # SLA violations ↓ (inverted)
        ]

    # Angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]   # close the loop

    # Style the polar axes
    ax.set_facecolor(BG_AX)
    ax.spines["polar"].set_edgecolor(COL_SPINE)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, color=COL_TEXT, fontsize=9.5, fontweight="bold")
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color=COL_GRID, linewidth=0.8, alpha=0.6)
    ax.xaxis.grid(True, color=COL_GRID, linewidth=0.8, alpha=0.6)
    ax.set_ylim(0, 1)
    # Concentric ring labels
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.text(0, r, f"{r:.2f}", ha="center", va="center",
                color=COL_MUTED, fontsize=7, alpha=0.7)

    # Draw each agent
    for name in AGENT_ORDER:
        vals   = normalised[name] + [normalised[name][0]]  # close loop
        col    = AGENT_COLORS[name]
        ax.plot(angles, vals, color=col, linewidth=2.2, label=AGENT_LABELS[name])
        ax.fill(angles, vals, color=col, alpha=0.10)

    ax.set_title("Multi-Metric Agent Comparison\n(Normalised — larger = better)",
                 color=COL_TEXT, fontsize=11, fontweight="bold", pad=20)
    _legend(ax, loc="upper right", bbox_to_anchor=(1.35, 1.12))

    if standalone:
        return _save(fig, "08_radar_chart.png")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard — 2 × 4 grid of all 8 charts
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(data: dict) -> str:
    """Render all 8 charts into a single high-resolution dashboard PNG."""
    fig = plt.figure(figsize=(28, 16), facecolor=BG_FIG)
    fig.suptitle(
        "Dynamic Request Batching  —  Agent Comparison Dashboard",
        color=COL_TEXT, fontsize=18, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 4,
        figure=fig,
        left=0.05, right=0.97,
        top=0.93, bottom=0.08,
        wspace=0.32, hspace=0.42,
    )

    # Row 0
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
    ]
    # Row 1
    axes += [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3], polar=True),
    ]

    chart_reward_bars(data,         ax=axes[0])
    chart_reward_boxplot(data,      ax=axes[1])
    chart_batch_distribution(data,  ax=axes[2])
    chart_latency_cdf(data,         ax=axes[3])
    chart_latency_boxplot(data,     ax=axes[4])
    chart_sla_violation_rate(data,  ax=axes[5])
    chart_throughput(data,          ax=axes[6])
    chart_radar(data,               ax=axes[7])

    # Shared legend at bottom
    fig.legend(
        handles=_agent_patches(),
        loc="lower center",
        ncol=len(AGENT_ORDER),
        facecolor=BG_AX, edgecolor=COL_SPINE, labelcolor=COL_TEXT,
        fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    path = os.path.join(GRAPHS_DIR, "00_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Standalone chart savers
# ─────────────────────────────────────────────────────────────────────────────

def save_all_individual(data: dict):
    """Save each chart as its own standalone PNG."""
    saved = []
    fns = [
        ("01_reward_bars.png",        chart_reward_bars),
        ("02_reward_boxplot.png",     chart_reward_boxplot),
        ("03_batch_distribution.png", chart_batch_distribution),
        ("04_latency_cdf.png",        chart_latency_cdf),
        ("05_latency_boxplot.png",    chart_latency_boxplot),
        ("06_sla_violation_rate.png", chart_sla_violation_rate),
        ("07_throughput.png",         chart_throughput),
    ]
    for fname, fn in fns:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)
        fn(data, ax=ax)
        path = os.path.join(GRAPHS_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
        plt.close(fig)
        saved.append(path)
        print(f"  ✓  {fname}")

    # Radar needs polar axis — handled separately
    fig = plt.figure(figsize=(9, 8), facecolor=BG_FIG)
    ax  = fig.add_subplot(111, polar=True)
    fig.subplots_adjust(top=0.88, bottom=0.08)
    chart_radar(data, ax=ax)
    path = os.path.join(GRAPHS_DIR, "08_radar_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)
    saved.append(path)
    print("  ✓  08_radar_chart.png")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(data: dict):
    W = 90
    print("\n" + "=" * W)
    hdr = (f"{'Agent':<16}  {'Mean Reward':>12}  {'Std':>8}  "
           f"{'Median Lat':>11}  {'p95 Lat':>9}  "
           f"{'Mean Batch':>11}  {'Throughput':>11}  {'SLA Viol%':>10}")
    print(hdr)
    print("-" * W)
    for name in AGENT_ORDER:
        d     = data[name]
        mr    = np.mean(d["rewards"])
        sr    = np.std(d["rewards"])
        lats  = d["latencies"] or [0]
        p50   = np.percentile(lats, 50)
        p95   = np.percentile(lats, 95)
        avg_b = np.mean(d["batch_sizes"]) if d["batch_sizes"] else 0
        tp    = np.mean(d["throughputs"])
        viols = sum(d["sla_violations"])
        serves= max(sum(d["n_serve_actions"]), 1)
        viol_pct = 100.0 * viols / serves
        print(
            f"  {AGENT_LABELS[name]:<14}  {mr:>+12.1f}  {sr:>8.1f}  "
            f"{p50:>11.1f}  {p95:>9.1f}  "
            f"{avg_b:>11.1f}  {tp:>11,.0f}  {viol_pct:>9.1f}%"
        )
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate comparison graphs for the RL batching project")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help=f"Episodes per agent (default {N_EPISODES})")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Skip the combined dashboard PNG")
    args = parser.parse_args()

    SEP = "=" * 64
    print(SEP)
    print("  Dynamic Request Batching — Comparison Graph Suite")
    print(SEP)
    print(f"  Episodes per agent : {args.episodes}")
    print(f"  Agents             : {', '.join(AGENT_ORDER)}")
    print(f"  Output directory   : {GRAPHS_DIR}")
    print(SEP + "\n")

    # 1. Evaluate
    print("[ Evaluation ]")
    data = run_evaluation(args.episodes)

    # 2. Summary table
    print_summary(data)

    # 3. Individual charts
    print("\n[ Saving individual charts ]")
    save_all_individual(data)

    # 4. Dashboard
    if not args.no_dashboard:
        print("\n[ Building dashboard ]")
        dash_path = build_dashboard(data)
        print(f"  ✓  00_dashboard.png")

    print(f"\n{SEP}")
    print(f"  All graphs saved to: {GRAPHS_DIR}")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
