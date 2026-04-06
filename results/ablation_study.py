"""
results/ablation_study.py

Feature-importance ablation study for the Dynamic Request Batching PPO agent.

For each of the 6 state features, we train a "crippled" variant of the agent
where that feature is permanently zeroed out.  The resulting mean reward is
compared against a full model baseline, and the drop (Δreward) is used as a
proxy for how much the agent *relies on* that feature.

Why ablation?
-------------
In RL, the policy is a black-box neural network.  We cannot simply inspect
weights to understand which observations matter most (unlike linear models).
Ablation is the standard practical alternative: you *remove* an input, retrain,
and measure the performance cost — features that hurt most when removed are
the most informative.

Usage:
    python results/ablation_study.py

Outputs:
    results/ablation_vertical.png   — vertical bar chart (reward drop)
    results/ablation_horizontal.png — horizontal bar chart sorted by importance
    results/ablation_dashboard.png  — both charts side by side
    Terminal                        — formatted table
"""

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env.batching_env import BatchingEnv
from config import CONFIG

# ── Hyperparameters for the ablation runs ────────────────────────────────────
ABLATION_TIMESTEPS = 200_000
N_SEEDS            = 2
N_EVAL_EPISODES    = 10
N_ENVS             = 4

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Observation feature names (must match BatchingEnv._get_obs() order) ──────
FEATURE_NAMES = [
    "pending_requests",
    "oldest_wait_ms",
    "request_rate",
    "since_serve_ms",
    "fill_ratio",
    "time_of_day",
]

FEATURE_LABELS = [
    "Pending\nRequests",
    "Oldest\nWait (ms)",
    "Request\nRate",
    "Since Last\nServe (ms)",
    "Fill\nRatio",
    "Time of\nDay",
]

# ── Dark-theme design tokens ──────────────────────────────────────────────────
BG_FIG    = "#0a0e1a"
BG_AX     = "#141c30"
COL_GRID  = "#2a3450"
COL_SPINE = "#2a3450"
COL_TEXT  = "#e8eaf6"
COL_MUTED = "#7986cb"

# Gradient palette: most important → least important
# These are assigned after sorting, so colour = rank, not feature index
RANK_COLORS = [
    "#00e5ff",   # rank 1 — cyan (most important)
    "#40c4ff",   # rank 2
    "#7c4dff",   # rank 3
    "#e040fb",   # rank 4
    "#ff6f00",   # rank 5
    "#f44336",   # rank 6 — red (least important)
]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "text.color":       COL_TEXT,
    "axes.labelcolor":  COL_TEXT,
    "xtick.color":      COL_TEXT,
    "ytick.color":      COL_TEXT,
})


# ─────────────────────────────────────────────────────────────────────────────
# Crippled environment wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ZeroFeatureWrapper(gym.ObservationWrapper):
    """Gymnasium wrapper that permanently zeros out one observation dimension.

    By zeroing a feature at the *environment* level (not the model level),
    we ensure the agent can never recover the information through correlated
    features — the ablation is clean.
    """

    def __init__(self, env: gym.Env, feature_idx: int):
        super().__init__(env)
        self.feature_idx = feature_idx

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        obs[self.feature_idx] = 0.0
        return obs


# ─────────────────────────────────────────────────────────────────────────────
# Training and evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_env_factory(feature_idx: int | None, seed: int):
    """Return a factory that builds a (possibly crippled) Monitor-wrapped env."""
    def _init():
        env = BatchingEnv(seed=seed)
        if feature_idx is not None:
            env = ZeroFeatureWrapper(env, feature_idx)
        env = Monitor(env, filename=None)
        return env
    return _init


def train_one(feature_idx: int | None, seed: int) -> float:
    """Train a PPO agent (possibly ablated) and return its mean eval reward."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_fns  = [_make_env_factory(feature_idx, seed + i) for i in range(N_ENVS)]
    train_env = DummyVecEnv(env_fns)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=ABLATION_TIMESTEPS, progress_bar=False)
    train_env.close()

    # Deterministic evaluation
    rewards  = []
    eval_env = BatchingEnv(seed=seed + 9999)
    if feature_idx is not None:
        eval_env = ZeroFeatureWrapper(eval_env, feature_idx)

    for ep in range(N_EVAL_EPISODES):
        obs, _ = eval_env.reset(seed=seed + 9999 + ep)
        total  = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(int(action))
            total += reward
        rewards.append(total)

    eval_env.close()
    return float(np.mean(rewards))


# ─────────────────────────────────────────────────────────────────────────────
# Main ablation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation() -> dict:
    """Train full model + one ablated model per feature; return results dict."""
    results = {}

    # ── Full model (no feature zeroed) ───────────────────────────────────────
    print("\n  Training FULL model (baseline)...")
    seed_rewards = []
    for s in range(N_SEEDS):
        t0 = time.time()
        r  = train_one(None, seed=s * 100)
        elapsed = time.time() - t0
        print(f"    seed {s}: mean_reward={r:+.1f}  ({elapsed:.0f}s)")
        seed_rewards.append(r)
    results["_full_"] = float(np.mean(seed_rewards))
    print(f"  → Full model mean reward: {results['_full_']:+.1f}\n")

    # ── One ablation per feature ──────────────────────────────────────────────
    for idx, name in enumerate(FEATURE_NAMES):
        print(f"  Ablating feature [{idx}] '{name}'...")
        seed_rewards = []
        for s in range(N_SEEDS):
            t0 = time.time()
            r  = train_one(idx, seed=s * 100)
            elapsed = time.time() - t0
            print(f"    seed {s}: mean_reward={r:+.1f}  ({elapsed:.0f}s)")
            seed_rewards.append(r)
        results[name] = float(np.mean(seed_rewards))
        drop = results["_full_"] - results[name]
        print(f"  → Mean reward: {results[name]:+.1f}   Δ = {drop:+.1f}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared by both plots
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax, title="", xlabel="", ylabel="", grid_axis="y"):
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


def _rank_colors(drops):
    """Assign gradient colours by rank (highest drop = rank 0 = most important)."""
    order = np.argsort(drops)[::-1]   # descending: most important first
    colours = [""] * len(drops)
    for rank, idx in enumerate(order):
        colours[idx] = RANK_COLORS[rank]
    return colours


# ─────────────────────────────────────────────────────────────────────────────
# Plot A — Vertical bar chart (original layout, enhanced)
# ─────────────────────────────────────────────────────────────────────────────

def plot_vertical(results: dict, ax=None) -> str | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
        fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.18)

    full_r = results["_full_"]
    drops  = [full_r - results[n] for n in FEATURE_NAMES]
    colors = _rank_colors(drops)
    x      = np.arange(len(FEATURE_NAMES))

    bars = ax.bar(x, drops, color=colors, width=0.58, alpha=0.90, zorder=3)

    for bar, drop, col in zip(bars, drops, colors):
        pct = 100.0 * drop / abs(full_r) if full_r != 0 else 0
        label = f"Δ{drop:+.0f}\n({pct:+.0f}%)"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, max(drop, 0)),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom",
            color=COL_TEXT, fontsize=8, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_LABELS, color=COL_TEXT, fontsize=8.5)
    ax.axhline(0, color=COL_TEXT, linewidth=0.7, linestyle="--", alpha=0.4)
    _style(ax,
           title=(f"Feature Importance via Ablation\n"
                  f"Full model reward: {full_r:+.0f}  |  higher bar = more important"),
           ylabel="Reward Drop  (Full − Ablated)")

    # Legend: rank 1 = most important
    legend_elems = [
        mpatches.Patch(facecolor=RANK_COLORS[0], label="Most important"),
        mpatches.Patch(facecolor=RANK_COLORS[-1], label="Least important"),
    ]
    ax.legend(handles=legend_elems, facecolor=BG_AX, edgecolor=COL_SPINE,
              labelcolor=COL_TEXT, fontsize=8.5)

    if standalone:
        path = os.path.join(RESULTS_DIR, "ablation_vertical.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
        plt.close(fig)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Plot B — Horizontal bar chart sorted by importance (easier to rank visually)
# ─────────────────────────────────────────────────────────────────────────────

def plot_horizontal(results: dict, ax=None) -> str | None:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG_FIG)
        fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.12)

    full_r = results["_full_"]
    drops  = np.array([full_r - results[n] for n in FEATURE_NAMES])

    # Sort descending (most important at top)
    order  = np.argsort(drops)[::-1]
    sorted_labels = [FEATURE_LABELS[i] for i in order]
    sorted_drops  = drops[order]
    sorted_colors = RANK_COLORS[:len(drops)]   # already ranked best→worst

    y = np.arange(len(sorted_drops))

    bars = ax.barh(y, sorted_drops, color=sorted_colors, height=0.60,
                   alpha=0.90, zorder=3)

    for bar, drop in zip(bars, sorted_drops):
        pct = 100.0 * drop / abs(full_r) if full_r != 0 else 0
        ax.annotate(
            f"  Δ{drop:+.0f}  ({pct:+.0f}%)",
            xy=(max(drop, 0), bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0), textcoords="offset points",
            ha="left", va="center",
            color=COL_TEXT, fontsize=8.5, fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_labels, color=COL_TEXT, fontsize=9)
    ax.invert_yaxis()   # most important at top
    ax.axvline(0, color=COL_TEXT, linewidth=0.7, linestyle="--", alpha=0.4)
    _style(ax,
           title="Feature Importance Ranking  (sorted)",
           xlabel="Reward Drop  (Full − Ablated)",
           grid_axis="x")

    # Rank badges
    for rank, (bar, lbl) in enumerate(zip(bars, sorted_labels)):
        ax.text(
            -0.03, bar.get_y() + bar.get_height() / 2,
            f"#{rank+1}",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center",
            color=sorted_colors[rank], fontsize=8.5, fontweight="bold",
        )

    if standalone:
        path = os.path.join(RESULTS_DIR, "ablation_horizontal.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
        plt.close(fig)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard: side-by-side
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(results: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG_FIG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.14, wspace=0.32)
    fig.suptitle(
        "Dynamic Request Batching — Feature Ablation Study",
        color=COL_TEXT, fontsize=14, fontweight="bold", y=0.97,
    )

    plot_vertical(results, ax=axes[0])
    plot_horizontal(results, ax=axes[1])

    path = os.path.join(RESULTS_DIR, "ablation_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: dict):
    full_r  = results["_full_"]
    drops   = [full_r - results[n] for n in FEATURE_NAMES]
    order   = sorted(range(len(FEATURE_NAMES)), key=lambda i: drops[i], reverse=True)

    W = 66
    print("\n" + "=" * W)
    print(f"  {'Rank':<5}  {'Feature':<22}  {'Ablated Reward':>14}  {'Δ Reward':>10}")
    print("-" * W)
    for rank, idx in enumerate(order):
        name  = FEATURE_NAMES[idx]
        r     = results[name]
        drop  = drops[idx]
        bar   = "█" * max(0, int(drop / max(max(drops), 1) * 18))
        print(f"  #{rank+1:<4}  {name:<22}  {r:>+14.1f}  {drop:>+10.1f}  {bar}")
    print("-" * W)
    print(f"  {'Full model (baseline)':<28}  {full_r:>+14.1f}")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 60
    print(SEP)
    print("  Dynamic Request Batching — Ablation Study")
    print(SEP)
    print(f"  Features      : {len(FEATURE_NAMES)}")
    print(f"  Seeds/feature : {N_SEEDS}")
    print(f"  Timesteps     : {ABLATION_TIMESTEPS:,}")
    print(f"  Total runs    : {(len(FEATURE_NAMES) + 1) * N_SEEDS}")
    t_est = (len(FEATURE_NAMES) + 1) * N_SEEDS * (ABLATION_TIMESTEPS / 1200) / 60
    print(f"  Est. runtime  : ~{t_est:.0f} min on CPU")
    print(SEP)

    t0 = time.time()
    results = run_ablation()
    elapsed = time.time() - t0

    print_table(results)

    v_path = os.path.join(RESULTS_DIR, "ablation_vertical.png")
    h_path = os.path.join(RESULTS_DIR, "ablation_horizontal.png")

    fig_v, ax_v = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
    fig_v.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.18)
    plot_vertical(results, ax=ax_v)
    fig_v.savefig(v_path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig_v)

    fig_h, ax_h = plt.subplots(figsize=(9, 6), facecolor=BG_FIG)
    fig_h.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.12)
    plot_horizontal(results, ax=ax_h)
    fig_h.savefig(h_path, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    plt.close(fig_h)

    d_path = plot_dashboard(results)

    print(f"\n✓  Vertical chart    → {v_path}")
    print(f"✓  Horizontal chart  → {h_path}")
    print(f"✓  Dashboard         → {d_path}")
    print(f"✓  Total elapsed     : {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
