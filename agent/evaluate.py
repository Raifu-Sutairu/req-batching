"""
agent/evaluate.py

Evaluation: PPO vs Discrete SAC vs Cloudflare baseline.

Generates
─────────
  results/comparison.png           – 6-panel comparison (dark theme)
  results/comparison_paper.png     – same figure, white-background research style
  results/decision_heatmap.png     – 2×2 PPO heatmap (dark)
  results/decision_heatmap_paper.png – same heatmap, paper style
  results/traffic_regimes.png      – regime robustness (dark)
  results/traffic_regimes_paper.png  – regime robustness (paper style)
  stdout                           – formatted summary table

Usage
─────
    python agent/evaluate.py
    python agent/evaluate.py --ppo-model models/ppo_final
    python agent/evaluate.py --skip-heatmap --skip-regimes
"""

import os
import sys
import argparse
import contextlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from env.batching_env import BatchingEnv, gpu_processing_ms
from baselines.cloudflare_formula import CloudflareBaseline, GreedyBatchBaseline
from agent.discrete_sac import DiscreteSAC
from agent.d3qn import D3QN
from config import CONFIG, SAC_CONFIG, D3QN_CONFIG, EXPERIMENT_CONFIGS

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR         = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_PPO_MODEL   = os.path.join(ROOT, "models", "ppo_final")
DEFAULT_SAC_MODEL   = os.path.join(ROOT, "models", "sac_final")
DEFAULT_D3QN_MODEL  = os.path.join(ROOT, "models", "d3qn_final")
PPO_VECNORM         = os.path.join(ROOT, "models", "ppo_vecnorm.pkl")
SAC_VECNORM         = os.path.join(ROOT, "models", "sac_vecnorm.pkl")
D3QN_VECNORM        = os.path.join(ROOT, "models", "d3qn_vecnorm.pkl")

N_EPISODES  = 30
SEED_OFFSET = 100

# ── Colour palettes ────────────────────────────────────────────────────────────
# Dark theme
DARK_BG    = "#0a0e1a"
DARK_AX    = "#141c30"
DARK_FG    = "#e8eaf6"
DARK_GRID  = "#2a3450"

DARK_COLORS = {
    "PPO":         "#00e5ff",
    "Discrete SAC":"#ff4081",
    "D3QN":        "#69ff47",
    "Cloudflare":  "#ff9100",
    "GreedyBatch": "#b2ff59",
}

# Paper theme — Wong (2011) colorblind-safe palette
PAPER_COLORS = {
    "PPO":         "#0072B2",   # blue
    "Discrete SAC":"#D55E00",   # vermillion
    "D3QN":        "#CC79A7",   # reddish purple
    "Cloudflare":  "#009E73",   # green
    "GreedyBatch": "#E69F00",   # orange
}

AGENTS       = ["PPO", "Discrete SAC", "D3QN", "Cloudflare"]
PAPER_AGENTS = ["PPO", "Discrete SAC", "D3QN", "Cloudflare"]

# ── Matplotlib style contexts ──────────────────────────────────────────────────

PAPER_RC = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#444444",
    "axes.grid":         True,
    "grid.color":        "#E0E0E0",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "text.color":        "black",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "axes.titlecolor":   "black",
}

@contextlib.contextmanager
def paper_style():
    with plt.rc_context(PAPER_RC):
        yield


# ── Agent wrappers ─────────────────────────────────────────────────────────────

class PPOWrapper:
    def __init__(self, model: PPO, vecnorm: VecNormalize | None = None):
        self.model   = model
        self.vecnorm = vecnorm

    def predict(self, obs: np.ndarray) -> int:
        o = self.vecnorm.normalize_obs(obs) if self.vecnorm else obs
        action, _ = self.model.predict(o, deterministic=True)
        return int(action)


class SACWrapper:
    def __init__(self, model: DiscreteSAC, vecnorm: VecNormalize | None = None):
        self.model   = model
        self.vecnorm = vecnorm

    def predict(self, obs: np.ndarray) -> int:
        o = self.vecnorm.normalize_obs(obs) if self.vecnorm else obs
        return self.model.predict(o, deterministic=True)


class D3QNWrapper:
    def __init__(self, model: D3QN, vecnorm: VecNormalize | None = None):
        self.model   = model
        self.vecnorm = vecnorm

    def predict(self, obs: np.ndarray) -> int:
        o = self.vecnorm.normalize_obs(obs) if self.vecnorm else obs
        return self.model.predict(o, deterministic=True)


# ── Model loaders ──────────────────────────────────────────────────────────────

def _load_ppo(model_path: str, vecnorm_path: str) -> PPOWrapper:
    m = PPO.load(model_path)
    v = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        dummy = DummyVecEnv([lambda: BatchingEnv()])
        v = VecNormalize.load(vecnorm_path, dummy)
        v.training = False; v.norm_reward = False
    return PPOWrapper(m, v)


def _load_sac(model_path: str, vecnorm_path: str) -> SACWrapper:
    obs_dim   = BatchingEnv().observation_space.shape[0]
    n_actions = BatchingEnv().action_space.n
    m = DiscreteSAC.load(model_path, obs_dim=obs_dim, n_actions=n_actions, cfg=SAC_CONFIG)
    v = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        dummy = DummyVecEnv([lambda: BatchingEnv()])
        v = VecNormalize.load(vecnorm_path, dummy)
        v.training = False; v.norm_reward = False
    return SACWrapper(m, v)


def _load_d3qn(model_path: str, vecnorm_path: str) -> D3QNWrapper:
    obs_dim   = BatchingEnv().observation_space.shape[0]
    n_actions = BatchingEnv().action_space.n
    m = D3QN.load(model_path, obs_dim=obs_dim, n_actions=n_actions, cfg=D3QN_CONFIG)
    v = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        dummy = DummyVecEnv([lambda: BatchingEnv()])
        v = VecNormalize.load(vecnorm_path, dummy)
        v.training = False; v.norm_reward = False
    return D3QNWrapper(m, v)


# ── Data collection ────────────────────────────────────────────────────────────

def collect(agent, env: BatchingEnv, n_episodes: int, seed_offset: int) -> dict:
    rewards, latency_raw, all_batch_sizes = [], [], []
    throughputs, sla_viol_rates, n_dispatches = [], [], []

    for ep in range(n_episodes):
        obs, _      = env.reset(seed=seed_offset + ep)
        ep_reward   = 0.0
        ep_dispatch = 0
        prev_served = 0
        terminated = truncated = False

        if hasattr(agent, "reset"):
            agent.reset()

        while not (terminated or truncated):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if action == 1:
                ep_dispatch += 1
                served_now = info["total_served"] - prev_served
                prev_served = info["total_served"]
                if served_now > 0:
                    all_batch_sizes.append(served_now)

        latency_raw.extend(env._latency_samples)
        rewards.append(ep_reward)
        throughputs.append(info["total_served"])
        sla_viol_rates.append(info["sla_violation_rate"])
        n_dispatches.append(ep_dispatch)

    return {
        "rewards":        rewards,
        "latency_raw":    latency_raw,
        "batch_sizes":    all_batch_sizes,
        "throughputs":    throughputs,
        "sla_viol_rates": sla_viol_rates,
        "n_dispatches":   n_dispatches,
    }


def run_all(ppo_path: str, ppo_vn: str,
            sac_path: str, sac_vn: str,
            d3qn_path: str, d3qn_vn: str) -> dict[str, dict]:
    env    = BatchingEnv()
    agents = {
        "PPO":         _load_ppo(ppo_path, ppo_vn),
        "Discrete SAC":_load_sac(sac_path, sac_vn),
        "D3QN":        _load_d3qn(d3qn_path, d3qn_vn),
        "Cloudflare":  CloudflareBaseline(),
        "GreedyBatch": GreedyBatchBaseline(),
    }
    all_data = {}
    for name in AGENTS + ["GreedyBatch"]:
        print(f"  Evaluating {name:<14} ({N_EPISODES} eps) …", end=" ", flush=True)
        d = collect(agents[name], env, N_EPISODES, SEED_OFFSET)
        all_data[name] = d
        print(f"reward = {np.mean(d['rewards']):+.0f}")
    env.close()
    return all_data


def run_regimes(ppo_path: str, ppo_vn: str,
                sac_path: str, sac_vn: str,
                d3qn_path: str, d3qn_vn: str,
                n_episodes: int = 8) -> dict[str, dict[str, dict]]:
    regime_keys = {
        "off_peak":  "Off-Peak\n(400 req/s)",
        "standard":  "Standard\n(2 000 req/s)",
        "peak_load": "Peak Load\n(5 000 req/s)",
    }
    results = {}
    for key, label in regime_keys.items():
        cfg = EXPERIMENT_CONFIGS[key]
        env = BatchingEnv(config=cfg)
        agents = {
            "PPO":         _load_ppo(ppo_path, ppo_vn),
            "Discrete SAC":_load_sac(sac_path, sac_vn),
            "D3QN":        _load_d3qn(d3qn_path, d3qn_vn),
            "Cloudflare":  CloudflareBaseline(config=cfg),
        }
        regime_data = {}
        for name in PAPER_AGENTS:
            regime_data[name] = collect(agents[name], env, n_episodes, SEED_OFFSET + 200)
        env.close()
        results[label] = regime_data
        print("    " + label.replace("\n", " ") + "  " +
              "  ".join(f"{n}={np.mean(regime_data[n]['rewards']):+.0f}"
                        for n in PAPER_AGENTS))
    return results


# ── Panel helpers ──────────────────────────────────────────────────────────────

def _style_dark(ax, legend=None):
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values(): sp.set_edgecolor(DARK_GRID)
    ax.tick_params(colors=DARK_FG, labelsize=9)
    ax.xaxis.label.set_color(DARK_FG); ax.yaxis.label.set_color(DARK_FG)
    ax.title.set_color(DARK_FG)
    ax.grid(True, color=DARK_GRID, linewidth=0.5, alpha=0.8)
    if legend:
        legend.get_frame().set(facecolor=DARK_AX, edgecolor=DARK_GRID)
        for t in legend.get_texts(): t.set_color(DARK_FG)


def _style_paper(ax, legend=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#E0E0E0", linestyle="--", linewidth=0.5)
    if legend:
        legend.get_frame().set(facecolor="white", edgecolor="#CCCCCC")


def _annotate_bars(ax, bars, values, fmt="{:.0f}", color="black", yoff=4):
    for bar, v in zip(bars, values):
        ax.annotate(fmt.format(v),
                    xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, yoff), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=color,
                    fontweight="bold")


# ── Comparison panels (theme-agnostic content) ─────────────────────────────────

def _panel_reward(ax, data, colors, style_fn, fg="black"):
    means  = [np.mean(data[n]["rewards"]) for n in PAPER_AGENTS]
    stds   = [np.std(data[n]["rewards"])  for n in PAPER_AGENTS]
    x      = np.arange(len(PAPER_AGENTS))
    bars   = ax.bar(x, means, yerr=stds, color=[colors[n] for n in PAPER_AGENTS],
                    width=0.55, capsize=5, alpha=0.88, zorder=3,
                    error_kw={"color": fg, "linewidth": 1.2})
    _annotate_bars(ax, bars, means, fmt="{:+.0f}", color=fg)
    ax.set_xticks(x); ax.set_xticklabels(PAPER_AGENTS)
    ax.axhline(0, color=fg, linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_title("Mean Episode Reward ± Std", fontweight="bold")
    ax.set_ylabel("Episode Reward")
    style_fn(ax)


def _panel_latency_bars(ax, data, colors, style_fn, fg="black"):
    sla   = CONFIG["max_latency_ms"]
    width = 0.22
    x     = np.arange(len(PAPER_AGENTS))
    pcts  = [50, 95, 99]
    alphas= [0.90, 0.65, 0.40]
    for i, (pct, al) in enumerate(zip(pcts, alphas)):
        vals = [float(np.percentile(data[n]["latency_raw"], pct))
                if data[n]["latency_raw"] else 0.0 for n in PAPER_AGENTS]
        bars = ax.bar(x + (i-1)*width, vals, width, label=f"P{pct}",
                      color=[colors[n] for n in PAPER_AGENTS], alpha=al, zorder=3)
        if pct == 99:
            for b in bars: b.set_hatch("//")
    ax.axhline(sla, color="#CC0000", linewidth=1.2, linestyle=":",
               alpha=0.8, label=f"SLA ({sla} ms)")
    ax.set_xticks(x); ax.set_xticklabels(PAPER_AGENTS)
    leg = ax.legend(fontsize=8, loc="upper right")
    ax.set_title("P50 / P95 / P99 Total Latency", fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    style_fn(ax, leg)


def _panel_batch_dist(ax, data, colors, style_fn, fg="black"):
    for name in PAPER_AGENTS:
        bs = data[name]["batch_sizes"]
        if not bs: continue
        ax.hist(bs, bins=40, range=(0, CONFIG["max_batch_size"]),
                alpha=0.40, color=colors[name], label=name, density=True)
        try:
            from scipy.stats import gaussian_kde
            if len(set(bs)) > 3:
                kde = gaussian_kde(bs, bw_method=0.2)
                xs  = np.linspace(0, CONFIG["max_batch_size"], 300)
                ax.plot(xs, kde(xs), color=colors[name], linewidth=2.0)
        except Exception:
            pass
    leg = ax.legend(fontsize=8)
    ax.set_title("Batch Size Distribution", fontweight="bold")
    ax.set_xlabel("Batch Size (requests)")
    ax.set_ylabel("Density")
    style_fn(ax, leg)


def _panel_latency_cdf(ax, data, colors, style_fn, fg="black"):
    sla = CONFIG["max_latency_ms"]
    for name in PAPER_AGENTS:
        raw = np.sort(np.array(data[name]["latency_raw"]))
        if raw.size == 0: continue
        cdf = np.arange(1, len(raw) + 1) / len(raw) * 100
        ax.plot(raw, cdf, color=colors[name], linewidth=2.2, label=name, alpha=0.9)
        p95 = float(np.percentile(raw, 95))
        ax.axvline(p95, color=colors[name], linewidth=0.8, linestyle="--", alpha=0.45)
    ax.axvline(sla, color="#CC0000", linewidth=1.4, linestyle=":",
               alpha=0.85, label=f"SLA ({sla} ms)")
    ax.set_xlabel("Total Latency (ms)")
    ax.set_ylabel("Cumulative Requests (%)")
    ax.set_ylim(0, 102); ax.set_xlim(left=0)
    leg = ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Latency CDF — Client-Perceived Latency", fontweight="bold")
    ax.axhspan(95, 100, color="#CC0000", alpha=0.05, zorder=0)
    ax.text(ax.get_xlim()[1]*0.98, 97, "tail (P95+)",
            ha="right", va="center", color="#CC0000", fontsize=7.5, alpha=0.7)
    style_fn(ax, leg)


def _panel_reward_per_dispatch(ax, data, colors, style_fn, fg="black"):
    means, stds, names_ = [], [], []
    for name in PAPER_AGENTS:
        rp = np.array(data[name]["rewards"]) / np.maximum(data[name]["n_dispatches"], 1)
        means.append(np.mean(rp)); stds.append(np.std(rp)); names_.append(name)
    x    = np.arange(len(names_))
    bars = ax.bar(x, means, yerr=stds, color=[colors[n] for n in names_],
                  width=0.55, capsize=5, alpha=0.88, zorder=3,
                  error_kw={"color": fg, "linewidth": 1.2})
    _annotate_bars(ax, bars, means, fmt="{:.1f}", color=fg)
    ax.set_xticks(x); ax.set_xticklabels(names_)
    ax.set_title("Reward per Dispatch\n(↑ higher = smarter batching)", fontweight="bold")
    ax.set_ylabel("Avg Reward / Dispatch")
    style_fn(ax)


def _panel_efficiency_scatter(ax, data, colors, style_fn, fg="black"):
    for name in PAPER_AGENTS:
        bs   = data[name]["batch_sizes"]
        dp   = data[name]["n_dispatches"]
        ax.scatter(np.mean(dp), np.mean(bs) if bs else 0,
                   s=180, color=colors[name], label=name,
                   zorder=4, edgecolors=fg, linewidths=0.7)
        ax.annotate(name, (np.mean(dp), np.mean(bs) if bs else 0),
                    textcoords="offset points", xytext=(7, 3),
                    color=colors[name], fontsize=9, fontweight="bold")
    ax.set_xlabel("Avg Dispatches per Episode")
    ax.set_ylabel("Avg Batch Size (requests)")
    ax.set_title("Dispatch Efficiency\n(↖ fewer dispatches, larger batches = better)", fontweight="bold")
    style_fn(ax)


# ── Full comparison figure ─────────────────────────────────────────────────────

def generate_figure(data: dict, style: str = "dark") -> str:
    is_dark  = style == "dark"
    colors   = DARK_COLORS if is_dark else PAPER_COLORS
    bg_fig   = DARK_BG if is_dark else "white"
    fg       = DARK_FG if is_dark else "black"
    style_fn = _style_dark if is_dark else _style_paper
    path     = os.path.join(RESULTS_DIR,
                            "comparison.png" if is_dark else "comparison_paper.png")

    rc = {"font.family": "DejaVu Sans", "text.color": fg} if is_dark else PAPER_RC
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 3, figsize=(20, 11),
                                 facecolor=bg_fig)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.09,
                            wspace=0.33, hspace=0.46)
        title_kw = dict(color=fg, fontsize=14, fontweight="bold", y=0.97)
        fig.suptitle("Dynamic Request Batching — PPO vs SAC vs D3QN vs Cloudflare",
                     **title_kw)

        _panel_reward(axes[0, 0], data, colors, style_fn, fg)
        _panel_latency_bars(axes[0, 1], data, colors, style_fn, fg)
        _panel_batch_dist(axes[0, 2], data, colors, style_fn, fg)
        _panel_reward_per_dispatch(axes[1, 0], data, colors, style_fn, fg)
        _panel_latency_cdf(axes[1, 1], data, colors, style_fn, fg)
        _panel_efficiency_scatter(axes[1, 2], data, colors, style_fn, fg)

        legend_elems = [mpatches.Patch(facecolor=colors[n], label=n) for n in PAPER_AGENTS]
        fig.legend(handles=legend_elems, loc="lower center", ncol=len(PAPER_AGENTS),
                   facecolor=DARK_AX if is_dark else "white",
                   edgecolor=DARK_GRID if is_dark else "#CCCCCC",
                   labelcolor=fg, fontsize=9,
                   bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=bg_fig)
        plt.close(fig)
    return path


# ── Decision heatmap ──────────────────────────────────────────────────────────

def generate_heatmap(ppo_path: str, ppo_vn: str,
                     style: str = "dark", grid: int = 70) -> str:
    import torch

    ppo_model = PPO.load(ppo_path)
    ppo_model.policy.set_training_mode(False)
    vecnorm   = None
    if ppo_vn and os.path.exists(ppo_vn):
        dummy   = DummyVecEnv([lambda: BatchingEnv()])
        vecnorm = VecNormalize.load(ppo_vn, dummy)
        vecnorm.training = False; vecnorm.norm_reward = False

    max_b  = CONFIG["max_batch_size"]
    max_w  = CONFIG["max_latency_ms"] // 2
    rate_v = CONFIG["arrival_rate"] * 1.2
    tod_v  = 14.0
    trend_v= 0.0   # neutral rate trend

    since_slices = [
        (10,  "since_serve = 10 ms\n(just dispatched — expect Wait)"),
        (40,  "since_serve = 40 ms\n(short gap)"),
        (80,  "since_serve = 80 ms\n(medium gap)"),
        (150, "since_serve = 150 ms\n(long idle — expect Serve)"),
    ]

    batch_vals = np.linspace(0, max_b, grid)
    wait_vals  = np.linspace(0, max_w, grid)

    is_dark  = style == "dark"
    bg_fig   = DARK_BG if is_dark else "white"
    fg       = DARK_FG if is_dark else "black"
    ax_bg    = DARK_AX if is_dark else "white"
    spine_c  = DARK_GRID if is_dark else "#888888"
    path     = os.path.join(RESULTS_DIR,
                            "decision_heatmap.png" if is_dark else "decision_heatmap_paper.png")

    rc = {"font.family": "DejaVu Sans"} if is_dark else PAPER_RC
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor=bg_fig)
        fig.subplots_adjust(left=0.08, right=0.91, top=0.90, bottom=0.08,
                            wspace=0.30, hspace=0.40)
        fig.suptitle(
            "PPO Learned Policy — P(Serve) across Time-Since-Last-Dispatch\n"
            "green = Serve  |  red = Wait  |  white dashed = P = 0.5 boundary",
            color=fg, fontsize=12, fontweight="bold", y=0.97,
        )

        norm  = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
        last_im = None

        for idx, (since_ms, subtitle) in enumerate(since_slices):
            ax = axes[idx // 2, idx % 2]
            p_serve = np.zeros((grid, grid), dtype=np.float32)

            for i, bs in enumerate(batch_vals):
                est_proc = gpu_processing_ms(max(1, int(bs)), CONFIG)
                budget   = max(1.0, CONFIG["max_latency_ms"] - est_proc)
                for j, ow in enumerate(wait_vals):
                    urgency = ow / budget
                    fill    = bs / max(max_b, 1)
                    obs = np.array([bs, ow, rate_v, since_ms, fill, tod_v,
                                    urgency, trend_v], dtype=np.float32)
                    if vecnorm is not None:
                        obs = vecnorm.normalize_obs(obs)
                    obs_t = ppo_model.policy.obs_to_tensor(obs)[0]
                    with torch.no_grad():
                        dist  = ppo_model.policy.get_distribution(obs_t)
                        logit = dist.distribution.logits[0, 1]
                        p     = torch.sigmoid(logit).item()
                    p_serve[j, i] = p

            last_im = ax.imshow(p_serve, origin="lower", aspect="auto",
                                extent=[0, max_b, 0, max_w],
                                cmap=plt.cm.RdYlGn, norm=norm, interpolation="bilinear")
            try:
                ax.contour(batch_vals, wait_vals, p_serve, levels=[0.5],
                           colors=["white"], linewidths=2.0, linestyles="--")
            except Exception:
                pass

            # SLA breach boundary
            sla_y = [min(max_w, CONFIG["max_latency_ms"] -
                         gpu_processing_ms(max(1, int(b)), CONFIG)) for b in batch_vals]
            ax.plot(batch_vals, sla_y, color="#CC0000", lw=1.4, ls=":",
                    alpha=0.85, label="SLA breach (urgency=1)")

            # Cloudflare 80% threshold
            cf_y = [min(max_w, 0.80 * max(50.0, CONFIG["max_latency_ms"] -
                    CONFIG["gpu_base_ms"] - CONFIG["gpu_per_request_ms"] * max(1, int(b))))
                    for b in batch_vals]
            ax.plot(batch_vals, cf_y, color="#FF8C00", lw=1.2, ls="-.",
                    alpha=0.75, label="Cloudflare 80% threshold")

            ax.set_facecolor(ax_bg)
            for sp in ax.spines.values(): sp.set_edgecolor(spine_c)
            ax.tick_params(colors=fg, labelsize=8)
            ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)
            ax.set_title(subtitle, color=fg, fontsize=9, fontweight="bold")
            ax.set_xlabel("Pending Batch Size (requests)", labelpad=6)
            ax.set_ylabel("Oldest Wait Time (ms)", labelpad=6)
            if idx == 0:
                leg = ax.legend(facecolor=ax_bg, edgecolor=spine_c,
                                labelcolor=fg, fontsize=7.5, loc="upper left")

        cax  = fig.add_axes([0.925, 0.15, 0.013, 0.70])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("P(Serve)", color=fg, fontsize=10)
        cbar.ax.tick_params(colors=fg)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=fg)
        fig.text(0.5, 0.01,
                 f"Fixed: rate={rate_v:.0f} req/s  |  time_of_day=14:00  "
                 f"|  rate_trend=0  |  since_serve varies across panels",
                 ha="center", color="#90a4ae" if is_dark else "#666666", fontsize=8)

        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=bg_fig)
        plt.close(fig)
    return path


# ── Traffic regime plot ────────────────────────────────────────────────────────

def generate_regime_plot(regime_data: dict, style: str = "dark") -> str:
    is_dark  = style == "dark"
    colors   = DARK_COLORS if is_dark else PAPER_COLORS
    bg_fig   = DARK_BG if is_dark else "white"
    fg       = DARK_FG if is_dark else "black"
    style_fn = _style_dark if is_dark else _style_paper
    path     = os.path.join(RESULTS_DIR,
                            "traffic_regimes.png" if is_dark else "traffic_regimes_paper.png")

    regimes = list(regime_data.keys())
    x       = np.arange(len(regimes))
    w       = 0.23

    rc = {"font.family": "DejaVu Sans", "text.color": fg} if is_dark else PAPER_RC
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=bg_fig)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.18, wspace=0.38)
        fig.suptitle("Traffic Regime Robustness — Off-Peak / Standard / Peak Load",
                     color=fg, fontsize=13, fontweight="bold", y=0.97)

        titles = [
            ("Mean Reward\n(↑ higher is better)",
             lambda d: np.mean(d["rewards"]),
             "Episode Reward",
             lambda v: f"{v/1000:.0f}k"),
            ("P95 Total Latency\n(↓ lower is better)",
             lambda d: float(np.percentile(d["latency_raw"], 95))
                       if d["latency_raw"] else 0.0,
             "Latency (ms)",
             None),
            ("Avg Batch Size\n(↑ larger = better GPU util)",
             lambda d: np.mean(d["batch_sizes"]) if d["batch_sizes"] else 0.0,
             "Avg Batch Size (requests)",
             None),
        ]

        for (ax, (title, fn, ylabel, fmt_fn)) in zip(axes, titles):
            for i, name in enumerate(PAPER_AGENTS):
                vals = [fn(regime_data[r][name]) for r in regimes]
                offset = (i - 1) * w
                bars = ax.bar(x + offset, vals, w, color=colors[name],
                              label=name, alpha=0.88, zorder=3)
                for bar, v in zip(bars, vals):
                    label = fmt_fn(v) if fmt_fn else f"{v:.0f}"
                    ax.annotate(label,
                                xy=(bar.get_x() + bar.get_width() / 2, v),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", color=fg,
                                fontsize=7.5, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(regimes, fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold")
            leg = ax.legend(fontsize=8, loc="upper left")
            style_fn(ax, leg)

        # SLA line on latency panel
        axes[1].axhline(CONFIG["max_latency_ms"], color="#CC0000",
                        linewidth=1.2, linestyle=":", alpha=0.8,
                        label=f"SLA ({CONFIG['max_latency_ms']} ms)")
        axes[1].legend(fontsize=8, loc="upper left")

        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=bg_fig)
        plt.close(fig)
    return path


# ── Summary table ──────────────────────────────────────────────────────────────

def print_table(data: dict):
    W = 105
    print("\n" + "=" * W)
    print(f"{'Agent':<16}  {'Mean Reward':>12}  {'Std':>8}  "
          f"{'P50':>8}  {'P95':>8}  {'P99':>8}  "
          f"{'SLA Viol%':>10}  {'AvgBatch':>9}  {'Rwd/Disp':>9}")
    print("-" * W)
    for name in AGENTS + ["GreedyBatch"]:
        d = data.get(name, {})
        if not d: continue
        mr  = np.mean(d["rewards"])
        sr  = np.std(d["rewards"])
        raw = d["latency_raw"]
        p50 = float(np.percentile(raw, 50)) if raw else float("nan")
        p95 = float(np.percentile(raw, 95)) if raw else float("nan")
        p99 = float(np.percentile(raw, 99)) if raw else float("nan")
        sla = np.mean(d["sla_viol_rates"]) * 100
        ab  = np.mean(d["batch_sizes"]) if d["batch_sizes"] else float("nan")
        rd  = np.mean(np.array(d["rewards"]) / np.maximum(d["n_dispatches"], 1))
        print(f"{name:<16}  {mr:>+12.0f}  {sr:>8.0f}  "
              f"{p50:>8.1f}  {p95:>8.1f}  {p99:>8.1f}  "
              f"{sla:>9.2f}%  {ab:>9.1f}  {rd:>9.1f}")
    print("=" * W)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-model",    default=DEFAULT_PPO_MODEL)
    parser.add_argument("--sac-model",    default=DEFAULT_SAC_MODEL)
    parser.add_argument("--d3qn-model",   default=DEFAULT_D3QN_MODEL)
    parser.add_argument("--ppo-vecnorm",  default=PPO_VECNORM)
    parser.add_argument("--sac-vecnorm",  default=SAC_VECNORM)
    parser.add_argument("--d3qn-vecnorm", default=D3QN_VECNORM)
    parser.add_argument("--skip-heatmap", action="store_true")
    parser.add_argument("--skip-regimes", action="store_true")
    args = parser.parse_args()

    ppo_ok  = os.path.exists(args.ppo_model + ".zip") or os.path.exists(args.ppo_model)
    sac_ok  = os.path.exists(args.sac_model + ".pt")
    d3qn_ok = os.path.exists(args.d3qn_model + ".pt")

    if not ppo_ok:
        print(f"[ERROR] PPO model not found: {args.ppo_model}")
        print("  Run: python agent/train.py")
        sys.exit(1)
    if not sac_ok:
        print(f"[ERROR] SAC model not found: {args.sac_model}")
        print("  Run: python agent/train_sac.py")
        sys.exit(1)
    if not d3qn_ok:
        print(f"[ERROR] D3QN model not found: {args.d3qn_model}")
        print("  Run: python agent/train_d3qn.py")
        sys.exit(1)

    print("=" * 70)
    print("  Dynamic Request Batching — Evaluation")
    print("=" * 70)
    print(f"  PPO model  : {args.ppo_model}")
    print(f"  SAC model  : {args.sac_model}")
    print(f"  D3QN model : {args.d3qn_model}")
    print(f"  Episodes   : {N_EPISODES} per agent")
    print("=" * 70 + "\n")

    # ── Main comparison ───────────────────────────────────────────────────
    data = run_all(args.ppo_model, args.ppo_vecnorm,
                   args.sac_model, args.sac_vecnorm,
                   args.d3qn_model, args.d3qn_vecnorm)
    print_table(data)

    for style in ("dark", "paper"):
        p = generate_figure(data, style=style)
        print(f"\n  Comparison ({style:5s})  → {p}")

    # ── Decision heatmap ──────────────────────────────────────────────────
    if not args.skip_heatmap:
        for style in ("dark", "paper"):
            print(f"  Heatmap ({style:5s})      …", end=" ", flush=True)
            hp = generate_heatmap(args.ppo_model, args.ppo_vecnorm, style=style)
            print(f"→ {hp}")

    # ── Traffic regime robustness ─────────────────────────────────────────
    if not args.skip_regimes:
        print("\n  Traffic regimes …")
        rd = run_regimes(args.ppo_model, args.ppo_vecnorm,
                         args.sac_model, args.sac_vecnorm,
                         args.d3qn_model, args.d3qn_vecnorm)
        for style in ("dark", "paper"):
            rp = generate_regime_plot(rd, style=style)
            print(f"  Regimes ({style:5s})      → {rp}")


if __name__ == "__main__":
    main()
