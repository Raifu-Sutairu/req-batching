import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches

# Global style rules
plt.rcParams['font.family'] = 'DejaVu Sans'

# Data
policies = [
    "No Batching", 
    "Fixed Timer (50ms)", 
    "Fixed Size Cap (128)", 
    "PPO ONNX Agent"
]

upstream_reduction = [0.00, 97.76, 98.18, 97.39]
p50_ms = [1.06, 50.88, 53.00, 43.46]
p99_ms = [31.71, 63.34, 4056.08, 63.34]
mean_batch_size = [1.00, 44.69, 54.83, 38.28]
forced_flush_pct = [0.00, 0.00, 93.52, 0.00]

colors = {
    "No Batching": "#888888",
    "Fixed Timer (50ms)": "#2196F3",
    "Fixed Size Cap (128)": "#E53935",
    "PPO ONNX Agent": "#43A047"
}

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "plots")
os.makedirs(out_dir, exist_ok=True)

def setup_plot(title, subtitle, xlabel, ylabel, both_grids=False):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_title(title, pad=30, fontsize=15, fontweight='bold')
    fig.text(0.5, 0.90, subtitle, ha='center', fontsize=10, color='#555555')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if both_grids:
        ax.grid(True, which="both", linestyle='--', alpha=0.4, color='#cccccc')
    else:
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#cccccc')
        
    return fig, ax

# Chart 1 - Latency Profile
fig1, ax1 = setup_plot(
    "RL Matches Throughput at Lower Latency", 
    "x-axis = median wait time, y-axis (log) = worst-case wait time. Bottom-left is best.",
    "p50 Latency (ms)", 
    "p99 Latency (ms, log scale)",
    both_grids=True
)
ax1.set_yscale('log')
ax1.set_xlim(0, 65)

# Safe zone rectangle
rect = patches.Rectangle((0, 0), 45, 100, linewidth=0, edgecolor='none', facecolor='#e8f5e9', zorder=1)
ax1.add_patch(rect)
ax1.text(5, 5, "safe zone", color='#81c784', fontsize=9, alpha=0.7, zorder=2)

# Ideal diagonal line
ax1.plot([0, 65], [1, 5000], linestyle='--', color='#d3d3d3', zorder=2)
ax1.text(25, 60, "ideal: low p50 + low p99 →", color='#b0b0b0', fontsize=9, rotation=25, zorder=2)

for i, p in enumerate(policies):
    ax1.scatter(p50_ms[i], p99_ms[i], color=colors[p], s=220, zorder=5)
    
    if p == "PPO ONNX Agent":
        ax1.annotate("PPO: 7ms lower p50\nthan Fixed Timer,\nsame p99 ceiling",
                     xy=(p50_ms[i], p99_ms[i]), xytext=(p50_ms[i]-15, p99_ms[i]*3),
                     arrowprops=dict(arrowstyle="->", color="#333333", connectionstyle="arc3,rad=.2"),
                     fontsize=9, color="#333333", ha='center')
        ax1.text(p50_ms[i], p99_ms[i]*0.8, p, fontsize=9, color="#333333", ha='center', va='top')
    elif p == "Fixed Size Cap (128)":
        ax1.annotate("Size Cap alone:\np99 blows out to 4056ms\nunder sparse traffic",
                     xy=(p50_ms[i], p99_ms[i]), xytext=(p50_ms[i]-10, p99_ms[i]*0.1),
                     arrowprops=dict(arrowstyle="->", color="#333333", connectionstyle="arc3,rad=-.2"),
                     fontsize=9, color="#333333", ha='center')
        ax1.text(p50_ms[i]-2, p99_ms[i], p, fontsize=9, color="#333333", ha='right', va='center')
    elif p == "Fixed Timer (50ms)":
        ax1.text(p50_ms[i]+2, p99_ms[i], p, fontsize=9, color="#333333", ha='left', va='center')
    elif p == "No Batching":
        ax1.annotate("0% upstream reduction",
                     xy=(p50_ms[i], p99_ms[i]), xytext=(p50_ms[i]+2, p99_ms[i]*0.5),
                     fontsize=9, color="#888888", ha='left', va='top')
        ax1.text(p50_ms[i]+2, p99_ms[i], p, fontsize=9, color="#333333", ha='left', va='center')
    else:
        ax1.text(p50_ms[i]+2, p99_ms[i], p, fontsize=9, color="#333333", ha='left', va='center')

fig1.tight_layout(rect=[0, 0.04, 1, 0.9])
out_path1 = os.path.join(out_dir, 'chart_1_latency_scatter.png')
fig1.savefig(out_path1, dpi=150)
print(f"Saved {out_path1}")

# Chart 2 - Efficiency Frontier
fig2, ax2 = setup_plot(
    "PPO Achieves Near-Identical Upstream Reduction with 14% Smaller Batches", 
    "Each point = one policy. Right = larger batches. Up = fewer upstream calls. PPO sits left of Fixed Timer at the same height.",
    "Mean Batch Size", 
    "Upstream Call Reduction (%)",
    both_grids=True
)
ax2.set_xlim(0, 60)
ax2.set_ylim(-5, 105)

# Dashed vertical lines
ax2.axvline(x=38.28, color='#43A047', linestyle='--', zorder=2)
ax2.text(38.28, 5, "PPO mean batch", color='#43A047', fontsize=9, rotation=90, va='bottom', ha='right')

ax2.axvline(x=44.69, color='#2196F3', linestyle='--', zorder=2)
ax2.text(44.69, 5, "Timer mean batch", color='#2196F3', fontsize=9, rotation=90, va='bottom', ha='left')

ax2.annotate("← 6.4 fewer requests\nper upstream call", xy=(41.5, 50), ha='center', fontsize=9, color='#333333')

for i, p in enumerate(policies):
    ax2.scatter(mean_batch_size[i], upstream_reduction[i], color=colors[p], s=220, zorder=5)
    
    if p == "PPO ONNX Agent":
        ax2.annotate("97.39% reduction\nwith smaller batches", xy=(mean_batch_size[i], upstream_reduction[i]), 
                     xytext=(mean_batch_size[i]-5, upstream_reduction[i]-15),
                     arrowprops=dict(arrowstyle="->", color="#333333"),
                     fontsize=9, color="#333333", ha='center')
        ax2.text(mean_batch_size[i], upstream_reduction[i]+2, p, fontsize=9, color="#333333", ha='center')
    elif p == "No Batching":
        ax2.annotate("baseline:\nno coalescing", xy=(1.0, 0), xytext=(5, -2),
                     fontsize=9, color="#333333", ha='left', va='center')
    elif p == "Fixed Timer (50ms)":
        ax2.text(mean_batch_size[i]+1, upstream_reduction[i]-2, p, fontsize=9, color="#333333", ha='left')

fig2.tight_layout(rect=[0, 0.04, 1, 0.9])
out_path2 = os.path.join(out_dir, 'chart_2_efficiency_frontier.png')
fig2.savefig(out_path2, dpi=150)
print(f"Saved {out_path2}")

# Chart 3 - p50 Latency Bar
viable_policies = [p for p in policies if p != "Fixed Size Cap (128)"]
viable_p50 = [p50_ms[policies.index(p)] for p in viable_policies]
viable_colors = [colors[p] for p in viable_policies]

fig3, ax3 = setup_plot(
    "RL Learns to Flush Early — Median Latency Beats Fixed Timer", 
    "The agent flushes before the 50ms deadline when it's smart to, reducing median wait time.",
    "", 
    "p50 Latency (ms)",
    both_grids=False
)
ax3.set_ylim(0, 60)
ax3.axhline(y=50, color='#E53935', linestyle='--', zorder=2)
ax3.text(0.1, 51, "50ms timeout ceiling", color='#E53935', fontsize=9, va='bottom')

x_pos = np.arange(len(viable_policies))
bars = ax3.bar(x_pos, viable_p50, color=viable_colors, width=0.5, zorder=3)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(viable_policies, fontsize=10)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}ms', ha='center', va='bottom', fontsize=9, color='#333333')

# Double-headed arrow between Fixed Timer and PPO
ax3.plot([1, 2], [50.88, 50.88], linestyle=':', color='#888888', zorder=2)
ax3.annotate('', xy=(2, 43.46), xytext=(2, 50.88), arrowprops=dict(arrowstyle='<->', color='#1b5e20'))
ax3.text(2.1, 47, "−7.42ms\n(14.6% faster)", color='#1b5e20', fontsize=9, va='center')

# Add note about excluded policy
fig3.text(0.5, 0.02, "Fixed Size Cap excluded — its 4056ms p99 makes its p50 an unreliable comparator.", 
          ha='center', fontsize=9, color='#888888')

fig3.tight_layout(rect=[0, 0.06, 1, 0.9])
out_path3 = os.path.join(out_dir, 'chart_3_p50_viable.png')
fig3.savefig(out_path3, dpi=150)
print(f"Saved {out_path3}")

# Chart 4 - Forced Flush Rate
fig4, ax4 = setup_plot(
    "Safety Envelope Validation: Hard Timer Prevents Flush Stalls", 
    "Forced flush rate = how often the safety net overrides the policy. PPO never needs it.",
    "", 
    "Forced Flush Rate (%)",
    both_grids=False
)
ax4.set_ylim(0, 110)
ax4.axhline(y=5, color='#ff9800', linestyle='--', zorder=2)
ax4.text(3, 7, "design target: <5%", color='#ff9800', fontsize=9, va='bottom', ha='center')

x_pos4 = np.arange(len(policies))
bar_colors = [colors[p] if p == "Fixed Size Cap (128)" else colors[p] for p in policies]
bar_alphas = [1.0 if p == "Fixed Size Cap (128)" else 0.4 for p in policies]

bars4 = ax4.bar(x_pos4, forced_flush_pct, color=bar_colors, width=0.5, zorder=3)
for i, bar in enumerate(bars4):
    bar.set_alpha(bar_alphas[i])

ax4.set_xticks(x_pos4)
ax4.set_xticklabels(policies, fontsize=10)

for i, bar in enumerate(bars4):
    height = bar.get_height()
    if height > 0:
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9, color='#333333')
    else:
        ax4.text(bar.get_x() + bar.get_width()/2., 1,
                 '0%', ha='center', va='bottom', fontsize=9, color='#333333')

# Annotation for Fixed Size Cap
cap_idx = policies.index("Fixed Size Cap (128)")
ax4.annotate("93.5%: the timer fallback\nwas disabled — batches\nstalled under low traffic",
             xy=(cap_idx, 93.5), xytext=(cap_idx + 0.3, 60),
             arrowprops=dict(arrowstyle="->", color="#333333", connectionstyle="angle,angleA=0,angleB=90,rad=10"),
             fontsize=9, color="#333333", ha='left')

fig4.tight_layout(rect=[0, 0.04, 1, 0.9])
out_path4 = os.path.join(out_dir, 'chart_4_forced_flush.png')
fig4.savefig(out_path4, dpi=150)
print(f"Saved {out_path4}")
