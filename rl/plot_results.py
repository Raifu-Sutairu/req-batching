import matplotlib.pyplot as plt
import numpy as np
import os

# Data
policies = [
    "No Batching", 
    "Fixed Timer (50ms)", 
    "Fixed Size Cap (128)", 
    "Cloudflare Exp-Prob", 
    "PPO ONNX Agent"
]

upstream_reduction = [0.00, 97.76, 98.18, 82.16, 97.39]
p50_ms = [1.06, 50.88, 53.00, 5.36, 43.46]
p99_ms = [31.71, 63.34, 4056.08, 63.24, 63.34]
mean_batch_size = [1.00, 44.69, 54.83, 5.61, 38.28]
forced_flush_pct = [0.00, 0.00, 93.52, 0.00, 0.00]

# Consistent colors
colors = {
    "No Batching": "#888888",          # Grey
    "Fixed Timer (50ms)": "#1f77b4",   # Blue
    "Fixed Size Cap (128)": "#d62728", # Red
    "Cloudflare Exp-Prob": "#ff7f0e",  # Orange
    "PPO ONNX Agent": "#2ca02c"        # Green
}

def setup_plot(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    return fig, ax

# 1. p50 vs p99 latency scatter
fig1, ax1 = setup_plot(
    "Latency Profile: Avoiding p99 Catastrophe", 
    "p50 Latency (ms)", 
    "p99 Latency (ms, log scale)"
)
ax1.set_yscale('log')
for i, p in enumerate(policies):
    ax1.scatter(p50_ms[i], p99_ms[i], color=colors[p], s=100, label=p, zorder=5)
    ax1.annotate(
        p, (p50_ms[i], p99_ms[i]),
        xytext=(8, 0), textcoords='offset points',
        va='center', fontsize=9
    )
ax1.grid(True, which="both", ls="--", alpha=0.3)
fig1.tight_layout()
fig1.savefig('chart_1_latency_scatter.png', dpi=300)
print("Saved chart_1_latency_scatter.png")

# 2. Upstream call reduction vs mean batch size
fig2, ax2 = setup_plot(
    "Efficiency Frontier: Higher Reduction per Batch Size", 
    "Mean Batch Size", 
    "Upstream Call Reduction (%)"
)
for i, p in enumerate(policies):
    ax2.scatter(mean_batch_size[i], upstream_reduction[i], color=colors[p], s=100, zorder=5)
    ax2.annotate(
        p, (mean_batch_size[i], upstream_reduction[i]),
        xytext=(8, -8), textcoords='offset points',
        va='top', fontsize=9
    )
fig2.tight_layout()
fig2.savefig('chart_2_efficiency_frontier.png', dpi=300)
print("Saved chart_2_efficiency_frontier.png")

# 3. p50 latency bar chart (exclude Fixed Size Cap)
viable_policies = [p for p in policies if p != "Fixed Size Cap (128)"]
viable_p50 = [p50_ms[policies.index(p)] for p in viable_policies]
viable_colors = [colors[p] for p in viable_policies]

fig3, ax3 = setup_plot(
    "RL Learns to Flush Early for Lower Latency", 
    "", 
    "p50 Latency (ms)"
)
x_pos = np.arange(len(viable_policies))
bars = ax3.bar(x_pos, viable_p50, color=viable_colors, width=0.6, zorder=3)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(viable_policies, rotation=15, ha='right')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)
fig3.tight_layout()
fig3.savefig('chart_3_p50_viable.png', dpi=300)
print("Saved chart_3_p50_viable.png")

# 4. Forced flush rate bar chart
fig4, ax4 = setup_plot(
    "Safety Envelope Validation: Outlier Exposed", 
    "", 
    "Forced Flush Rate (%)"
)
x_pos4 = np.arange(len(policies))
bars4 = ax4.bar(x_pos4, forced_flush_pct, color=[colors[p] for p in policies], width=0.6, zorder=3)
ax4.set_xticks(x_pos4)
ax4.set_xticklabels(policies, rotation=15, ha='right')

for bar in bars4:
    height = bar.get_height()
    if height > 0:
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(bar.get_x() + bar.get_width()/2., 0.5,
                 '0%', ha='center', va='bottom', fontsize=9, alpha=0.5)

fig4.tight_layout()
fig4.savefig('chart_4_forced_flush.png', dpi=300)
print("Saved chart_4_forced_flush.png")
