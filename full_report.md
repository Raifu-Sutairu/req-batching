# Full Project Report — Dynamic Request Batching RL Agent
**Date:** 2026-03-11 | **Seed:** 42 | **Traffic pattern trained:** Poisson

---

## 1. Project Goal

Build a Reinforcement Learning agent that learns an optimal **request-batching policy** for a server:
- **Wait (0):** hold requests in a queue, accumulate a larger batch
- **Serve (1):** dispatch all queued requests as one batch

The goal is to balance **throughput** (reward bigger batches) vs **latency** (penalise long queue wait times), while respecting a **500 ms SLA**.

---

## 2. Repository Structure

```
├── env/
│   ├── batching_env.py        # Gymnasium environment
│   └── traffic_generator.py   # Poisson / Bursty / Time-varying generators
├── sac_agent/
│   ├── sac_agent.py           # SACAgent class (full train + inference loop)
│   ├── network.py             # LSTMActor + LSTMCritic (twin critics)
│   ├── replay_buffer.py       # Prioritized Experience Replay (PER) buffer
│   ├── train_sac.py           # Training entry-point script
│   └── evaluate_sac.py        # Evaluation / rendering script
├── baselines/
│   ├── random_agent.py
│   ├── greedy_agent.py
│   └── cloudflare_formula.py
├── agent/                     # PPO/DQN agents (earlier iterations)
├── results/
│   ├── ablation_study.py
│   ├── comparison_plots.png
│   └── decision_heatmap.png
├── checkpoints_sac/           # Saved model weights (.pth files)
├── logs_sac/                  # Training logs per traffic pattern
├── config.py                  # All hyperparameters in one place
└── run_all.py                 # Multi-agent benchmark runner
```

---

## 3. Environment ([BatchingEnv](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/env/batching_env.py#41-286))

| Property | Value |
|---|---|
| **State space** | 6-dimensional Box |
| **Action space** | Discrete(2) — Wait / Serve |
| **Episode length** | 60,000 ms (60 seconds simulated) |
| **Decision interval** | 10 ms → **1,000 steps/episode** |
| **SLA threshold** | 500 ms (oldest request) |

### State Vector

| Index | Feature | Description |
|---|---|---|
| 0 | `pending_requests` | # requests currently queued |
| 1 | [oldest_wait_ms](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/env/batching_env.py#266-271) | Age of the oldest pending request (ms) |
| 2 | `request_rate` | EMA of observed arrival rate (req/s) |
| 3 | `time_since_last_serve_ms` | ms since last Serve action |
| 4 | `batch_fill_ratio` | `pending / max_batch_size` ∈ [0, 1] |
| 5 | `time_of_day` | Fractional hour ∈ [0, 24) |

### Reward Function
```
r = α × batch_size           (if Serve)
  - β × oldest_wait_ms
  - γ × idle_penalty          (if Wait & queue empty)
  - sla_penalty               (if oldest_wait_ms > 500 ms)
```

| Param | Value | Role |
|---|---|---|
| α | 1.0 | Efficiency incentive |
| β | 0.01 | Latency penalty multiplier |
| γ | 0.1 | Idle-wait penalty |
| SLA penalty | -5.0 | Hard penalty per SLA breach |

### Traffic Patterns
| Pattern | Description |
|---|---|
| **Poisson** | Constant-rate random arrivals (λ=10 req/s) |
| **Bursty** | Periodic traffic spikes |
| **Time-varying** | Peak hours (08:00–18:00) at 2.5× rate, off-peak at 0.5× |

---

## 4. SAC Agent Architecture

The core algorithm is **Soft Actor-Critic (SAC)** with three key extensions:

### 4.1 LSTM Backbone
All networks share an LSTM backbone that consumes the last 20 timesteps (= 200 ms of traffic history), enabling the agent to detect burst onsets and regime changes.

```
Input: (batch, seq_len=20, state_dim=6)
    → LSTM(hidden=256, layers=1)
    → FC(256 → 128) + ReLU
    → Output head
```

### 4.2 Twin Critics
Two independent [LSTMCritic](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/sac_agent/network.py#134-177) networks (Q1, Q2). The Bellman target always uses `min(Q1, Q2)` to prevent Q-value overestimation — a core SAC stability mechanism.

### 4.3 Automatic Entropy Tuning
The temperature `α` is learned via gradient descent to keep policy entropy near `target_entropy = -1.0` (= -dim(A) for binary actions). No manual tuning required.

### 4.4 Prioritized Experience Replay (PER)
| PER Param | Value |
|---|---|
| Buffer capacity | 100,000 transitions |
| `per_alpha` (prioritisation) | 0.6 |
| `per_beta_start` (IS correction) | 0.4 → 1.0 |
| `per_beta_frames` (annealing) | 100,000 steps |
| Training minibatch size | 256 |
| Warm-up (random actions) | 1,000 steps |

### 4.5 Key Hyperparameters

| Param | Value |
|---|---|
| Learning rate | 3 × 10⁻⁴ (Adam) |
| Discount γ | 0.99 |
| Polyak τ | 0.005 |
| α_init (entropy temp) | 0.2 |
| Target entropy | -1.0 |
| Episodes (Poisson run) | 400 |
| Max steps/episode | 1,000 |
| Gradient clipping | max_norm = 1.0 |

---

## 5. Training Results — Poisson Traffic (Seed 42)

### 5.1 Episode Reward Progression

| Phase | Episodes | Mean Reward | Peak Reward |
|---|---|---|---|
| Early (1–50) | 50 | ~71.5 | 126.5 |
| Mid (51–200) | 150 | ~89.0 | 138.9 |
| Late (201–400) | 200 | ~95.5 | 139.0 |

**Overall (400 episodes):**
- **Mean episode reward:** ~90.3
- **Max episode reward:** ~139.0 (ep. 211)
- **Min episode reward:** ~33.1 (ep. 2)
- **All episodes ran full 1,000 steps** — no early terminations

### 5.2 Learning Curve Summary

The agent shows a clear **upward trend**:
- Episodes 1–30: high variance, rapid early learning (~33–115 range)
- Episodes 30–100: oscillating around 80–130, policy solidifying
- Episodes 100–400: stable performance, mean reward converging ≥ 100 on many episodes
- **Best checkpoint saved** at episode with highest reward (tracked automatically)

### 5.3 Loss Curves

**Actor loss** (negative is good for SAC — maximising Q − αH):

| Stage | Actor Loss Range |
|---|---|
| Early | −0.35 → −0.54 |
| Mid to Late | Continues decreasing (more negative) |

**Critic loss** (lower = better Bellman fit):

| Stage | Critic Loss (avg Q1+Q2)/2 |
|---|---|
| Early | ~0.0198 → 0.0165 |
| Trending | Steadily decreasing → near zero |

**Entropy temperature (α)**:

| Stage | α Value |
|---|---|
| Start | 0.208 |
| Growing | 0.216 → 0.235 |
| Later | Auto-tuned, converges toward target |

**Policy entropy** (nats):

| Stage | Entropy |
|---|---|
| Early | 0.45 → 0.57 (increasing, exploring) |
| Late | Decreasing toward ~0.001–0.003 (policy converging/deterministic) |

> [!NOTE]
> The entropy collapsing near zero at late training is a sign the policy is becoming very deterministic. This is expected once the agent has found a confident strategy, but may indicate reduced adaptability to unseen traffic. Consider monitoring this if deploying on bursty or time-varying traffic.

### 5.4 Avg Wait Time (per episode)

- Episode 1: **14.4 ms** (high — agent still random)
- Episodes 30–100: **5–7 ms range** (agent learned to avoid long waits)
- Episodes 100–400: **4–6 ms range** (stable, well within SLA)

> [!TIP]
> Average wait times stabilize between **4–6 ms**, far below the 500 ms SLA — indicating the agent effectively prevents SLA violations.

### 5.5 Avg Batch Size

- Episode 1: 0.123 (agent mostly serving tiny/empty batches)
- Episodes 5–30: Rapidly drops toward 0.001–0.005
- Episodes 30+: Near **0.0** (agent has learned a policy where it mostly dispatches small or nearly-full batches very regularly)

> [!IMPORTANT]
> The avg_batch_size metric dropping to ~0.0 after ep. 30 could be a **logging artifact** (e.g., normalised by max_batch_size=100), rather than the agent truly making zero-size batches. The rising reward confirms it is not dispatching empty batches, as that would incur latency penalties with no alpha reward. Verify the metric computation in [train_sac.py](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/sac_agent/train_sac.py).

---

## 6. Checkpoints

| File | Description |
|---|---|
| [checkpoints_sac/poisson/sac_ep50_poisson.pth](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/checkpoints_sac/poisson/sac_ep50_poisson.pth) | Checkpoint at ep 50 |
| [checkpoints_sac/poisson/sac_ep100_poisson.pth](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/checkpoints_sac/poisson/sac_ep100_poisson.pth)–`ep400` | Every 50 episodes |
| [checkpoints_sac/poisson/sac_best.pth](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/checkpoints_sac/poisson/sac_best.pth) | Best reward checkpoint |
| [checkpoints_sac/poisson/sac_final.pth](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/checkpoints_sac/poisson/sac_final.pth) | End-of-run final weights |
| [checkpoints_sac/sac_best.pth](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/checkpoints_sac/sac_best.pth) | Root-level best (top-level) |
| [checkpoints_sac/sac_final.pth](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/checkpoints_sac/sac_final.pth) | Root-level final |

All checkpoint files are **~12.8 MB** (full network + optimizer states).

---

## 7. Baselines

Three baselines are implemented in `baselines/`:

| Agent | Strategy |
|---|---|
| `RandomAgent` | Uniform random Wait/Serve each step |
| `GreedyAgent` | Always serve immediately (action = 1) |
| `CloudflareFormula` | Rule-based: serve when queue fill ratio or wait exceeds thresholds |

These are used in [run_all.py](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/run_all.py) and [results/ablation_study.py](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/results/ablation_study.py) for comparison.

---

## 8. Multi-Traffic Training Plan

Config defines training across 3 traffic patterns × 3 seeds:

| Traffic | Seed | Status |
|---|---|---|
| Poisson | 42 | ✅ Complete (400 ep) |
| Bursty | 42 | 🔲 Not started |
| Time-varying | 42 | 🔲 Not started |
| Poisson | 123, 777 | 🔲 Not started |
| Bursty | 123, 777 | 🔲 Not started |
| Time-varying | 123, 777 | 🔲 Not started |

---

## 9. Known Issues & Observations

| Issue | Severity | Notes |
|---|---|---|
| `avg_batch_sizes` drops to 0.0 after ~ep 30 | Medium | Likely a logging normalisation bug — rewards still positive, so agent IS batching |
| Policy entropy collapses near ep 200+ | Low/Expected | Policy converging, but may hurt generalisation to bursty/time-varying |
| Only Poisson traffic trained so far | High (for full study) | No bursty/time-varying checkpoints exist yet |
| Root [logs_sac/training_logs.json](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/logs_sac/training_logs.json) has only 5 episodes | Low | Appears to be a stale/initial test log — detailed per-traffic logs in subdirs |
| `avg_batch_sizes` & `avg_wait_times` in root log are all 0.0 | Low | Confirmed bug in the 5-episode root run — not tracked properly |

---

## 10. Recommendations & Next Steps

1. **Run bursty and time-varying training** — configure [train_sac.py](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/sac_agent/train_sac.py) for `traffic_pattern = "bursty"` / `"time_varying"` and run 400 episodes each
2. **Run 3 seeds per pattern** — seeds [42, 123, 777] as configured; needed for statistically robust results
3. **Investigate avg_batch_size logging** — verify the metric computation; likely needs `batch_size / max_batch_size` distinction
4. **Run formal evaluation** using [evaluate_sac.py](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/sac_agent/evaluate_sac.py) with [sac_best.pth](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/checkpoints_sac/sac_best.pth) against all baselines
5. **Generate comparison plots** — [results/ablation_study.py](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/results/ablation_study.py) + [results/comparison_plots.png](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/results/comparison_plots.png) already scaffolded
6. **Monitor entropy at late training** — if entropy collapses fully on bursty traffic, consider a minimum entropy floor
7. **Generate decision heatmap** — already referenced in [results/decision_heatmap.png](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/results/decision_heatmap.png), useful for visualising policy behaviour across queue-fill vs wait-time space

---

## 11. File Health Summary

| Component | Status |
|---|---|
| [env/batching_env.py](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/env/batching_env.py) | ✅ Complete, well-documented |
| [sac_agent/network.py](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/sac_agent/network.py) | ✅ LSTM Actor + Twin Critics, orthogonal init |
| [sac_agent/sac_agent.py](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/sac_agent/sac_agent.py) | ✅ Full SAC with PER, auto-entropy tuning |
| [sac_agent/train_sac.py](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/sac_agent/train_sac.py) | ✅ Entry-point, active file |
| [config.py](file:///c:/Users/shiri/RL%20project1/Dynamic-Request-Batching-Reinforcement-Learning-Agent/config.py) | ✅ Well-structured with rationale comments |
| `baselines/` | ✅ 3 baselines implemented |
| `results/` | ⚠️ Plots exist but ablation incomplete |
| `logs_sac/bursty/` | ⚠️ Directory exists but likely empty |
| `logs_sac/time_varying/` | ⚠️ Directory exists but likely empty |
