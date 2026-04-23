# RL Agents for Request Batching
  
> **Part:** Policy Gradient, Actor-Critic, PPO, and Forecast-Aware Accelerated RL

This folder implements several reinforcement learning agents for the **dynamic request batching** problem. The agent learns *when* to serve accumulated requests (`WAIT` vs `SKIP`) to maximize batching efficiency while minimizing client-side latency.

---

## Agents Implemented

| Agent | Method | Key Idea |
|-------|--------|----------|
| **REINFORCE** | Policy Gradient | Collects full episodes, computes Monte Carlo returns, uses a learned value baseline for variance reduction |
| **A2C** | Advantage Actor-Critic | Shared actor-critic network with N-step TD updates; lower variance than REINFORCE via bootstrapping |
| **PPO** | Proximal Policy Optimization | Clipped surrogate objective + GAE (Generalized Advantage Estimation); multi-epoch mini-batch updates for best sample efficiency |
| **Predictive Dyna-Q** | Forecast-aware accelerated RL | Uses grouped arrival-rate prediction, discretized Q-learning, and model-based planning updates to react faster under changing traffic |

## Why Add Predictive Dyna-Q?

This repo now includes a fourth path that is deliberately **different** from the teammate methods you listed:

- not `SAC + LSTM + PER`
- not `IMPALA`
- not `NN + PPO`
- not `PPO / DQN / SAC`

The design is inspired by the paper:

- **Using Grouped Linear Prediction and Accelerated Reinforcement Learning for Online Content Caching**

It is not a literal cache-replacement reproduction, because this environment is about **request batching**, not content eviction. Instead, it adapts the paper's two transferable ideas:

- **Grouped linear prediction** of near-future demand
- **Accelerated RL** via simulated/planning updates from a learned one-step model

That makes it a strong "same project, new method" option.

### Why Policy-Based Methods?

From the project notes:
- The action space is **discrete** (WAIT/SKIP), but the policy should be **continuous** (smooth transition between waiting and serving)
- Q-Learning / DQN struggle with smooth decision boundaries — a small change in state shouldn't cause a drastic action flip
- Policy-based methods output **probabilities** `π(WAIT|s) = 0.42` → 42% lean towards serving, which smoothly adapts as conditions change
- PPO is the state-of-the-art for this class of problems

---

## Project Structure

```
Ashrith/
├── predictive_dynaq_agent.py   # Main grouped prediction + Dyna-Q planning agent
├── live_simulation.py          # Live dashboard for final demo
├── compare_all.py              # Final multi-seed evaluation table
├── train_agents.py             # Unified training script
├── run_all.py                  # One-command experiment runner
├── env/                        # Environment + traffic model
├── baselines/                  # Rule-based baselines
├── legacy/                     # Older REINFORCE/A2C/PPO code kept for comparison
├── checkpoints/                # Main model artifacts
├── logs/                       # Main training logs
└── results/                    # Final evaluation outputs
```

---

## How to Run

```bash
# From the project root directory:

# Train all 4 agents, evaluate, and generate comparison plots
python3 -m Ashrith.run_all

# Train a specific agent
python3 -m Ashrith.train_agents --agent reinforce --episodes 300
python3 -m Ashrith.train_agents --agent a2c --episodes 300
python3 -m Ashrith.train_agents --agent ppo --episodes 300
python3 -m Ashrith.train_agents --agent predictive_dynaq --episodes 300

# Run comparison only (requires trained checkpoints)
python3 -m Ashrith.compare_all

# Run inference API for reverse proxy
python3 -m Ashrith.api_dynaq
```

---

## Paper Fit Notes

Of the paper directions discussed, the best fit for this environment is:

- **Grouped Linear Prediction + Accelerated RL**

The other two papers focus on richer cache-management settings with cache contents, hits/misses, replacement decisions, or batch-level cache admission. Those ideas are interesting, but they do not map as cleanly onto this repo's current `WAIT` vs `SKIP` batching environment without redesigning the problem itself.

---

## Predictive Dyna-Q Overview

At each step, the agent:

1. observes the current batching state
2. predicts the next arrival-rate feature from grouped recent history
3. augments the state with that forecast
4. updates a Q-table on the real transition
5. performs extra planning updates from its learned transition model

This gives a simple, fast, paper-inspired baseline that is easier to explain than a deep recurrent architecture.

---

## Algorithm Deep Dive

### REINFORCE (Policy Gradient)

```
for each episode:
    Collect trajectory: (s₀, a₀, r₀), ..., (sₜ, aₜ, rₜ)
    Compute returns:    Gₜ = Σ γᵏ rₜ₊ₖ
    Compute advantage:  Aₜ = Gₜ - V(sₜ)          ← baseline reduces variance
    Update policy:      θ ← θ + α ∇log π(aₜ|sₜ) · Aₜ
    Update baseline:    minimize MSE(V(sₜ), Gₜ)
```

### A2C (Advantage Actor-Critic)

```
for each N steps:
    Compute N-step return: Rₜ = rₜ + γrₜ₊₁ + ... + γⁿV(sₜ₊ₙ)
    Compute advantage:     Aₜ = Rₜ - V(sₜ)
    Update actor:  minimize  -log π(a|s) · A(s,a)
    Update critic: minimize  MSE(V(s), Rₜ)
    Both share the same network trunk → critic improves actor's features
```

### PPO (Proximal Policy Optimization)

```
for each rollout of T steps:
    Compute GAE: Aₜ = Σ (γλ)ˡ δₜ₊ₗ   where δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
    for K epochs:
        for each mini-batch:
            ratio = π_new(a|s) / π_old(a|s)
            L_clip = min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
            Update θ to maximize L_clip
```

---

## Environment Interface

The agents interact with `BatchingEnv` from `Ashrith/env/`:

- **State** (6-dim, normalized [0,1]): `[batch_size, wait_time, queue_length, time_since_last_skip, arrival_rate, system_load]`
- **Action**: `0 = WAIT` (keep accumulating), `1 = SKIP` (send batch now)
- **Reward**: `α × (batch_efficiency) - β × (latency_penalty)²`

---

## Dependencies

All standard — no additional packages beyond what's in `requirements.txt`:

- `torch` (PyTorch)
- `gymnasium`
- `numpy`
- `matplotlib`
- `tqdm`
