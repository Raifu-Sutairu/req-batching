# Policy-Based RL Agents for Request Batching
  
> **Part:** Model-Free RL — Policy Gradient, Actor-Critic, and PPO

This folder implements three policy-based reinforcement learning agents for the **dynamic request batching** problem. The RL agent learns *when* to serve accumulated requests (WAIT vs SKIP) to maximize batching efficiency while minimizing client-side latency.

---

## Agents Implemented

| Agent | Method | Key Idea |
|-------|--------|----------|
| **REINFORCE** | Policy Gradient | Collects full episodes, computes Monte Carlo returns, uses a learned value baseline for variance reduction |
| **A2C** | Advantage Actor-Critic | Shared actor-critic network with N-step TD updates; lower variance than REINFORCE via bootstrapping |
| **PPO** | Proximal Policy Optimization | Clipped surrogate objective + GAE (Generalized Advantage Estimation); multi-epoch mini-batch updates for best sample efficiency |

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
├── networks.py            # Neural network architectures
│   ├── PolicyNetwork        → state → action probabilities (Softmax)
│   ├── ValueNetwork         → state → V(s) scalar value
│   └── ActorCriticNetwork   → shared trunk with actor + critic heads
│
├── reinforce_agent.py     # REINFORCE with learned baseline
├── a2c_agent.py           # Advantage Actor-Critic (N-step)
├── ppo_agent.py           # PPO with GAE + clipped objective
│
├── train_agents.py        # Unified training script
├── compare_all.py         # Evaluation & comparison plots
├── run_all.py             # Master runner (train + compare + plot)
│
├── checkpoints/           # Saved model weights (.pth)
├── logs/                  # Training logs (JSON)
└── results/               # Comparison & training curve plots
```

---

## How to Run

```bash
# From the project root directory:

# Train all 3 agents, evaluate, and generate comparison plots
python3 -m Ashrith.run_all

# Train a specific agent
python3 -m Ashrith.train_agents --agent reinforce --episodes 300
python3 -m Ashrith.train_agents --agent a2c --episodes 300
python3 -m Ashrith.train_agents --agent ppo --episodes 300

# Run comparison only (requires trained checkpoints)
python3 -m Ashrith.compare_all
```

---

## Training Results

All agents trained for **300 episodes** on Poisson traffic (λ=5.0 req/s):

| Agent | Best Reward | Final Avg (last 50) | Training Time |
|-------|-------------|---------------------|---------------|
| REINFORCE | -7.75 | -9.44 | ~2 min |
| A2C | -1.44 | -4.72 | ~3 min |
| **PPO** | **-0.78** | **-2.33** | ~1 min |

---

## Evaluation Comparison

Evaluated over 20 episodes against DQN and baseline policies:

| Agent | Mean Reward | Batch Size | Wait Time |
|-------|-------------|------------|-----------|
| **A2C** | **-1.76 ± 0.58** | 1.8 | 0.172s |
| **PPO** | -1.82 ± 0.38 | 1.8 | 0.171s |
| DQN (existing) | -1.81 ± 0.55 | 1.8 | 0.172s |
| REINFORCE | -8.49 ± 1.08 | 2.7 | 0.278s |
| Random | -16.64 | 2.4 | 0.327s |
| Fixed Wait | -20.86 | 6.5 | 0.684s |
| Fixed Batch | -31.84 | 11.0 | 1.128s |

### Key Takeaways

- **A2C, PPO, and DQN** all converge to similar top-tier performance
- **PPO** has the **lowest variance** (±0.38) → most stable/reliable policy
- **REINFORCE** is weaker due to high variance Monte Carlo estimates, but still **2× better** than the Random baseline
- All RL agents **massively outperform** rule-based baselines (10–30× better reward)

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

The agents interact with `BatchingEnv` from `env/`:

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