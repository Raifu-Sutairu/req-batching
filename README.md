# Dynamic Request Batching — Reinforcement Learning Agent

A project implementing a **Soft Actor-Critic (SAC)** agent enhanced with **LSTM** (for temporal sequence modeling) and **Prioritized Experience Replay (PER)**. The agent learns *when* to serve accumulated cache requests. The agent observes various state signals and chooses every 10 ms whether to dispatch the current queue as a batch or wait for more requests to arrive — balancing throughput efficiency against client latency.

---

## Installation

```bash
pip install gymnasium scipy matplotlib numpy torch
```

> **Python ≥ 3.12** is recommended.

---

## Quick Start

### Run the full pipeline end-to-end
```bash
python3 run_all.py
```
This script orchestrates the environment validation, trains the SAC+LSTM+PER agent across different traffic patterns, and evaluates the agent against internal heuristic baselines.

---

## MDP Formulation

The batching problem is cast as a **Markov Decision Process**:

### State (8-dimensional continuous)

| Index | Feature |
|---|---|
| 0 | `pending_requests` |
| 1 | `sla_urgency` (derived from oldest wait) |
| 2 | `request_rate` |
| 3 | `delta_rate` (change in rate) |
| 4 | `since_serve_ms` |
| 5 | `batch_fill_ratio` |
| 6 | `time_of_day_sin` |
| 7 | `time_of_day_cos` |

### Action Space
`Discrete(4)` — **0 = Serve Now**, **1 = Wait 20ms**, **2 = Wait 50ms**, **3 = Wait 100ms**

### Reward Function
```
r = α · batch_size  −  β · oldest_wait_ms
```

### Why SAC + LSTM + PER over PPO?
While PPO is a strong baseline, dynamic web traffic (e.g., bursty or time-varying) contains partial observability and rapid distribution shifts. 
- **LSTM**: Captures temporal trends (like rising traffic spikes) across the last `N` steps, solving the partial observability problem.
- **SAC**: An off-policy algorithm that maximizes both expected reward and policy entropy, preventing premature convergence and exploring wait/serve trade-offs more smoothly.
- **PER**: Prioritized Experience Replay ensures the network learns rapidly from rare, high-mistake states (like sudden SLA violations) instead of uniformly sampling boring "Wait" decisions.

---

## Project Structure

```
Rl project/
├── config.py                  # All hyperparameters for SAC and environments
├── run_all.py                 # End-to-end pipeline orchestrator
├── plot_results.py            # Generates data visualizations
│
├── env/
│   ├── batching_env.py        # Base Gymnasium BatchingEnv 
│   └── traffic_generator.py   # Poisson, bursty, and time-varying traffic generators
│
├── sac_agent/
│   ├── sac_agent.py           # Core SAC algorithm implementation
│   ├── network.py             # Actor and Critic networks (with LSTM)
│   ├── replay_buffer.py       # Prioritized Experience Replay (PER) buffer
│   ├── extended_env.py        # 8D state wrapper for complex traffic
│   ├── train_sac.py           # Training loop
│   └── evaluate_sac.py        # Evaluation script vs baseline heuristics
│
├── results/                   # Evaluation plots and JSON summaries
├── checkpoints_sac/           # Saved PyTorch models
└── logs_sac/                  # Training metrics logic
```
