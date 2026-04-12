# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PPO reinforcement learning agent for **dynamic request batching** on an inference server.  Every 10 ms the agent decides whether to dispatch accumulated requests to a GPU backend or wait for a larger batch.  The core trade-off: small batches = low latency but poor GPU efficiency; large batches = good efficiency but SLA risk.

**Main model**: PPO (Proximal Policy Optimization)
**Comparison**: Cloudflare dual-threshold heuristic (the competitive production baseline)
**Floor**: GreedyBatch (always-serve, shows minimum efficiency)

## Commands

### Installation
```bash
pip install "stable-baselines3[extra]" gymnasium scipy matplotlib numpy
```

### Development workflow
```bash
python3 test_baselines.py          # Sanity check: env spec + baseline eval
python3 agent/train.py             # Train PPO (1M steps, ~20-40 min on CPU)
tensorboard --logdir tensorboard_logs/  # Monitor training at localhost:6006
python3 agent/evaluate.py          # Generate comparison.png + decision_heatmap.png
python3 agent/evaluate.py --model models/best/best_model  # Use best checkpoint
python3 agent/evaluate.py --skip-heatmap  # Skip heatmap (faster)
python3 deploy/middleware.py       # Smoke-test production deployment class
```

## Architecture

### Observation space (7-dim)
- `[0] pending_requests` — queue depth
- `[1] oldest_wait_ms` — head-of-queue age
- `[2] request_rate` — EMA of arrival rate (req/s)
- `[3] since_serve_ms` — time since last dispatch
- `[4] batch_fill_ratio` — pending / max_batch_size
- `[5] time_of_day` — fractional hour [0, 24)
- `[6] urgency_ratio` — `oldest_wait / (SLA - gpu_processing_time(batch_size))`. When >1.0 the current batch **will** violate SLA if served now. This is the critical 7th feature added in this revision.

### Reward structure
- **SERVE**: `alpha * batch_size − beta * oldest_wait − dispatch_cost − sla_penalty_per_ms * Σ(violation_ms)`
- **WAIT**: `−beta * oldest_wait − idle_penalty` (idle only if queue empty)
- `dispatch_cost = 5.0` forces the agent to accumulate ≥5 requests before serving is profitable
- SLA penalty is **proportional** to ms of violation (not binary), giving gradient signal for near-misses

### GPU processing model
`processing_time(n) = 8.0 + 0.12 * n  ms` — SLA is on TOTAL latency (wait + processing), not just wait time. This is realistic and makes urgency_ratio a non-trivial signal.

### Key components

**`env/batching_env.py`** — `BatchingEnv(gym.Env)`: 7-dim Box observation, Discrete(2) actions. `step()` generates Poisson arrivals → updates EMA → computes reward with GPU processing time. Tracks P50/P95/P99 latency in `_get_info()`.

**`baselines/cloudflare_formula.py`** — Two baselines:
- `CloudflareBaseline`: Deterministic dual-threshold (urgency OR efficiency). The competitive target — this is what a production engineer would actually deploy.
- `GreedyBatchBaseline`: Serve when `pending >= min_batch_size`. The floor.

**`agent/train.py`** — PPO with `VecNormalize` (critical for stable training across 7 features with different scales). 4 parallel envs, 1M steps, `[128, 128]` MLP. Saves `ppo_final.zip` + `ppo_vecnorm.pkl` (both required for deployment).

**`agent/evaluate.py`** — 3-agent comparison (30 episodes each). Produces 6-panel figure + decision heatmap. The heatmap overlays the learned PPO decision boundary against the Cloudflare threshold lines, showing the non-linear adaptation.

**`deploy/middleware.py`** — `BatchingMiddleware`: Production class. Loads model + VecNormalize stats. API: `record_arrivals(n)` → `should_dispatch()` → `flush()`. The `urgency_ratio` observation is computed from real wall-clock time.

**`config.py`** — Single source of truth: env params, GPU model, reward coefficients, PPO hyperparameters, training settings, experiment presets.

### VecNormalize — critical deployment note
`ppo_vecnorm.pkl` stores running mean/variance for all 7 observation dimensions. **Both `ppo_final.zip` and `ppo_vecnorm.pkl` must be regenerated together** when retraining. The middleware loads both at startup via `VecNormalize.load()`.

### Why Cloudflare baseline is competitive (not trivially beaten)
The original `exp(-λ·t)` formula gives `exp(-2000 * 0.5) ≈ 0` at 2000 req/s — it never dispatches. The fixed dual-threshold is what production systems actually implement: adaptive batch target (`rate * 0.050`) + urgency gate (80% of latency budget). PPO must genuinely outperform this by learning time-of-day patterns and EMA rate trajectories.

## Output artifacts
- `models/ppo_final.zip` — final PPO weights
- `models/ppo_vecnorm.pkl` — VecNormalize statistics (required for deploy)
- `models/best/best_model.zip` — best checkpoint by eval reward
- `results/comparison.png` — 6-panel figure
- `results/decision_heatmap.png` — P(Serve) policy heatmap with Cloudflare threshold overlay
- `tensorboard_logs/` — training curves
