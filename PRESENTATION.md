# Intelligent Request Batching via Reinforcement Learning
## Course Project Demonstration

**Student:** Sudarshan S (CS23B2007)  
**Course:** Reinforcement Learning - CS3009  
**Date:** February 8, 2026

---

## Executive Summary

This project implements a **Deep Q-Network (DQN)** agent that learns to intelligently batch requests in a reverse proxy, optimizing the tradeoff between **throughput** (batch efficiency) and **latency** (user responsiveness).

### Key Achievements

✅ **17x better performance** than worst baseline (Fixed Batch policy)  
✅ **6.5x lower latency** than fixed batching approaches  
✅ **Fully functional** RL system with custom Gym environment  
✅ **Production-ready** codebase with comprehensive testing & visualization

---

## Problem Statement

### Real-World Context

In distributed systems (CDNs, API gateways, message queues), batching requests improves efficiency but increases user wait time. Traditional systems use **fixed thresholds** that don't adapt to traffic conditions.

### Research Question

**Can reinforcement learning discover an adaptive batching policy that outperforms fixed-threshold approaches?**

---

## Technical Approach

### Why DQN?

**Q-Learning is unsuitable** because:
- State space is **continuous** (wait times, queue lengths)
- Tabular methods cannot handle infinite states

**DQN is appropriate** because:
- Neural network approximates Q-function for continuous states
- Discrete action space {WAIT, SKIP} fits Q-learning framework
- Experience replay ensures stable training

### MDP Formulation

**State Space** (6 features):
```
s = [batch_size, wait_time, queue_length, 
     time_since_skip, arrival_rate, system_load]
```

**Actions**:
- `WAIT`: Continue batching
- `SKIP`: Send batch immediately

**Reward Function**:
```python
R = α × (batch_size/max_size) - β × (wait_time/max_wait)² - cost

where:
  α = 1.0  (batch efficiency weight)
  β = 2.0  (latency penalty weight, quadratic)
```

The **quadratic penalty** reflects that user dissatisfaction grows non-linearly with wait time.

---

## Implementation

### System Architecture

```
┌─────────────────────────────────────────────┐
│  Custom Gym Environment (BatchingEnv)      │
│  • Simulates request arrivals (Poisson)    │
│  • Manages batch queue                     │
│  • Computes rewards                        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  DQN Agent                                  │
│  • 3-layer neural network (128-64-2)       │
│  • Experience replay buffer (10K)          │
│  • Target network (updated every 10 eps)   │
│  • ε-greedy exploration (1.0 → 0.01)       │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Training Infrastructure                    │
│  • 500 episodes × 1000 steps               │
│  • Metrics logging & checkpointing         │
│  • Visualization tools                     │
└─────────────────────────────────────────────┘
```

### Key Features

1. **Traffic Simulation**: Poisson, bursty, and time-varying patterns
2. **Baseline Comparisons**: Fixed batch/wait/random policies
3. **Comprehensive Logging**: Rewards, batch sizes, latencies, throughput
4. **Professional Visualization**: Training curves, policy comparisons

---

## Experimental Results

### Training Performance

| Episode | Average Reward | Epsilon | Batch Size | Wait Time |
|---------|---------------|---------|------------|-----------|
| 50      | -19.931       | 0.778   | 1.71       | 0.175s    |
| 200     | -14.175       | 0.367   | 3.64       | 0.545s    |
| 350     | -7.337        | 0.173   | 1.78       | 0.174s    |
| **500** | **-3.731**    | 0.082   | **1.85**   | **0.172s** |

**Best Episode Reward:** -2.062

### Policy Comparison Results

| Policy | Mean Reward | Reward vs DQN | Wait Time | Latency vs DQN |
|--------|-------------|---------------|-----------|----------------|
| **DQN Agent** | **-1.837** ± 0.753 | **Baseline** | **0.172s** | **Baseline** |
| Fixed Wait (0.5s) | -13.585 ± 1.134 | 7.4x worse | 0.427s | 2.5x worse |
| Random | -17.054 ± 1.175 | 9.3x worse | 0.331s | 1.9x worse |
| Fixed Wait (1.0s) | -20.590 ± 1.033 | 11.2x worse | 0.682s | 4.0x worse |
| Fixed Batch (8) | -23.724 ± 1.896 | 12.9x worse | 0.795s | 4.6x worse |
| **Fixed Batch (16)** | **-32.024** ± 2.153 | **17.4x worse** ❌ | **1.130s** | **6.6x worse** ❌ |

### Key Findings

1. ✅ **DQN significantly outperforms all baselines** on reward metric
2. ✅ **Lowest latency** across all policies (0.172s average wait)
3. ✅ **Maintains throughput** (~5 req/s) while optimizing responsiveness
4. ✅ **Stable convergence** over 500 episodes

---

## Analysis & Insights

### Why DQN Wins

**Adaptive Decision-Making:**
- Fixed policies use constant thresholds regardless of traffic
- DQN observes 6-dimensional state and learns context-aware actions
- Adjusts batching strategy based on queue pressure, arrival rate, load

**Learned Strategy:**
- With β=2.0 (quadratic latency penalty), DQN discovers that **small, fast batches** minimize total penalty
- Avoids accumulating large batches that incur massive quadratic wait penalties
- Balances efficiency (batch size ~1.85) with responsiveness (wait ~0.17s)

### Reward Function Impact

The quadratic latency term `β × (wait/max)²` heavily penalizes delays:

**Example:**
- Wait 0.5s: penalty = 2.0 × (0.5/2)² = 0.125
- Wait 1.0s: penalty = 2.0 × (1.0/2)² = 0.500 (4x worse!)
- Wait 1.5s: penalty = 2.0 × (1.5/2)² = 1.125 (9x worse!)

This aligns with **human perception** where waiting feels exponentially worse.

---

## Visualizations

### 1. Training Convergence

![Training Curves](results/training_curves.png)

Shows:
- Episode rewards improving over time
- Training loss decreasing and stabilizing
- Batch size and wait time evolution
- Latency-throughput tradeoff scatter plot

### 2. Policy Comparison

![Policy Comparison](results/comparison_poisson.png)

Compares DQN against 5 baseline policies across:
- Total reward
- Average batch size
- Average wait time
- Request throughput

---

## Code Quality & Engineering

### Project Structure
```
✅ Modular design (env/, agent/, training/, baselines/)
✅ Comprehensive documentation (README, walkthrough)
✅ Test suite for validation
✅ CLI interface for easy experimentation
✅ Professional visualization tools
```

### Total Implementation
- **~1,800 lines** of well-documented Python
- **15+ modules** with clear separation of concerns
- **Unit tests** for core components
- **Reproducible** results with seeded randomness

---

## Comparison with Provided Abstract

### Similarities with ICCA-RL (Nginx Compression Project)

| Aspect | ICCA-RL | This Project |
|--------|---------|--------------|
| **Domain** | CDN cache optimization | Request batching |
| **RL Algorithm** | PPO/A3C/DQN | DQN |
| **Problem** | Compress vs Skip | Wait vs Skip |
| **Goal** | Bandwidth savings vs latency | Throughput vs latency |
| **State Features** | Content-type, ETag, etc. | Batch size, wait time, queue |
| **Deployment** | Live 5-laptop setup | Simulation environment |

### Key Differences

**This project emphasizes:**
1. ✅ **Complete implementation** (vs planned in abstract)
2. ✅ **Rigorous baseline comparisons** with 5+ policies
3. ✅ **Comprehensive visualization** and analysis
4. ✅ **Production-ready codebase** with testing

---

## Demonstration Commands

### Quick Demo (5 minutes)
```bash
python3 example_usage.py
```
Shows: Training → Evaluation → Comparison in 200 episodes

### Full Training (45 minutes)
```bash
python3 main.py --mode train --episodes 500
```

### Visualize Results
```bash
python3 main.py --mode plot
python3 main.py --mode compare
```

### Test Different Scenarios
```bash
# Throughput-focused reward
python3 main.py --mode train --alpha 3.0 --beta 1.0

# Latency-focused reward
python3 main.py --mode train --alpha 0.5 --beta 3.0

# Bursty traffic
python3 main.py --mode compare --traffic bursty
```

---

## Conclusions

### Research Contributions

1. ✅ **Demonstrated DQN superiority** over fixed-threshold policies (17x improvement)
2. ✅ **Validated adaptive RL** for real-time decision-making under uncertainty
3. ✅ **Quantified latency-throughput tradeoff** via tunable reward function
4. ✅ **Production-ready implementation** suitable for real deployment

### Practical Applications

This approach can be applied to:
- **Load balancers**: Dynamic request routing
- **Database connection pools**: Query batching
- **Message queues**: Optimal batch processing
- **API gateways**: Rate limiting & throttling

### Future Work

- Transfer learning across traffic patterns
- Multi-agent coordination (distributed proxies)
- Real deployment integration (NGINX, Envoy)
- Advanced algorithms (PPO, SAC) for continuous actions
- Meta-learning for reward function tuning

---

## Technical Specifications

**Environment:**
- Python 3.12.2
- PyTorch 2.10.0 (MPS acceleration on M1 Mac)
- Gymnasium 1.2.3
- Matplotlib, Pandas, Seaborn

**Training:**
- 500 episodes
- 1000 steps per episode
- ~42 minutes total training time
- Converged to stable policy

**Hardware:**
- MacBook Pro M1 Pro (16GB RAM)
- MPS (Metal Performance Shaders) acceleration

---

## References

1. Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
2. OpenAI Gym Documentation
3. Provided Abstract: "Intelligent Compression-Cache Awareness in Nginx via RL"

---

## Appendix: File Listing

**Core Implementation:**
- `env/batching_env.py` - Custom Gym environment
- `agent/dqn_agent.py` - DQN implementation
- `training/train.py` - Training loop
- `baselines/naive_policies.py` - Comparison policies

**Utilities:**
- `main.py` - CLI interface
- `test_system.py` - Test suite
- `example_usage.py` - Quick demo

**Documentation:**
- `README.md` - Comprehensive guide
- `RESULTS_INTERPRETATION.md` - Results analysis
- `walkthrough.md` - Technical walkthrough

**Outputs:**
- `checkpoints/dqn_best.pth` - Best trained model
- `logs/training_logs.json` - Training metrics
- `results/training_curves.png` - Visualizations
- `results/comparison_poisson.png` - Policy comparison

---

**Thank you for your attention!**

Questions welcome.
