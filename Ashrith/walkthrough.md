# Walkthrough

## Main Layout

- `Ashrith/predictive_dynaq_agent.py`: primary paper-inspired method
- `Ashrith/live_simulation.py`: live demo dashboard
- `Ashrith/compare_all.py`: final offline evaluation across traffic patterns and seeds
- `Ashrith/env/`: batching environment and traffic generators
- `Ashrith/legacy/`: older REINFORCE, A2C, and PPO baselines kept for comparison only

## Final Evaluation Metrics

- mean reward
- average wait time
- p95 wait time
- average batch size
- throughput
- average queue length
- SLO violation rate

## Live Simulation

The live simulation is set up to emphasize the final-report metrics:

- average wait time
- p95 wait time
- average batch size
- predicted vs observed demand
- live counters for throughput, batches sent, requests served, SLO violations, and current action

## Why Predictive Dyna-Q

This is the cleanest fit for the current batching environment because it adapts grouped demand prediction and accelerated RL ideas from the cited caching paper without turning the project into a different cache-replacement problem.
