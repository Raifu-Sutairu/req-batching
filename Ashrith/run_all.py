"""
Master Runner for Ashrith's request batching experiments.

Runs legacy baselines plus the main Predictive Dyna-Q agent and then
builds the final evaluation table.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ashrith.train_agents import train
from Ashrith.compare_all import compare_all


def main():
    NUM_EPISODES = 300
    TRAFFIC = 'poisson'
    SEED = 42
    
    print("\n" + "█"*60)
    print("  ASHRITH's RL IMPLEMENTATIONS")
    print("  Predictive Dyna-Q + Legacy Baselines")
    print("  Request Batching Environment")
    print("█"*60)
    
    # ── Phase 1: Train All Agents ──
    print("\n\n" + "━"*60)
    print("  PHASE 1: TRAINING ALL AGENTS")
    print("━"*60)
    
    agents = {}
    
    for agent_type in ['reinforce', 'a2c', 'ppo', 'predictive_dynaq']:
        print(f"\n{'─'*40}")
        print(f"  Training {agent_type.upper()}...")
        print(f"{'─'*40}")
        agent, rewards, losses = train(
            agent_type=agent_type,
            num_episodes=NUM_EPISODES,
            traffic_pattern=TRAFFIC,
            seed=SEED
        )
        agents[agent_type] = (agent, rewards, losses)
    
    # ── Phase 2: Compare All ──
    print("\n\n" + "━"*60)
    print("  PHASE 2: COMPARING ALL AGENTS")
    print("━"*60)
    
    results = compare_all()
    
    # ── Summary ──
    print("\n\n" + "█"*60)
    print("  ALL DONE!")
    print("█"*60)
    print("\n  Generated files:")
    print("  ├── Ashrith/checkpoints/   (main model artifacts)")
    print("  ├── Ashrith/logs/          (main training logs)")
    print("  ├── Ashrith/legacy/        (older baseline agents/artifacts)")
    print("  └── Ashrith/results/       (final evaluation outputs)")
    print()
    
    # Print quick ranking
    if results:
        ranked = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
        print("  📊 Reward Ranking:")
        for i, (name, r) in enumerate(ranked, 1):
            marker = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
            print(f"  {marker} {i}. {name:>12}: {r['mean_reward']:.3f} ± {r['std_reward']:.3f}")
    print()


if __name__ == '__main__':
    main()
