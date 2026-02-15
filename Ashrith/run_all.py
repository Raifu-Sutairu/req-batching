"""
Master Runner — Train All Agents & Generate Comparisons

Runs everything with one command:
    python -m Ashrith.run_all

This will:
1. Train REINFORCE for 300 episodes
2. Train A2C for 300 episodes
3. Train PPO for 300 episodes
4. Evaluate all agents + DQN + baselines
5. Generate comparison plots & training curves
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ashrith.train_agents import train
from Ashrith.compare_all import compare_all, plot_comparison, plot_training_curves


def main():
    NUM_EPISODES = 300
    TRAFFIC = 'poisson'
    SEED = 42
    
    print("\n" + "█"*60)
    print("  ASHRITH's RL IMPLEMENTATIONS")
    print("  Policy Gradient | Actor-Critic | PPO")
    print("  Request Batching Environment")
    print("█"*60)
    
    # ── Phase 1: Train All Agents ──
    print("\n\n" + "━"*60)
    print("  PHASE 1: TRAINING ALL AGENTS")
    print("━"*60)
    
    agents = {}
    
    for agent_type in ['reinforce', 'a2c', 'ppo']:
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
    
    results = compare_all(
        traffic_pattern=TRAFFIC,
        num_eval_episodes=20,
        seed=SEED
    )
    
    # ── Phase 3: Generate Plots ──
    print("\n\n" + "━"*60)
    print("  PHASE 3: GENERATING PLOTS")
    print("━"*60)
    
    os.makedirs(os.path.join('Ashrith', 'results'), exist_ok=True)
    
    if results:
        plot_comparison(results, os.path.join('Ashrith', 'results', 'comparison.png'))
    
    plot_training_curves(os.path.join('Ashrith', 'results', 'training_curves.png'))
    
    # ── Summary ──
    print("\n\n" + "█"*60)
    print("  ALL DONE!")
    print("█"*60)
    print("\n  Generated files:")
    print("  ├── Ashrith/checkpoints/   (trained models)")
    print("  ├── Ashrith/logs/          (training logs)")
    print("  └── Ashrith/results/       (comparison plots)")
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
