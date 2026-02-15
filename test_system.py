"""
Simple Test Script

Tests core functionality of the RL batching system.
"""

import sys
import numpy as np


def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60)
    
    try:
        import gymnasium
        print("✅ gymnasium")
    except ImportError as e:
        print(f"❌ gymnasium: {e}")
        return False
    
    try:
        import torch
        print(f"✅ torch (device: {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'})")
    except ImportError as e:
        print(f"❌ torch: {e}")
        return False
    
    try:
        import matplotlib
        print("✅ matplotlib")
    except ImportError as e:
        print(f"❌ matplotlib: {e}")
        return False
    
    try:
        from env import BatchingEnv, TrafficGenerator
        print("✅ env module (BatchingEnv, TrafficGenerator)")
    except ImportError as e:
        print(f"❌ env module: {e}")
        return False
    
    try:
        from agent import DQNAgent, ReplayBuffer, DQN
        print("✅ agent module (DQNAgent, ReplayBuffer, DQN)")
    except ImportError as e:
        print(f"❌ agent module: {e}")
        return False
    
    try:
        from baselines import FixedBatchPolicy, FixedWaitPolicy
        print("✅ baselines module")
    except ImportError as e:
        print(f"❌ baselines module: {e}")
        return False
    
    return True


def test_environment():
    """Test environment creation and basic functionality."""
    print("\n" + "="*60)
    print("Testing Environment")
    print("="*60)
    
    try:
        from env import BatchingEnv
        
        # Create environment
        env = BatchingEnv(traffic_pattern='poisson', max_steps=50, seed=42)
        print("✅ Environment created")
        
        # Reset
        state, info = env.reset()
        assert state.shape == (6,), f"Expected state shape (6,), got {state.shape}"
        print(f"✅ Reset successful (state shape: {state.shape})")
        
        # Take steps
        total_reward = 0
        for _ in range(10):
            action = np.random.randint(0, 2)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        print(f"✅ 10 steps completed (total reward: {total_reward:.3f})")
        
        # Test metrics
        metrics = env.get_metrics()
        print(f"✅ Metrics: batch_size={metrics['avg_batch_size']:.2f}, "
              f"wait_time={metrics['avg_wait_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent():
    """Test DQN agent creation and training."""
    print("\n" + "="*60)
    print("Testing DQN Agent")
    print("="*60)
    
    try:
        from agent import DQNAgent
        import torch
        
        # Create agent
        agent = DQNAgent(state_dim=6, action_dim=2)
        print(f"✅ Agent created (device: {agent.device})")
        
        # Test action selection
        state = np.random.rand(6).astype(np.float32)
        action = agent.select_action(state, explore=True)
        assert action in [0, 1], f"Invalid action: {action}"
        print(f"✅ Action selection: {action}")
        
        # Store transitions
        for _ in range(100):
            s = np.random.rand(6).astype(np.float32)
            a = np.random.randint(0, 2)
            r = np.random.randn()
            ns = np.random.rand(6).astype(np.float32)
            agent.store_transition(s, a, r, ns, False)
        
        print(f"✅ Stored 100 transitions")
        
        # Training step
        loss = agent.train_step()
        print(f"✅ Training step (loss: {loss:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_traffic_patterns():
    """Test different traffic patterns."""
    print("\n" + "="*60)
    print("Testing Traffic Patterns")
    print("="*60)
    
    try:
        from env import TrafficGenerator
        
        patterns = ['poisson', 'bursty', 'time_varying']
        
        for pattern in patterns:
            gen = TrafficGenerator(pattern=pattern, base_rate=5.0, seed=42)
            gen.reset()
            
            arrivals = []
            for _ in range(100):
                num = gen.generate_arrivals(dt=0.1)
                arrivals.append(num)
            
            avg_arrivals = np.mean(arrivals)
            print(f"✅ {pattern:15s}: avg={avg_arrivals:.2f} arrivals/step")
        
        return True
        
    except Exception as e:
        print(f"❌ Traffic pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baselines():
    """Test baseline policies."""
    print("\n" + "="*60)
    print("Testing Baseline Policies")
    print("="*60)
    
    try:
        from baselines import FixedBatchPolicy, FixedWaitPolicy, RandomPolicy
        
        # Test policies
        state = np.random.rand(6).astype(np.float32)
        info = {'batch_size': 10, 'wait_time': 0.5, 'queue_length': 5}
        
        policies = {
            'FixedBatch(16)': FixedBatchPolicy(16),
            'FixedWait(1.0)': FixedWaitPolicy(1.0),
            'Random(0.3)': RandomPolicy(0.3, seed=42)
        }
        
        for name, policy in policies.items():
            action = policy.select_action(state, info)
            assert action in [0, 1], f"Invalid action from {name}"
            print(f"✅ {name:18s}: action={action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RL Batching System - Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    if results[-1][1]:  # Only continue if imports succeeded
        results.append(("Environment", test_environment()))
        results.append(("Agent", test_agent()))
        results.append(("Traffic Patterns", test_traffic_patterns()))
        results.append(("Baselines", test_baselines()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed!")
        print("="*60)
        print("\nYou can now:")
        print("  1. Run quick demo:    python3 example_usage.py")
        print("  2. Train full agent:  python3 main.py --mode train")
        print("  3. View README:       cat README.md")
        return 0
    else:
        print("❌ Some tests failed")
        print("="*60)
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
