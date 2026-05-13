import sys
import time
import numpy as np
from pathlib import Path
import logging
import onnxruntime as ort

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from data.kafka_consumer import TelemetryConsumer
from environments.env_v1 import BatchFlushEnv
from config import config
from reward import compute_reward
from normalise import obs_to_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_baseline(episodes, policy_fn, name):
    total_upstream_calls = 0
    latencies = []
    batch_sizes = []
    forced_flushes = 0

    for ep in episodes:
        for idx, step in enumerate(ep.steps):
            obs = obs_to_tensor(
                step.batch_size, step.batch_age_ms, step.upstream_p99_ms, 
                step.request_rate, config.max_batch_size, config.batch_timeout_ms
            )
            action = policy_fn(obs, step.batch_size, step.batch_age_ms)
            
            is_last = (idx == len(ep.steps) - 1)
            
            if action == 1 or is_last:
                total_upstream_calls += 1
                latencies.append(step.batch_age_ms)
                batch_sizes.append(step.batch_size)
                if is_last and action == 0:
                    forced_flushes += 1
                break

    baseline_calls = sum(len(ep.steps) for ep in episodes)
    mean_batch_size = np.mean(batch_sizes) if batch_sizes else 1.0
    upstream_call_reduction = (1.0 - 1.0 / mean_batch_size) * 100
    
    return {
        "upstream_call_reduction": upstream_call_reduction,
        "p50_latency_ms": np.percentile(latencies, 50) if latencies else 0,
        "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
        "mean_batch_size": mean_batch_size,
        "forced_flush_rate": (forced_flushes / total_upstream_calls * 100) if total_upstream_calls else 0,
    }

def rl_policy(session):
    def policy(obs, batch_size, batch_age):
        logits = session.run(None, {"obs": obs.reshape(1, 4)})[0]
        return int(np.argmax(logits, axis=1)[0])
    return policy

def no_batch_policy(obs, batch_size, batch_age):
    return 1

def timer_policy(obs, batch_size, batch_age):
    return 1 if batch_age >= config.batch_timeout_ms else 0

def size_cap_policy(obs, batch_size, batch_age):
    return 1 if batch_size >= config.max_batch_size else 0

class CloudflarePolicy:
    def __init__(self, lam: float = 2.0, seed: int = 42):
        self.lam = lam
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, batch_size, batch_age):
        remaining_ratio = 1.0 - (batch_age / config.batch_timeout_ms)
        p_flush = np.exp(-self.lam * remaining_ratio)
        return 1 if self.rng.random() < p_flush else 0

def main():
    consumer = TelemetryConsumer(buffer_size=5000, group_id="rl-eval-group-2")
    logger.info("Gathering eval telemetry from Kafka... (waiting 5s)")
    start_time = time.time()
    while time.time() - start_time < 5.0:
        consumer.consume_batch()
    consumer.close()
    
    episodes = list(consumer.buffer)
    if not episodes:
        logger.error("No telemetry data for evaluation.")
        return
        
    logger.info(f"Evaluating policies on {len(episodes)} offline episodes...\n")
    
    def log_results(name, results_list):
        logger.info(f"--- {name} ---")
        metrics = ["upstream_call_reduction", "p50_latency_ms", "p99_latency_ms", "mean_batch_size", "forced_flush_rate"]
        for m in metrics:
            vals = [r[m] for r in results_list]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            if m in ["upstream_call_reduction", "forced_flush_rate"]:
                logger.info(f"{m:25s}: {mean_val:>6.2f}% ± {std_val:>5.2f}%")
            else:
                logger.info(f"{m:25s}: {mean_val:>6.2f}  ± {std_val:>5.2f}")
        logger.info("")

    # Deterministic baselines only need 1 run
    log_results("No Batching", [evaluate_baseline(episodes, no_batch_policy, "No Batching")])
    log_results("Fixed Timer (50ms)", [evaluate_baseline(episodes, timer_policy, "Fixed Timer (50ms)")])
    log_results("Fixed Size Cap (128)", [evaluate_baseline(episodes, size_cap_policy, "Fixed Size Cap (128)")])
    
    # Probabilistic baseline needs multiple runs
    cf_results = []
    for seed in [42, 1024]:
        cf_results.append(evaluate_baseline(episodes, CloudflarePolicy(lam=2.0, seed=seed), f"Cloudflare Exp-Prob (seed {seed})"))
    log_results("Cloudflare Exp-Prob", cf_results)
    
    try:
        session = ort.InferenceSession("ppo_batch_agent.onnx", providers=["CPUExecutionProvider"])
        # Deterministic policy, 1 run
        log_results("PPO ONNX Agent", [evaluate_baseline(episodes, rl_policy(session), "PPO ONNX Agent")])
    except Exception as e:
        logger.error(f"Could not load ONNX model for evaluation: {e}")

if __name__ == "__main__":
    main()
