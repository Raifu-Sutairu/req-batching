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
    total_reward = 0.0
    total_latency = 0.0
    flushes = 0
    forced = 0
    
    for ep in episodes:
        for idx, step in enumerate(ep.steps):
            obs = obs_to_tensor(
                step.batch_size, step.batch_age_ms, step.upstream_p99_ms, 
                step.request_rate, config.max_batch_size, config.batch_timeout_ms
            )
            action = policy_fn(obs, step.batch_size, step.batch_age_ms)
            
            is_last = (idx == len(ep.steps) - 1)
            was_forced = False
            
            if action == 1 or is_last:
                if is_last and action == 0:
                    was_forced = True
                
                r = compute_reward(1, step.batch_size, step.batch_age_ms, 
                                   step.upstream_p99_ms, step.request_rate, 
                                   config.batch_timeout_ms, was_forced)
                total_reward += r
                total_latency += step.batch_age_ms
                flushes += 1
                if was_forced:
                    forced += 1
                break
            else:
                r = compute_reward(0, step.batch_size, step.batch_age_ms, 
                                   step.upstream_p99_ms, step.request_rate, 
                                   config.batch_timeout_ms, False)
                total_reward += r
                
    avg_reward = total_reward / len(episodes) if episodes else 0
    avg_latency = total_latency / flushes if flushes else 0
    forced_pct = (forced / flushes * 100) if flushes else 0
    
    logger.info(f"--- {name} ---")
    logger.info(f"Avg Reward per Episode : {avg_reward:.2f}")
    logger.info(f"Avg Batch Latency (ms) : {avg_latency:.2f}")
    logger.info(f"Forced Flush %         : {forced_pct:.1f}%")
    logger.info("")

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

def exponential_prob_policy(obs, batch_size, batch_age):
    alpha = 2.0
    age_ratio = batch_age / config.batch_timeout_ms
    prob = np.exp(alpha * age_ratio) / np.exp(alpha)
    return 1 if np.random.random() < prob else 0

def main():
    consumer = TelemetryConsumer(buffer_size=5000)
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
    
    evaluate_baseline(episodes, no_batch_policy, "No Batching")
    evaluate_baseline(episodes, timer_policy, "Fixed Timer (50ms)")
    evaluate_baseline(episodes, size_cap_policy, "Fixed Size Cap (128)")
    evaluate_baseline(episodes, exponential_prob_policy, "Cloudflare Exp-Prob")
    
    try:
        session = ort.InferenceSession("ppo_batch_agent.onnx", providers=["CPUExecutionProvider"])
        evaluate_baseline(episodes, rl_policy(session), "PPO ONNX Agent")
    except Exception as e:
        logger.error(f"Could not load ONNX model for evaluation: {e}")

if __name__ == "__main__":
    main()
