import time
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from data.kafka_consumer import TelemetryConsumer
from environments.env_v1 import BatchFlushEnv
from algorithms.ppo import create_ppo_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Phase A: Offline Pre-training from Kafka")
    
    # 1. Consume telemetry to build offline replay buffer
    consumer = TelemetryConsumer(buffer_size=50000)
    logger.info("Polling Kafka for batch telemetry... (waiting 10 seconds)")
    
    start_time = time.time()
    total_added = 0
    while time.time() - start_time < 10.0:
        added = consumer.consume_batch(max_messages=5000, timeout_s=1.0)
        total_added += added
        
    consumer.close()
    logger.info(f"Gathered {total_added} episodes from Kafka.")
    
    if len(consumer.buffer) == 0:
        logger.error("No telemetry data found! Run the proxy and generate GET traffic first.")
        return
        
    # 2. Create Gymnasium environment
    env = BatchFlushEnv(list(consumer.buffer))
    
    # 3. Create and train PPO model
    logger.info("Initialising PPO model...")
    model = create_ppo_model(env)
    
    logger.info("Starting offline training for 100,000 timesteps...")
    # progress_bar=False avoids needing the 'rich' library
    model.learn(total_timesteps=100_000, progress_bar=False)
    
    # 4. Save PyTorch model
    model_path = "ppo_batch_agent.zip"
    model.save(model_path)
    logger.info(f"Saved trained PyTorch model to {model_path}")

    # 5. Export to ONNX in the same run
    logger.info("Exporting model to ONNX...")
    try:
        import torch

        class OnnxablePolicy(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            def forward(self, observation):
                features = self.policy.extract_features(observation)
                latent_pi = self.policy.mlp_extractor.forward_actor(features)
                return self.policy.action_net(latent_pi)

        onnx_model = OnnxablePolicy(model.policy)
        onnx_model.eval()
        dummy_input = torch.randn(1, 4)
        onnx_path = "ppo_batch_agent.onnx"
        torch.onnx.export(
            onnx_model,
            dummy_input,
            onnx_path,
            opset_version=17,
            input_names=["obs"],
            output_names=["action_logits"],
            dynamic_axes={"obs": {0: "batch_size"}, "action_logits": {0: "batch_size"}}
        )
        logger.info(f"ONNX model saved to {onnx_path}")
        logger.info("Training pipeline complete! Restart rl-agent to load the new model.")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        logger.info("The .zip model was saved — run 'python export_onnx.py' to export manually.")

if __name__ == "__main__":
    main()
