import torch
import logging
import sys
from pathlib import Path
from stable_baselines3 import PPO

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # Extract features (for MlpPolicy this is usually just flattening)
        features = self.policy.extract_features(observation)
        # Pass through the actor network layers
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        # Get logits from the action layer
        action_logits = self.policy.action_net(latent_pi)
        return action_logits

def export_to_onnx(model_path: str, output_path: str):
    logger.info(f"Loading PyTorch model from {model_path}...")
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Wrap the policy to only export the actor network
    onnxable_model = OnnxablePolicy(model.policy)
    onnxable_model.eval()

    # Create dummy observation [batch_size=1, features=4]
    dummy_input = torch.randn(1, 4)

    logger.info(f"Exporting to {output_path}...")
    torch.onnx.export(
        onnxable_model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["obs"],
        output_names=["action_logits"],
        dynamic_axes={"obs": {0: "batch_size"}, "action_logits": {0: "batch_size"}}
    )
    logger.info("Export successful!")

if __name__ == "__main__":
    export_to_onnx("ppo_batch_agent.zip", "ppo_batch_agent.onnx")
