import grpc
from concurrent import futures
import onnxruntime as ort
import numpy as np
import logging
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "proto_out"))

import rl_agent_pb2 as rl_agent_pb2
import rl_agent_pb2_grpc as rl_agent_pb2_grpc

from normalise import obs_to_tensor
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RlAgentServicer(rl_agent_pb2_grpc.RlAgentServicer):
    def __init__(self, onnx_path: str):
        if not Path(onnx_path).exists():
            logger.warning(f"ONNX model {onnx_path} not found! Falling back to ALWAYS-WAIT strategy.")
            self.session = None
        else:
            self.session = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"]
            )
            logger.info(f"Loaded ONNX model from {onnx_path}")

    def Decide(self, request, context):
        # Sanity check — 3x multiplier is purely a unit-mismatch detector (since the Rust hard-limit should enforce 1x)
        # Do not raise this limit! If it fires, Rust is likely sending microseconds instead of milliseconds.
        assert request.batch_age_ms <= config.batch_timeout_ms * 3, \
            f"batch_age_ms={request.batch_age_ms} far exceeds timeout, unit mismatch suspected"

        if self.session is None:
            # Fallback heuristic
            should_flush = request.batch_size >= config.max_batch_size or request.batch_age_ms >= config.batch_timeout_ms
            return rl_agent_pb2.FlushDecision(should_flush=should_flush)

        # Normalize identical to training
        obs = obs_to_tensor(
            batch_size=request.batch_size,
            batch_age_ms=request.batch_age_ms,
            upstream_p99_ms=request.upstream_p99_ms,
            request_rate=request.request_rate,
            max_batch_size=config.max_batch_size,
            batch_timeout_ms=config.batch_timeout_ms
        )
        # Reshape to [batch_size=1, features=4]
        obs = obs.reshape(1, 4)

        # Greedy inference via ONNX Runtime
        logits = self.session.run(None, {"obs": obs})[0]
        action = int(np.argmax(logits, axis=1)[0])

        return rl_agent_pb2.FlushDecision(should_flush=bool(action))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rl_agent_pb2_grpc.add_RlAgentServicer_to_server(
        RlAgentServicer("ppo_batch_agent.onnx"), server
    )
    port = "[::]:50051"
    server.add_insecure_port(port)
    server.start()
    logger.info(f"RL Agent ONNX gRPC Server started on {port}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
