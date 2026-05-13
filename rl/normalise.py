import numpy as np

def obs_to_tensor(
    batch_size: float,
    batch_age_ms: float,
    upstream_p99_ms: float,
    request_rate: float,
    max_batch_size: int,
    batch_timeout_ms: float
) -> np.ndarray:
    """
    Normalises the raw 4D state vector from the proxy into a stable range
    for the neural network. This function MUST be identically used in both
    training (env_v1.py) and serving (grpc_server.py).
    """
    return np.array([
        min(batch_size / max_batch_size, 1.0),
        min(batch_age_ms / batch_timeout_ms, 1.0),
        np.log1p(upstream_p99_ms) / np.log1p(500.0),
        np.tanh(request_rate / 100.0),
    ], dtype=np.float32)
