import numpy as np

def compute_reward(
    action: int,            # 0=WAIT, 1=FLUSH
    batch_size: int,
    batch_age_ms: float,
    upstream_p99_ms: float,
    request_rate: float,
    timeout_ms: float,
    was_forced: bool,       # True if heuristic forced the flush
    alpha: float = 2.0,     # urgency steepness (exponential base)
    beta: float = 0.3,      # upstream load weight
    gamma: float = 0.5,     # forced-flush penalty weight
) -> float:
    """
    Computes the reward for a single timestep or terminal step of the batch flush episode.
    """
    # Normalised age in [0, 1]
    age_ratio = float(batch_age_ms) / float(timeout_ms)

    if action == 1:  # FLUSH
        # Throughput gain: log reward for batching (diminishing returns)
        throughput_gain = np.log2(1.0 + batch_size)

        # Urgency bonus: the closer to deadline, the more credit for flushing
        urgency_bonus = age_ratio ** 2

        # Upstream health penalty: if backend is already slow, flushing adds more load
        upstream_penalty = beta * (upstream_p99_ms / 50.0)

        # Forced flush penalty: if we get here because heuristic fired
        forced_penalty = gamma if was_forced else 0.0

        reward = throughput_gain + urgency_bonus - upstream_penalty - forced_penalty

    else:  # WAIT
        # Reward for waiting: positive if request_rate is high
        arrival_bonus = np.tanh(request_rate / 50.0)

        # Cost of waiting: exponential urgency
        urgency_cost = (np.exp(alpha * age_ratio) - 1.0) / (np.exp(alpha) - 1.0)

        reward = arrival_bonus - urgency_cost

    return float(reward)
