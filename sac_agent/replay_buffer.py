"""
replay_buffer.py  (v2 — all buffer problems fixed)
---------------------------------------------------
Fixes applied vs v1:
  [P19] Beta annealing corrected — old code annealed over 50k *outer* steps.
        With W100 (11 inner steps per outer step), that's only ~91 episodes.
        New default anneals over 200k frames — roughly 364 episodes at W100.
        This gives the IS correction time to actually kick in.
  [P20] Larger buffer — capacity increased from 50k to 200k transitions.
        At W100 with ~545 outer steps/episode, the old 50k buffer filled in
        ~91 episodes, evicting valuable early burst-onset transitions.
        200k holds ~364 episodes of data, covering multiple full cycles of
        bursty traffic peaks and off-peak periods.
  [FIX] State dim updated to 8 to match expanded state vector.
  [FIX] Sequence sampling now guards against cross-episode contamination
        (don't sample sequences that span a done=True boundary).
"""

import numpy as np
from collections import deque


# ── SumTree ──────────────────────────────────────────────────────────────────

class SumTree:
    """
    Binary tree for O(log N) priority sampling and updates.
    Leaves = individual transition priorities.
    Internal nodes = sum of subtree.
    Root = total priority sum.
    """

    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity)
        self.data      = np.empty(capacity, dtype=object)
        self.write_ptr = 0
        self.size      = 0

    @property
    def total(self):
        return self.tree[1]

    def _propagate(self, leaf_idx, change):
        parent = leaf_idx // 2
        while parent >= 1:
            self.tree[parent] += change
            parent //= 2

    def update(self, leaf_idx, priority):
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, change)

    def add(self, priority, data):
        leaf_idx = self.write_ptr + self.capacity
        self.data[self.write_ptr] = data
        self.update(leaf_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, target_value):
        idx = 1
        while idx < self.capacity:
            left  = 2 * idx
            right = 2 * idx + 1
            if target_value <= self.tree[left]:
                idx = left
            else:
                target_value -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity
        return idx, data_idx, self.tree[idx], self.data[data_idx]


# ── PER Buffer ────────────────────────────────────────────────────────────────

class PERBuffer:
    """
    Prioritized Experience Replay for LSTM-SAC.

    Key changes vs v1:
      - capacity: 50k → 200k  [P20]
      - beta_frames: 50k → 200k  [P19]
      - state_dim: 6 → 8 to match expanded state vector
      - Episode boundary tracking — prevents sequences from spanning
        done=True transitions (which would contaminate LSTM training
        with stale episode state leaking across resets)

    Args:
        capacity:    max transitions to store (default 200k)
        seq_len:     LSTM sequence length (default 30)
        state_dim:   state feature dimension (default 8)
        alpha:       priority exponent (0=uniform, 1=full)
        beta_start:  initial IS weight exponent
        beta_frames: training steps to anneal beta 0.4 → 1.0  [P19]
    """

    def __init__(self, capacity=200_000, seq_len=30, state_dim=8,
                 alpha=0.6, beta_start=0.4, beta_frames=200_000):
        self.capacity    = capacity
        self.seq_len     = seq_len
        self.state_dim   = state_dim
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames   # [P19] was 50k, now 200k
        self.frame       = 1

        self.tree        = SumTree(capacity)
        self.max_priority = 1.0

        # Flat transition arrays
        self._states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions     = np.zeros((capacity,),           dtype=np.int64)
        self._rewards     = np.zeros((capacity,),           dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones       = np.zeros((capacity,),           dtype=np.float32)

        self.write_ptr = 0
        self.size      = 0

    @property
    def beta(self):
        """Beta anneals linearly from beta_start → 1.0 over beta_frames."""
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def add(self, state, action, reward, next_state, done):
        """Store one transition. New transitions get max priority."""
        idx = self.write_ptr

        self._states[idx]      = state
        self._actions[idx]     = action
        self._rewards[idx]     = reward
        self._next_states[idx] = next_state
        self._dones[idx]       = float(done)

        priority = self.max_priority ** self.alpha
        self.tree.add(priority, idx)

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size      = min(self.size + 1, self.capacity)

    def _find_valid_start(self, end_idx):
        """
        Find a valid sequence start index that doesn't cross a done=True.
        Searches back seq_len steps; if it hits a done boundary, truncates
        the sequence to start just after that boundary (zero-pads the rest).

        Returns (start_idx, pad_len) where pad_len leading steps are zeros.
        """
        start = end_idx - self.seq_len + 1

        if start < 0:
            return 0, self.seq_len  # full padding

        # Scan backwards for any done=True in [start, end_idx-1]
        # (done at end_idx itself is fine — that's the terminal transition)
        for i in range(end_idx - 1, max(start - 1, 0), -1):
            if self._dones[i] > 0.5:
                # Sequence would cross episode boundary at i
                # Start fresh from i+1
                clean_start = i + 1
                pad_len = clean_start - start
                return clean_start, pad_len

        return start, 0

    def sample(self, batch_size):
        """
        Sample a batch of sequences proportional to priority.

        Returns:
            batch:   dict of states/actions/rewards/next_states/dones
            weights: IS weights (batch_size,)
            indices: SumTree leaf indices for priority updates
        """
        assert self.size >= self.seq_len

        batch_indices = []
        tree_indices  = []
        priorities    = []

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)

            tree_idx, data_idx, priority, _ = self.tree.get(value)
            data_idx = max(data_idx, self.seq_len - 1)

            batch_indices.append(data_idx)
            tree_indices.append(tree_idx)
            priorities.append(priority)

        # Build sequence arrays
        state_seqs      = np.zeros((batch_size, self.seq_len, self.state_dim), dtype=np.float32)
        next_state_seqs = np.zeros((batch_size, self.seq_len, self.state_dim), dtype=np.float32)
        actions         = np.zeros((batch_size,), dtype=np.int64)
        rewards         = np.zeros((batch_size,), dtype=np.float32)
        dones           = np.zeros((batch_size,), dtype=np.float32)

        for i, idx in enumerate(batch_indices):
            start, pad_len = self._find_valid_start(idx)
            end = idx + 1

            seq_len_actual = end - start
            if seq_len_actual > 0:
                state_seqs[i,      pad_len:]  = self._states[start:end]
                next_state_seqs[i, pad_len:]  = self._next_states[start:end]

            actions[i] = self._actions[idx]
            rewards[i] = self._rewards[idx]
            dones[i]   = self._dones[idx]

        # IS weights
        probs   = np.array(priorities) / (self.tree.total + 1e-8)
        weights = (self.size * probs) ** (-self.beta)
        weights /= (weights.max() + 1e-8)

        self.frame += 1

        batch = {
            "states":      state_seqs,
            "actions":     actions,
            "rewards":     rewards,
            "next_states": next_state_seqs,
            "dones":       dones,
        }

        return batch, weights.astype(np.float32), tree_indices

    def update_priorities(self, tree_indices, td_errors):
        """Update leaf priorities after a training step."""
        for tree_idx, td_error in zip(tree_indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.tree.update(tree_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.size

    def ready(self, batch_size):
        return self.size >= max(batch_size, self.seq_len)