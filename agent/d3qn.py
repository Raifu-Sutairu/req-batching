"""
agent/d3qn.py

D3QN: Dueling Double DQN with Prioritized Experience Replay + n-step returns.

Four-component composition:
──────────────────────────────────────────────────────────────────────────────
1. Dueling Q-network
     Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]
     Value stream: overall state quality.
     Advantage stream: relative benefit of each action.
     Key insight: for binary-action problems V(s) absorbs the gradient from
     "obvious" states, leaving A(s,a) to learn the fine-grained boundary.

2. Double DQN
     target = r + γ Q_target(s', argmax_a Q_online(s', a))
     Online net selects the action; target net evaluates it.
     Decoupling prevents the systematic overestimation that destabilises
     vanilla DQN on high-variance reward signals.

3. Prioritized Experience Replay (PER)
     P(i) ∝ |δ_i|^α,  importance-sampling weights w_i = (N·P(i))^{−β}
     β annealed 0.4 → 1.0 over training to correct the distributional bias.
     Key insight: SLA-violating transitions have large TD errors and get
     replayed ~40× more than in uniform replay.

4. n-step returns
     G_t = Σ_{k=0}^{n−1} γ^k r_{t+k} + γ^n Q(s_{t+n}, a*)
     Connects the wait→accumulate→serve sequence in one update step
     instead of bootstrapping through 3 chained 1-step TD estimates.
"""

from __future__ import annotations

import os
import sys
import math
import random
from collections import deque
from typing import Tuple, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sum Tree ───────────────────────────────────────────────────────────────────

class SumTree:
    """
    Binary sum-tree for O(log n) priority-proportional sampling.

    Leaf i stores priority p_i.  Internal nodes store the sum of their subtree.
    root = total priority.  Sampling a value s ∈ [0, total] descends the tree
    in O(log n) to find the leaf whose cumulative range covers s.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree  = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: List = [None] * capacity
        self._write = 0
        self._size  = 0

    # ── internal ───────────────────────────────────────────────────────────────

    def _propagate(self, leaf_idx: int, delta: float) -> None:
        idx = leaf_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def _retrieve(self, node: int, s: float) -> int:
        while True:
            left  = 2 * node + 1
            right = left + 1
            if left >= len(self.tree):
                return node
            if s <= self.tree[left]:
                node = left
            else:
                s   -= self.tree[left]
                node = right

    # ── public ─────────────────────────────────────────────────────────────────

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        leaf_idx = self._write + self.capacity - 1
        self.data[self._write] = data
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)
        self._write = (self._write + 1) % self.capacity
        self._size  = min(self._size + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, s: float) -> Tuple[int, float, object]:
        """Return (leaf_tree_idx, priority, data) for cumulative value s."""
        leaf_idx = self._retrieve(0, s)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), self.data[data_idx]

    def __len__(self) -> int:
        return self._size


# ── PER Buffer ─────────────────────────────────────────────────────────────────

class PERBuffer:
    """
    Prioritized Experience Replay buffer with n-step return pre-computation.

    Stored tuples: (obs, action, n_step_G, next_obs_n, done)
    where n_step_G = Σ_{k=0}^{n−1} γ^k r_{t+k}  (accumulated before storage).
    Episode-end transitions are flushed with shorter-n returns and done=True.
    """

    def __init__(self, capacity: int, obs_dim: int, n_step: int,
                 gamma: float, alpha: float, beta_start: float,
                 beta_end: float, per_eps: float):
        self.capacity   = capacity
        self.n          = n_step
        self.gamma      = gamma
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.per_eps    = per_eps

        self.tree      = SumTree(capacity)
        self._max_prio = 1.0
        self._nstep_buf: deque = deque()

    # ── n-step helpers ─────────────────────────────────────────────────────────

    def _nstep_return(self) -> Tuple:
        """Compute n-step return from the front of the deque."""
        obs0, act0, _, _, _ = self._nstep_buf[0]
        G, last_nobs, last_done = 0.0, None, False
        for k, (_, _, r, nobs, done) in enumerate(self._nstep_buf):
            G          += (self.gamma ** k) * r
            last_nobs   = nobs
            last_done   = done
            if done:          # episode ended inside window — stop accumulating
                break
        return (obs0.astype(np.float32), int(act0),
                float(G),
                last_nobs.astype(np.float32), float(last_done))

    # ── public API ─────────────────────────────────────────────────────────────

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: float) -> None:
        self._nstep_buf.append((obs, action, reward, next_obs, done))

        if len(self._nstep_buf) < self.n and not done:
            return   # not enough context yet

        # Pop oldest — its n-step return is now complete
        transition = self._nstep_return()
        self._nstep_buf.popleft()
        self.tree.add(self._max_prio ** self.alpha, transition)

        if done:
            # Flush remaining entries with shorter effective n
            while self._nstep_buf:
                transition = self._nstep_return()
                self._nstep_buf.popleft()
                self.tree.add(self._max_prio ** self.alpha, transition)

    def sample(self, batch_size: int,
               step: int, total_steps: int) -> Tuple[np.ndarray, ...]:
        """
        Stratified priority-proportional sampling.
        Returns (obs, acts, rewards, next_obs, dones, is_weights, leaf_indices).
        """
        beta = min(1.0, self.beta_start +
                   (self.beta_end - self.beta_start) * step / total_steps)

        segment = self.tree.total / batch_size
        leaf_indices = np.empty(batch_size, dtype=np.int32)
        priorities   = np.empty(batch_size, dtype=np.float32)

        obs_l, act_l, rew_l, nobs_l, done_l = [], [], [], [], []

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            leaf_idx, prio, data = self.tree.get(s)
            if data is None:           # guard for very early sampling
                s = self.tree.total * 0.5
                leaf_idx, prio, data = self.tree.get(s)
            leaf_indices[i] = leaf_idx
            priorities[i]   = max(float(prio), 1e-9)
            obs_l.append(data[0]); act_l.append(data[1])
            rew_l.append(data[2]); nobs_l.append(data[3]); done_l.append(data[4])

        # Importance-sampling weights, normalised so max weight = 1
        probs   = priorities / self.tree.total
        n       = len(self.tree)
        weights = (1.0 / (n * probs)) ** beta
        weights /= weights.max()

        return (
            np.stack(obs_l).astype(np.float32),
            np.array(act_l,  dtype=np.int64),
            np.array(rew_l,  dtype=np.float32),
            np.stack(nobs_l).astype(np.float32),
            np.array(done_l, dtype=np.float32),
            weights.astype(np.float32),
            leaf_indices,
        )

    def update_priorities(self, leaf_indices: np.ndarray,
                          td_errors: np.ndarray) -> None:
        prios = (np.abs(td_errors) + self.per_eps) ** self.alpha
        self._max_prio = max(self._max_prio, float(prios.max()))
        for idx, p in zip(leaf_indices.tolist(), prios.tolist()):
            self.tree.update(int(idx), float(p))

    def __len__(self) -> int:
        return len(self.tree)


# ── Dueling Q-network ──────────────────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """
    Dueling architecture for discrete actions.

    Shared MLP feature extractor splits into:
      - value stream  V(s)        → scalar
      - advantage stream A(s, a)  → n_actions values

    Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]
    Subtracting the mean identifiability-normalises A so
    V uniquely captures average state value.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.feature         = nn.Sequential(*layers)
        self.value_stream    = nn.Linear(in_dim, 1)
        self.advantage_stream = nn.Linear(in_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.feature(obs)
        V    = self.value_stream(feat)                        # (B, 1)
        A    = self.advantage_stream(feat)                    # (B, n_actions)
        return V + A - A.mean(dim=1, keepdim=True)           # (B, n_actions)


# ── D3QN Agent ─────────────────────────────────────────────────────────────────

class D3QN:
    """
    Dueling Double DQN + Prioritized Experience Replay + n-step returns.

    Compatible with VecNormalize-wrapped DummyVecEnv (same interface as SAC).
    Observations received from the env are already normalised.
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: dict):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.device    = torch.device("cpu")

        hidden = cfg.get("net_arch", [128, 128])

        # Networks
        self.online = DuelingQNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target = DuelingQNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

        lr = cfg.get("learning_rate", 3e-4)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

        # Hyper-parameters
        self.gamma           = cfg.get("gamma",                    0.99)
        self.tau             = cfg.get("tau",                      0.005)
        self.n_step          = cfg.get("n_step",                   3)
        self.batch_size      = cfg.get("batch_size",               256)
        self.learning_starts = cfg.get("learning_starts",          5_000)
        self.train_freq      = cfg.get("train_freq",               4)
        self.gradient_steps  = cfg.get("gradient_steps",           1)
        self.reward_scale    = cfg.get("reward_scale",             1e-3)
        self.eps_start       = cfg.get("exploration_initial_eps",  1.0)
        self.eps_end         = cfg.get("exploration_final_eps",    0.02)
        self.eps_fraction    = cfg.get("exploration_fraction",     0.15)

        # PER replay buffer
        self.buffer = PERBuffer(
            capacity   = cfg.get("buffer_size",    500_000),
            obs_dim    = obs_dim,
            n_step     = self.n_step,
            gamma      = self.gamma,
            alpha      = cfg.get("per_alpha",      0.6),
            beta_start = cfg.get("per_beta_start", 0.4),
            beta_end   = cfg.get("per_beta_end",   1.0),
            per_eps    = cfg.get("per_eps",        1e-6),
        )

        self._n_updates       = 0
        self._t               = 0
        self._total_timesteps = 1     # set in learn()

    # ── ε-greedy schedule ──────────────────────────────────────────────────────

    def _epsilon(self, t: int) -> float:
        frac = min(1.0, t / max(1, self.eps_fraction * self._total_timesteps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Return greedy (deterministic=True) or ε-greedy action."""
        if not deterministic and random.random() < self._epsilon(self._t):
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online(obs_t)
        return int(q.argmax(dim=-1).item())

    def learn(self, vec_env, total_timesteps: int,
              log_freq: int = 20_000) -> "D3QN":
        self._total_timesteps = total_timesteps
        obs = vec_env.reset()

        for t in range(1, total_timesteps + 1):
            self._t = t

            # ── Action ───────────────────────────────────────────────────────
            if len(self.buffer) < self.learning_starts:
                action = [vec_env.action_space.sample()]
            else:
                action = [self.predict(obs[0], deterministic=False)]

            # ── Step ─────────────────────────────────────────────────────────
            next_obs, rewards, dones, _ = vec_env.step(action)

            self.buffer.push(
                obs[0], action[0],
                float(rewards[0]) * self.reward_scale,
                next_obs[0], float(dones[0])
            )
            obs = next_obs

            # ── Gradient updates ──────────────────────────────────────────────
            if (len(self.buffer) >= self.learning_starts
                    and t % self.train_freq == 0):
                for _ in range(self.gradient_steps):
                    self._update(t)

            # ── Logging ──────────────────────────────────────────────────────
            if t % log_freq == 0:
                print(f"  [D3QN] step={t:>8,}  "
                      f"ε={self._epsilon(t):.4f}  "
                      f"updates={self._n_updates:,}  "
                      f"buffer={len(self.buffer):,}")

        return self

    # ── Training internals ─────────────────────────────────────────────────────

    def _update(self, t: int) -> None:
        if len(self.buffer) < self.batch_size:
            return

        obs_np, acts_np, rews_np, nobs_np, done_np, w_np, leaf_idx = \
            self.buffer.sample(self.batch_size, t, self._total_timesteps)

        obs_t  = torch.FloatTensor(obs_np ).to(self.device)
        acts_t = torch.LongTensor( acts_np).to(self.device)
        rews_t = torch.FloatTensor(rews_np).unsqueeze(1).to(self.device)
        nobs_t = torch.FloatTensor(nobs_np).to(self.device)
        done_t = torch.FloatTensor(done_np).unsqueeze(1).to(self.device)
        w_t    = torch.FloatTensor(w_np   ).unsqueeze(1).to(self.device)

        # ── Double DQN target ─────────────────────────────────────────────────
        with torch.no_grad():
            # Online net selects best action in next state
            best_a = self.online(nobs_t).argmax(dim=1, keepdim=True)
            # Target net evaluates that action (decoupling kills overestimation)
            q_next = self.target(nobs_t).gather(1, best_a)
            # rews_t already contains Σ γ^k r_{t+k}; bootstrap with γ^n
            target_q = rews_t + (self.gamma ** self.n_step) * (1.0 - done_t) * q_next

        # Current Q-value for the taken action
        q_curr = self.online(obs_t).gather(1, acts_t.unsqueeze(1))

        # Huber loss weighted by PER importance-sampling weights
        td_errors = (q_curr - target_q).detach().squeeze(1).cpu().numpy()
        loss = (w_t * F.huber_loss(q_curr, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities in the sum-tree
        self.buffer.update_priorities(leaf_idx, td_errors)

        # Soft target update: θ_target ← τ θ_online + (1−τ) θ_target
        with torch.no_grad():
            for p, tp in zip(self.online.parameters(),
                             self.target.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        self._n_updates += 1

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "online":    self.online.state_dict(),
            "obs_dim":   self.obs_dim,
            "n_actions": self.n_actions,
        }, path + ".pt")

    @classmethod
    def load(cls, path: str, obs_dim: int, n_actions: int, cfg: dict) -> "D3QN":
        agent = cls(obs_dim, n_actions, cfg)
        state = torch.load(path + ".pt", map_location="cpu", weights_only=False)
        agent.online.load_state_dict(state["online"])
        agent.target.load_state_dict(state["online"])  # target = online at eval
        return agent
