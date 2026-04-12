"""
agent/discrete_sac.py

Discrete Soft Actor-Critic (SAC) for the Dynamic Request Batching problem.

Reference: Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings"
           arXiv:1910.07207

Key differences from continuous SAC
─────────────────────────────────────
  • Actor outputs a softmax distribution over discrete actions (not mean/log_std)
  • Critic outputs Q(s, a) for ALL actions simultaneously (no action input needed)
  • Entropy is computed analytically: H[π] = −Σ_a π(a|s) log π(a|s)
  • Soft value function: V(s) = Σ_a π(a|s) [min(Q1,Q2)(s,a) − α log π(a|s)]

Why SAC for this problem
─────────────────────────
  • Off-policy: learns from a replay buffer — far more sample-efficient than PPO
  • Maximum-entropy objective: naturally produces a stochastic dispatch policy
    that hedges at the Wait/Serve boundary (PPO's entropy bonus is a heuristic;
    SAC's is principled via temperature α)
  • Auto-tuned temperature: α is adjusted so the policy maintains a target
    entropy level — no manual entropy coefficient tuning required
  • Twin critics: avoids Q-value overestimation that destabilises DQN
  • Stateless at deployment: same predict() interface as PPO, no LSTM state
"""

from __future__ import annotations

import os
import sys
import math
import random
from collections import deque
from typing import Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Replay buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-capacity circular replay buffer storing (s, a, r, s', done) tuples."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: float) -> None:
        self.buffer.append((
            obs.astype(np.float32),
            int(action),
            float(reward),
            next_obs.astype(np.float32),
            float(done),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.stack(obs),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_obs),
            np.array(dones,   dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── Network architectures ──────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Stochastic policy: outputs probability distribution over discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (probs, log_probs) of shape (batch, n_actions)."""
        logits    = self.net(obs)
        probs     = F.softmax(logits,     dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


class TwinCritic(nn.Module):
    """Twin Q-networks: each outputs Q(s, a) for all actions simultaneously."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]):
        super().__init__()
        self.q1 = _mlp(obs_dim, hidden, n_actions)
        self.q2 = _mlp(obs_dim, hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs), self.q2(obs)

    def q_min(self, obs: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs)
        return torch.min(q1, q2)


# ── Discrete SAC ───────────────────────────────────────────────────────────────

class DiscreteSAC:
    """
    Discrete Soft Actor-Critic.

    Trains from a VecNormalize-wrapped DummyVecEnv (same interface as PPO).
    Observations received from the env are already normalised.

    Parameters
    ----------
    obs_dim   : int   – observation dimension (8 for this project)
    n_actions : int   – number of discrete actions (2: Wait / Serve)
    cfg       : dict  – SAC_CONFIG from config.py
    """

    def __init__(self, obs_dim: int, n_actions: int, cfg: dict):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.device    = torch.device("cpu")

        hidden = cfg.get("net_arch", [128, 128])

        # Networks
        self.actor  = Actor(obs_dim, n_actions, hidden).to(self.device)
        self.critic = TwinCritic(obs_dim, n_actions, hidden).to(self.device)
        self.critic_target = TwinCritic(obs_dim, n_actions, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        lr = cfg.get("learning_rate", 3e-4)
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Learnable temperature (log_α for numerical stability)
        self.log_alpha   = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt   = torch.optim.Adam([self.log_alpha], lr=lr)
        ratio            = cfg.get("target_entropy_ratio", 0.98)
        self.target_entropy = math.log(n_actions) * ratio  # H_target

        # Hyper-parameters
        self.gamma           = cfg.get("gamma",           0.99)
        self.tau             = cfg.get("tau",             0.005)
        self.batch_size      = cfg.get("batch_size",      256)
        self.learning_starts = cfg.get("learning_starts", 5_000)
        self.train_freq      = cfg.get("train_freq",      1)
        self.gradient_steps  = cfg.get("gradient_steps",  1)
        # Reward scale: divides raw rewards before buffering so Q-values stay
        # in a range where the entropy temperature α is numerically meaningful.
        # Without this, episode rewards in the ±200k range make α≈0.007 negligible.
        self.reward_scale    = cfg.get("reward_scale",    1.0)

        # Replay buffer
        self.buffer = ReplayBuffer(cfg.get("buffer_size", 300_000), obs_dim)
        self._n_updates = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Return action for a single (already-normalised) observation."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.actor(obs_t)
        if deterministic:
            return int(probs.argmax(dim=-1).item())
        return int(torch.multinomial(probs, 1).item())

    def learn(self, vec_env, total_timesteps: int, log_freq: int = 20_000) -> "DiscreteSAC":
        """
        Train on a VecNormalize-wrapped DummyVecEnv.

        The env returns normalised obs and handles auto-reset on episode end.
        """
        obs = vec_env.reset()               # shape (1, obs_dim)
        episode_reward = 0.0
        episode_count  = 0

        for t in range(1, total_timesteps + 1):
            # ── Sample action ─────────────────────────────────────────────
            if len(self.buffer) < self.learning_starts:
                action = [vec_env.action_space.sample()]
            else:
                action = [self.predict(obs[0], deterministic=False)]

            # ── Environment step ──────────────────────────────────────────
            next_obs, rewards, dones, infos = vec_env.step(action)

            # Store transition — scale reward so Q-values are numerically
            # comparable to the entropy temperature α.
            self.buffer.push(
                obs[0], action[0],
                rewards[0] * self.reward_scale,
                next_obs[0], float(dones[0])
            )

            episode_reward += rewards[0]
            obs = next_obs

            if dones[0]:
                episode_count += 1
                episode_reward = 0.0

            # ── Gradient update ───────────────────────────────────────────
            if (t >= self.learning_starts and t % self.train_freq == 0):
                for _ in range(self.gradient_steps):
                    self._update()

            # ── Logging ───────────────────────────────────────────────────
            if t % log_freq == 0:
                print(f"  [SAC] step={t:>8,}  "
                      f"alpha={self.alpha.item():.4f}  "
                      f"updates={self._n_updates:,}  "
                      f"buffer={len(self.buffer):,}")

        return self

    # ── Training internals ─────────────────────────────────────────────────────

    def _update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return

        obs_np, acts_np, rews_np, nobs_np, done_np = self.buffer.sample(self.batch_size)

        obs_t  = torch.FloatTensor(obs_np ).to(self.device)
        acts_t = torch.LongTensor( acts_np).to(self.device)
        rews_t = torch.FloatTensor(rews_np).unsqueeze(1).to(self.device)
        nobs_t = torch.FloatTensor(nobs_np).to(self.device)
        done_t = torch.FloatTensor(done_np).unsqueeze(1).to(self.device)

        # ── Critic update ──────────────────────────────────────────────────
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(nobs_t)
            # Soft value: V(s') = Σ_a π(a|s') [min_Q(s',a) − α log π(a|s')]
            q1_next, q2_next = self.critic_target(nobs_t)
            min_q_next       = torch.min(q1_next, q2_next)
            v_next           = (next_probs * (min_q_next - self.alpha * next_log_probs)
                               ).sum(dim=1, keepdim=True)
            target_q = rews_t + self.gamma * (1.0 - done_t) * v_next

        q1_all, q2_all = self.critic(obs_t)
        q1 = q1_all.gather(1, acts_t.unsqueeze(1))
        q2 = q2_all.gather(1, acts_t.unsqueeze(1))
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor update ───────────────────────────────────────────────────
        probs, log_probs = self.actor(obs_t)
        with torch.no_grad():
            min_q_curr = self.critic.q_min(obs_t)

        # Maximise soft Q: L_π = -V(s) = -Σ_a π(a|s)[min_Q(s,a) - α log π(a|s)]
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q_curr)
                     ).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Temperature update ─────────────────────────────────────────────
        # Maximise entropy to target_entropy: L_α = -α (H[π] - H_target)
        entropy    = -(probs * log_probs).sum(dim=1).mean().detach()
        alpha_loss = -self.log_alpha * (entropy - self.target_entropy)

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── Soft target update ─────────────────────────────────────────────
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(),
                             self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        # Clamp log_alpha: min prevents collapse, max=0 keeps α≤1.0
        # so entropy term (~α×0.35) stays ≤10% of Q-values (≈3–5 with reward_scale=1e-3)
        self.log_alpha.data.clamp_(min=-5.0, max=0.0)

        self._n_updates += 1

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save network weights and temperature to ``path.pt``."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "actor":     self.actor.state_dict(),
            "critic":    self.critic.state_dict(),
            "log_alpha": self.log_alpha.data,
            "obs_dim":   self.obs_dim,
            "n_actions": self.n_actions,
        }, path + ".pt")

    @classmethod
    def load(cls, path: str, obs_dim: int, n_actions: int, cfg: dict) -> "DiscreteSAC":
        """Load a previously saved model from ``path.pt``."""
        agent = cls(obs_dim, n_actions, cfg)
        state = torch.load(path + ".pt", map_location="cpu", weights_only=False)
        agent.actor.load_state_dict(state["actor"])
        agent.critic.load_state_dict(state["critic"])
        agent.log_alpha.data = state["log_alpha"]
        return agent
