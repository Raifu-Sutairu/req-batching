"""
sac_agent.py  (v2 — all SAC mechanics problems fixed)
------------------------------------------------------
Fixes applied vs v1:
  [P8]  Alpha curriculum — α starts high (1.0) for exploration, then
        anneals to a lower value (0.1) as training progresses. This gives
        the agent a proper exploration warmup phase before committing.
        target_entropy=None still disables auto-tuning; curriculum is
        controlled by the trainer via set_alpha().
  [P17] train_every corrected — old value of 4 was calibrated for binary
        env with 1 inner step per action. With W100 = 11 inner steps,
        the effective update frequency was 44 inner steps between updates.
        New default: train_every=1 (every outer step) so learning keeps
        pace with the longer action horizons.
  [P18] Warm-up diversity — warm_up_steps are now passed through the
        trainer's phase-balanced episode sampling, so the buffer is
        seeded with a mix of peak and off-peak transitions from the start.
  [FIX] state_dim and seq_len updated to match v2 defaults (8D, seq=30).
  [FIX] Buffer state_dim passed explicitly so PERBuffer uses 8 not 6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .network import LSTMActor, LSTMCritic, init_weights
from .replay_buffer import PERBuffer


class SACAgent:
    """
    SAC agent with LSTM networks, PER, and all v2 fixes.

    Args:
        state_dim:       number of state features (8 for v2 extended env)
        action_dim:      number of discrete actions (4)
        seq_len:         LSTM context window (30 = 300ms)
        lstm_hidden:     LSTM hidden units
        fc_hidden:       FC head hidden units
        lr:              Adam learning rate
        gamma:           discount factor
        tau:             polyak averaging coefficient
        alpha_init:      initial entropy temperature (high for exploration)
        alpha_final:     final entropy temperature after curriculum
        alpha_anneal_eps: episodes over which alpha anneals
        target_entropy:  if not None, enables auto-tuning
        buffer_capacity: PER buffer capacity (200k default)
        batch_size:      training minibatch size
        device:          'cpu'|'cuda'|'mps'|None (auto)
    """

    def __init__(
        self,
        state_dim=8,           # [FIX] was 6
        action_dim=4,
        seq_len=30,            # [FIX] was 10
        lstm_hidden=128,
        fc_hidden=128,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha_init=1.0,        # [P8] start high for exploration
        alpha_final=0.1,       # [P8] anneal to this value
        alpha_anneal_eps=100,  # [P8] episodes to complete anneal
        target_entropy=None,   # None = curriculum; float = auto-tune
        buffer_capacity=200_000,   # [P20]
        batch_size=64,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=200_000,   # [P19]
        device=None,
    ):
        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        print(f"[SAC] Using device: {self.device}")

        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.seq_len     = seq_len
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size

        # [P8] Alpha curriculum
        self.alpha_init      = alpha_init
        self.alpha_final     = alpha_final
        self.alpha_anneal_eps = alpha_anneal_eps
        self._use_auto_alpha  = target_entropy is not None
        if self._use_auto_alpha:
            self.target_entropy = float(target_entropy)
        else:
            self.target_entropy = -float(action_dim)  # kept for reference

        # Networks
        net_kwargs = dict(
            state_dim=state_dim, action_dim=action_dim,
            lstm_hidden=lstm_hidden, fc_hidden=fc_hidden, seq_len=seq_len
        )

        self.actor   = LSTMActor(**net_kwargs).to(self.device)
        self.critic1 = LSTMCritic(**net_kwargs).to(self.device)
        self.critic2 = LSTMCritic(**net_kwargs).to(self.device)

        self.critic1_target = LSTMCritic(**net_kwargs).to(self.device)
        self.critic2_target = LSTMCritic(**net_kwargs).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor.apply(init_weights)
        self.critic1.apply(init_weights)
        self.critic2.apply(init_weights)

        for p in self.critic1_target.parameters(): p.requires_grad = False
        for p in self.critic2_target.parameters(): p.requires_grad = False

        # Alpha: log-space for positivity guarantee
        self.log_alpha = torch.tensor(
            np.log(alpha_init), dtype=torch.float32,
            requires_grad=True, device=self.device
        )

        # Optimisers
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(),   lr=lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        self.alpha_opt   = torch.optim.Adam([self.log_alpha],          lr=lr)

        # PER Buffer — pass state_dim explicitly [FIX]
        self.buffer = PERBuffer(
            capacity=buffer_capacity,
            seq_len=seq_len,
            state_dim=state_dim,   # [FIX] was hardcoded 6
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
        )

        # Rolling state window for inference
        self._state_queue = np.zeros((seq_len, state_dim), dtype=np.float32)

        # Tracking
        self.train_steps   = 0
        self.episode_count = 0     # [P8] for alpha curriculum
        self.actor_losses  = []
        self.critic_losses = []
        self.alpha_values  = []
        self.entropy_values = []

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ── Alpha curriculum ──────────────────────────────────────────────────────

    def on_episode_end(self):
        """
        Call at the end of each training episode.
        [P8] Anneals alpha from alpha_init → alpha_final over alpha_anneal_eps.
        This gives a proper exploration warmup phase.
        """
        self.episode_count += 1
        if not self._use_auto_alpha:
            progress = min(1.0, self.episode_count / self.alpha_anneal_eps)
            new_alpha = self.alpha_init + progress * (self.alpha_final - self.alpha_init)
            with torch.no_grad():
                self.log_alpha.copy_(
                    torch.tensor(np.log(new_alpha), device=self.device)
                )

    # ── Interaction ───────────────────────────────────────────────────────────

    def reset_state_queue(self):
        self._state_queue = np.zeros((self.seq_len, self.state_dim), dtype=np.float32)

    def update_state_queue(self, state):
        self._state_queue = np.roll(self._state_queue, shift=-1, axis=0)
        self._state_queue[-1] = state

    def select_action(self, state, deterministic=False):
        self.update_state_queue(state)
        state_seq = torch.FloatTensor(self._state_queue).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor.get_action(state_seq, deterministic=deterministic)
        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    # ── Training ──────────────────────────────────────────────────────────────

    def train_step(self):
        """One full gradient update across all networks."""
        if not self.buffer.ready(self.batch_size):
            return None

        batch, weights, tree_indices = self.buffer.sample(self.batch_size)

        state_seqs = torch.FloatTensor(batch["states"]).to(self.device)
        actions    = torch.LongTensor(batch["actions"]).to(self.device)
        rewards    = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_seqs  = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones      = torch.FloatTensor(batch["dones"]).to(self.device)
        is_weights = torch.FloatTensor(weights).to(self.device)

        # ── Bellman targets ───────────────────────────────────────────
        with torch.no_grad():
            next_probs, next_log_probs, _ = self.actor(next_seqs)
            q1_next, _ = self.critic1_target(next_seqs)
            q2_next, _ = self.critic2_target(next_seqs)
            min_q_next = torch.min(q1_next, q2_next)
            v_next     = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1)
            q_target   = rewards + self.gamma * v_next * (1.0 - dones)

        # ── Critic losses ─────────────────────────────────────────────
        q1_all, _ = self.critic1(state_seqs)
        q2_all, _ = self.critic2(state_seqs)
        q1 = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        td_errors    = (q1.detach() - q_target).abs().cpu().numpy()
        critic1_loss = (is_weights * F.huber_loss(q1, q_target, reduction='none')).mean()
        critic2_loss = (is_weights * F.huber_loss(q2, q_target, reduction='none')).mean()

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_opt.step()

        # ── Actor loss ────────────────────────────────────────────────
        probs, log_probs, _ = self.actor(state_seqs)
        with torch.no_grad():
            q1_pi, _ = self.critic1(state_seqs)
            q2_pi, _ = self.critic2(state_seqs)
            min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q_pi)).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # ── Alpha loss (only when auto-tuning enabled) ────────────────
        if self._use_auto_alpha:
            entropy    = -(probs.detach() * log_probs.detach()).sum(dim=1).mean()
            alpha_loss = -(self.log_alpha * (entropy + self.target_entropy))
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # ── Soft target update ────────────────────────────────────────
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        # ── PER priority update ───────────────────────────────────────
        self.buffer.update_priorities(tree_indices, td_errors)

        # ── Logging ───────────────────────────────────────────────────
        self.train_steps += 1
        entropy_val = -(probs.detach() * log_probs.detach()).sum(dim=1).mean().item()
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append((critic1_loss.item() + critic2_loss.item()) / 2)
        self.alpha_values.append(self.alpha.item())
        self.entropy_values.append(entropy_val)

        return {
            "actor_loss":    actor_loss.item(),
            "critic_loss":   (critic1_loss.item() + critic2_loss.item()) / 2,
            "alpha":         self.alpha.item(),
            "entropy":       entropy_val,
            "q1_mean":       q1.mean().item(),
            "td_error_mean": td_errors.mean(),
        }

    def _soft_update(self, online_net, target_net):
        for op, tp in zip(online_net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, path):
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic1":        self.critic1.state_dict(),
            "critic2":        self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha":      self.log_alpha.detach().cpu(),
            "actor_opt":      self.actor_opt.state_dict(),
            "critic1_opt":    self.critic1_opt.state_dict(),
            "critic2_opt":    self.critic2_opt.state_dict(),
            "alpha_opt":      self.alpha_opt.state_dict(),
            "train_steps":    self.train_steps,
            "episode_count":  self.episode_count,
        }, path)
        print(f"[SAC] Saved checkpoint → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic1_opt.load_state_dict(ckpt["critic1_opt"])
        self.critic2_opt.load_state_dict(ckpt["critic2_opt"])
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        self.train_steps  = ckpt["train_steps"]
        self.episode_count = ckpt.get("episode_count", 0)
        print(f"[SAC] Loaded checkpoint ← {path}")