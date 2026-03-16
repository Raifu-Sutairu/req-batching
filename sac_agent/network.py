"""
network.py  (v2 — all network problems fixed)
---------------------------------------------
Fixes applied vs v1:
  [P9]  LayerNorm after LSTM output — prevents hidden state explosion
        with unnormalised (or partially normalised) inputs. Speeds up
        convergence and improves stability significantly.
  [P10] Default seq_len changed 10 → 30 (300ms context window).
        100ms was too short to detect burst onsets that ramp over 300–500ms.
        With rate-delta in the state, 300ms window is the right balance.
  [P11] Deeper FC head: LSTM → FC(128) → FC(64) → out.
        Previous single FC layer was not enough capacity to learn the
        4-action Q-function boundary (serve vs wait20 vs wait50 vs wait100).
  [NEW] State dim updated to 8 to match expanded state vector in extended_env.
  [NEW] Dueling architecture option for critic — separates value V(s) from
        advantage A(s,a) for more stable Q-value estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMActor(nn.Module):
    """
    Actor: given state sequence → action probability distribution π(a|s).

    Architecture:
      Input (batch, seq_len, state_dim)
        → LSTM(lstm_hidden)
        → LayerNorm(lstm_hidden)        [P9]
        → FC(lstm_hidden → fc_hidden)
        → ReLU
        → FC(fc_hidden → fc_hidden//2)  [P11]
        → ReLU
        → FC(fc_hidden//2 → action_dim)
        → Softmax

    state_dim defaults to 8 to match extended state vector.
    """

    def __init__(self, state_dim=8, action_dim=4, lstm_hidden=128,
                 fc_hidden=128, seq_len=30):
        super().__init__()
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.lstm_hidden = lstm_hidden
        self.seq_len     = seq_len

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden,
            num_layers=2,           # 2 layers for richer temporal representation
            batch_first=True,
            dropout=0.1,            # light dropout between LSTM layers
        )

        # [P9] LayerNorm stabilises hidden state magnitudes
        self.layer_norm = nn.LayerNorm(lstm_hidden)

        # [P11] Deeper FC head — two hidden layers
        self.fc1 = nn.Linear(lstm_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.out = nn.Linear(fc_hidden // 2, action_dim)

    def forward(self, state_seq, hidden=None):
        """
        Args:
            state_seq: (batch, seq_len, state_dim)
            hidden:    optional LSTM hidden state

        Returns:
            probs:     (batch, action_dim) — action probabilities
            log_probs: (batch, action_dim) — log probabilities
            hidden:    updated LSTM hidden state
        """
        lstm_out, hidden = self.lstm(state_seq, hidden)
        last_out = lstm_out[:, -1, :]          # (batch, lstm_hidden)
        last_out = self.layer_norm(last_out)   # [P9]

        x = F.relu(self.fc1(last_out))
        x = F.relu(self.fc2(x))
        logits = self.out(x)

        probs     = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp(min=1e-8))

        return probs, log_probs, hidden

    def get_action(self, state_seq, hidden=None, deterministic=False):
        if state_seq.dim() == 2:
            state_seq = state_seq.unsqueeze(0)

        probs, log_probs, hidden = self.forward(state_seq, hidden)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()

        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action, action_log_prob, hidden


class LSTMCritic(nn.Module):
    """
    Critic: given state sequence → Q-values for all actions Q(s,a).

    Uses dueling architecture to separate state value V(s) from
    action advantage A(s,a):
        Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    This stabilises Q-value estimates — the agent learns "how good
    is this state" independently from "which action is relatively better",
    which converges faster than raw Q-value estimation.

    Architecture:
      Input → LSTM → LayerNorm → shared FC →
        ├── Value head:     FC → 1
        └── Advantage head: FC → action_dim
      Combined: Q = V + A - mean(A)
    """

    def __init__(self, state_dim=8, action_dim=4, lstm_hidden=128,
                 fc_hidden=128, seq_len=30):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        self.layer_norm = nn.LayerNorm(lstm_hidden)    # [P9]

        # Shared feature extractor
        self.fc_shared = nn.Linear(lstm_hidden, fc_hidden)

        # Dueling heads
        self.value_head  = nn.Sequential(
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden // 2, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden // 2, action_dim)
        )

    def forward(self, state_seq, hidden=None):
        """
        Args:
            state_seq: (batch, seq_len, state_dim)
            hidden:    optional LSTM hidden state

        Returns:
            q_values: (batch, action_dim)
            hidden:   updated LSTM hidden state
        """
        lstm_out, hidden = self.lstm(state_seq, hidden)
        last_out = lstm_out[:, -1, :]
        last_out = self.layer_norm(last_out)   # [P9]

        shared = F.relu(self.fc_shared(last_out))

        value  = self.value_head(shared)                        # (batch, 1)
        adv    = self.adv_head(shared)                          # (batch, action_dim)

        # Dueling combination: subtract mean advantage to reduce variance
        q_values = value + adv - adv.mean(dim=1, keepdim=True)  # (batch, action_dim)

        return q_values, hidden


def init_weights(module):
    """
    Orthogonal weight initialisation — standard for RL networks.
    Helps gradient flow and prevents early saturation.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=1.0)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)