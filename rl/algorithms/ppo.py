import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

#hyperparams
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4
UPDATE_EPOCHS = 10
BATCH_SIZE = 64

class ActorCritic(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.shared(x)
        mean = torch.sigmoid(self.actor_mean(x))  #bound to [0,1], scale externally
        std = self.actor_log_std.exp().expand_as(mean)
        return mean, std, self.critic(x).squeeze(-1)

    def get_action(self, state):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(0, 1), log_prob, value

    def evaluate(self, state, action):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy


def compute_gae(rewards, values, dones, next_value):
    advantages = []
    gae = 0
    values = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


class PPO:
    def __init__(self, state_dim=3, action_dim=1, max_wait_us=10000):
        self.max_wait_us = max_wait_us
        self.net = ActorCritic(state_dim, action_dim)
        self._init_policy_with_cloudflare()
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.buffer = []

    def _init_policy_with_cloudflare(self):
        #warm-start: bias actor to serve when wait_time is high
        #actor_mean last layer: weight on wait_time feature (index 1) set positive
        with torch.no_grad():
            self.net.actor_mean.weight[0][1] = 2.0   #wait_time influence
            self.net.actor_mean.weight[0][0] = -1.0  #batch_size: larger batch = wait more
            self.net.actor_mean.weight[0][2] = -0.5  #arrival_rate: high rate = wait more
            self.net.actor_mean.bias[0] = -1.0

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.net.get_action(state_t)
        wait_us = action.item() * self.max_wait_us
        return wait_us, action.item(), log_prob.item(), value.item()

    def store(self, transition):
        self.buffer.append(transition)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return None

        states = torch.FloatTensor([t[0] for t in self.buffer])
        actions = torch.FloatTensor([t[1] for t in self.buffer])
        old_log_probs = torch.FloatTensor([t[2] for t in self.buffer])
        rewards = [t[3] for t in self.buffer]
        values = [t[4] for t in self.buffer]
        dones = [t[5] for t in self.buffer]

        with torch.no_grad():
            _, _, next_val = self.net.forward(states[-1].unsqueeze(0))
        advantages, returns = compute_gae(rewards, values, dones, next_val.item())

        advantages_t = torch.FloatTensor(advantages)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        returns_t = torch.FloatTensor(returns)

        total_loss = 0
        for _ in range(UPDATE_EPOCHS):
            idx = torch.randperm(len(self.buffer))
            for start in range(0, len(self.buffer), BATCH_SIZE):
                mb = idx[start:start+BATCH_SIZE]
                log_probs, values_pred, entropy = self.net.evaluate(states[mb], actions[mb].unsqueeze(-1))
                ratio = (log_probs - old_log_probs[mb]).exp()
                surr1 = ratio * advantages_t[mb]
                surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * advantages_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = VALUE_COEF * (returns_t[mb] - values_pred).pow(2).mean()
                entropy_loss = -ENTROPY_COEF * entropy.mean()
                loss = actor_loss + critic_loss + entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self.buffer.clear()
        return total_loss