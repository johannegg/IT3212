"""Tabular Q-learning and PPO (actor-critic MLP)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .env import N_ACTIONS, N_STATES, OBS_DIM


# --------------------------------------------------------------------------
# Q-learning
# --------------------------------------------------------------------------
class QLearning:
    """Off-policy TD control with an epsilon-greedy behavior policy."""

    def __init__(self, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05,
                 eps_decay_steps=150_000, seed=0):
        self.q = np.zeros((N_STATES, N_ACTIONS))
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.rng = np.random.default_rng(seed)

    def epsilon(self, env_steps: int) -> float:
        frac = min(1.0, env_steps / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def greedy(self, sid: int, mask) -> int:
        q = np.where(mask, self.q[sid], -np.inf)
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def act(self, sid: int, env_steps: int, mask) -> int:
        if self.rng.random() < self.epsilon(env_steps):
            return int(self.rng.choice(np.flatnonzero(mask)))
        return self.greedy(sid, mask)

    def update(self, sid, a, r, next_sid, terminal, next_mask):
        target = r if terminal else r + self.gamma * self.q[next_sid][next_mask].max()
        self.q[sid, a] += self.alpha * (target - self.q[sid, a])

    def action_values(self, sid):
        return self.q[sid].copy()

    def save(self, path):
        np.save(path, self.q)

    def load(self, path):
        self.q = np.load(path)


# --------------------------------------------------------------------------
# PPO
# --------------------------------------------------------------------------
def _mlp(out_dim, hidden, out_gain):
    layers = [nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out_dim)]
    for i, layer in enumerate(l for l in layers if isinstance(l, nn.Linear)):
        nn.init.orthogonal_(layer.weight, out_gain if i == 2 else np.sqrt(2))
        nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.actor = _mlp(N_ACTIONS, hidden, 0.01)
        self.critic = _mlp(1, hidden, 1.0)

    def dist(self, obs, mask=None):
        logits = self.actor(obs)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class PPO:
    """On-policy actor-critic with the clipped surrogate objective."""

    def __init__(self, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01,
                 vf_coef=0.5, epochs=4, rollout=2048, minibatch=256, hidden=64,
                 max_grad_norm=0.5, seed=0):
        torch.manual_seed(seed)
        self.net = ActorCritic(hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.gamma, self.lam, self.clip = gamma, gae_lambda, clip
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
        self.epochs, self.rollout, self.minibatch = epochs, rollout, minibatch
        self.gen = torch.Generator().manual_seed(seed)
        self.buf = {k: [] for k in ["obs", "mask", "act", "logp", "val", "rew", "done"]}

    @torch.no_grad()
    def act(self, obs, mask, mode="sampled"):
        o = torch.as_tensor(obs).unsqueeze(0)
        d = self.net.dist(o, torch.as_tensor(mask).unsqueeze(0))
        if mode == "sampled":
            a = torch.multinomial(d.probs, 1, generator=self.gen).squeeze()
        else:
            a = d.probs.argmax(-1).squeeze()
        return int(a), float(d.log_prob(a)), float(self.net.value(o))

    @torch.no_grad()
    def probs(self, obs, mask):
        d = self.net.dist(torch.as_tensor(obs).unsqueeze(0), torch.as_tensor(mask).unsqueeze(0))
        return d.probs.squeeze(0).numpy()

    @torch.no_grad()
    def value(self, obs):
        return float(self.net.value(torch.as_tensor(obs).unsqueeze(0)))

    def store(self, obs, mask, a, logp, val, r, done):
        for k, v in zip(self.buf, [obs, mask, a, logp, val, r, done]):
            self.buf[k].append(v)

    def ready(self):
        return len(self.buf["rew"]) >= self.rollout

    def update(self, last_obs, last_done):
        b = {k: np.array(v) for k, v in self.buf.items()}
        n = len(b["rew"])
        next_val = 0.0 if last_done else self.value(last_obs)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            nv = next_val if t == n - 1 else b["val"][t + 1]
            nonterm = 1.0 - b["done"][t]
            delta = b["rew"][t] + self.gamma * nv * nonterm - b["val"][t]
            gae = delta + self.gamma * self.lam * nonterm * gae
            adv[t] = gae
        ret = adv + b["val"]

        obs = torch.as_tensor(b["obs"], dtype=torch.float32)
        mask = torch.as_tensor(b["mask"])
        act = torch.as_tensor(b["act"])
        old_logp = torch.as_tensor(b["logp"], dtype=torch.float32)
        adv_t, ret_t = torch.as_tensor(adv), torch.as_tensor(ret, dtype=torch.float32)
        stats = []
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=self.gen)
            for i in range(0, n, self.minibatch):
                idx = perm[i:i + self.minibatch]
                d = self.net.dist(obs[idx], mask[idx])
                logp = d.log_prob(act[idx])
                ratio = torch.exp(logp - old_logp[idx])
                a = adv_t[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a).mean()
                vloss = ((self.net.value(obs[idx]) - ret_t[idx]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pg + self.vf_coef * vloss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
                stats.append((pg.item(), vloss.item(), ent.item(),
                              ((ratio.detach() - 1).abs() > self.clip).float().mean().item()))
        self.buf = {k: [] for k in self.buf}
        return dict(zip(["policy_loss", "value_loss", "entropy", "clip_frac"], np.mean(stats, 0)))

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
