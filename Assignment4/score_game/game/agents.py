"""Tabular Q-learning (one row per direction) and PPO (the same small network for every direction)."""
from __future__ import annotations

import pickle

import numpy as np
import torch
import torch.nn as nn

from .env import OBS_DIR, N_DIR, N_GLOB, q_keys, ppo_view


class QAgent:
    """Off-policy TD control, epsilon-greedy, with Watkins' Q(lambda). Q(s, a) is a table value of what she sees in direction a.

    hp: alpha, gamma, lam, eps_end, eps_frac, q0. Exploration falls from 1 to eps_end over eps_frac of the budget, then stays."""

    def __init__(self, hp, seed, budget):
        self.h, self.budget, self.rng = hp, budget, np.random.default_rng(seed)
        self.q, self.ids, self.e, self.steps = np.full(4096, float(hp["q0"])), {}, {}, 0

    def index(self, key):
        i = self.ids.get(key)
        if i is None:
            i = self.ids[key] = len(self.ids)
            if i >= len(self.q): self.q = np.concatenate([self.q, np.full(len(self.q), float(self.h["q0"]))])
        return i

    def state(self, game): return tuple(self.index(k) for k in q_keys(game))

    def epsilon(self):
        return 1.0 + min(1.0, self.steps / max(1, int(self.h["eps_frac"] * self.budget))) * (self.h["eps_end"] - 1.0)

    def greedy(self, s):
        v = self.q[list(s)]; b = np.flatnonzero(v == v.max())
        return int(b[0]) if len(b) == 1 else int(self.rng.choice(b))

    def act(self, s):
        """(action, whether it is the greedy one)"""
        if self.rng.random() < self.epsilon():
            a = int(self.rng.integers(4)); v = self.q[list(s)]
            return a, bool(v[a] == v.max())
        return self.greedy(s), True

    def update(self, s, a, r, s2, terminal, was_greedy):
        h = self.h; q = self.q
        d = (r if terminal else r + h["gamma"] * q[list(s2)].max()) - q[s[a]]
        if h["lam"] == 0: q[s[a]] += h["alpha"] * d; return
        if not was_greedy: self.e = {}              # Watkins: an exploring action cuts the trace
        self.e[s[a]] = 1.0; dead = []
        for k, v in self.e.items():
            q[k] += h["alpha"] * d * v; v *= h["gamma"] * h["lam"]
            if v < 0.01: dead.append(k)
            else: self.e[k] = v
        for k in dead: del self.e[k]
        if terminal: self.e = {}

    def end_game(self): self.e = {}

    def values(self, game): return self.q[list(self.state(game))]

    def save(self, path): pickle.dump((self.ids, self.q), open(path, "wb"))
    def load(self, path): self.ids, self.q = pickle.load(open(path, "rb"))


def mlp(n_in, hidden, n_out, gain):
    m = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out))
    for l in m:
        if isinstance(l, nn.Linear): nn.init.orthogonal_(l.weight, np.sqrt(2)); nn.init.zeros_(l.bias)
    nn.init.orthogonal_(m[-1].weight, gain)
    return m


class Net(nn.Module):
    """Actor: one network, used for each of the 4 directions (the same idea as one table row per direction). Critic: sees the
    four directions in her heading's order, and the situation as a whole."""

    def __init__(self, hidden=64, shared=True):
        super().__init__(); self.shared = shared
        self.actor = mlp(OBS_DIR, hidden, 1, 0.01) if shared else mlp(4 * N_DIR + N_GLOB, hidden, 4, 0.01)
        self.critic = mlp(4 * N_DIR + N_GLOB, hidden, 1, 1.0)

    @staticmethod
    def flat(o, order):
        """o: (B, 4, OBS_DIR), order: (B, 4) -> (B, 4*N_DIR+N_GLOB), directions in her heading's order."""
        d = torch.gather(o[:, :, :N_DIR], 1, order.unsqueeze(-1).expand(-1, -1, N_DIR)).flatten(1)
        return torch.cat([d, o[:, 0, N_DIR:]], 1)

    def logits(self, o, order):
        if self.shared: return self.actor(o).squeeze(-1)
        l = self.actor(self.flat(o, order)); return torch.zeros_like(l).scatter(1, order, l)       # one output per direction in heading order

    def dist(self, o, order): return torch.distributions.Categorical(logits=self.logits(o, order))
    def value(self, o, order): return self.critic(self.flat(o, order)).squeeze(-1)


class PPOAgent:
    """hp: lr, gamma, gae_lambda, ent_coef, epochs, reward_scale. Fixed: clip 0.2, value weight 0.5, rollout 2048, minibatch 256, hidden 64."""
    CLIP, VF, ROLLOUT, MINIBATCH, HIDDEN, MAXNORM = 0.2, 0.5, 2048, 256, 64, 0.5

    def __init__(self, hp, seed, shared=True):
        torch.manual_seed(seed); self.h = hp; self.net = Net(self.HIDDEN, shared)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=hp["lr"], eps=1e-5); self.gen = torch.Generator().manual_seed(seed)
        self.buf = {k: [] for k in ["o", "ord", "act", "logp", "val", "rew", "done", "boot"]}

    @torch.no_grad()
    def act(self, game, mode="sampled"):
        o, order = ppo_view(game); O = torch.from_numpy(o).unsqueeze(0); Q = torch.tensor([order])
        d = self.net.dist(O, Q)
        a = int(torch.multinomial(d.probs, 1, generator=self.gen)) if mode == "sampled" else int(d.probs.argmax(-1))
        return a, (o, order, float(d.log_prob(torch.tensor([a]))), float(self.net.value(O, Q)))

    @torch.no_grad()
    def probs(self, game):
        o, order = ppo_view(game); return self.net.dist(torch.from_numpy(o).unsqueeze(0), torch.tensor([order])).probs.squeeze(0).numpy()

    @torch.no_grad()
    def value_now(self, game):
        o, order = ppo_view(game); return float(self.net.value(torch.from_numpy(o).unsqueeze(0), torch.tensor([order])))

    def store(self, view, a, r, done, boot):
        """boot: value of the state after the last step of a game that was cut off without game over (0 = none)."""
        o, order, lp, v = view
        for k, x in zip(self.buf, [o, order, a, lp, v, r * self.h["reward_scale"], float(done), boot]): self.buf[k].append(x)

    def ready(self): return len(self.buf["rew"]) >= self.ROLLOUT

    def update(self, last_value):
        h = self.h; b = {k: np.array(v) for k, v in self.buf.items()}; n = len(b["rew"]); adv = np.zeros(n, np.float32); gae = 0.0
        for t in reversed(range(n)):
            nonterm = 1.0 - b["done"][t]
            nv = last_value if t == n - 1 else b["val"][t + 1]
            delta = b["rew"][t] + h["gamma"] * (nv * nonterm + b["boot"][t]) - b["val"][t]
            gae = delta + h["gamma"] * h["gae_lambda"] * nonterm * gae; adv[t] = gae
        ret = adv + b["val"]
        O = torch.as_tensor(b["o"], dtype=torch.float32); R = torch.as_tensor(b["ord"]); A = torch.as_tensor(b["act"]); L = torch.as_tensor(b["logp"], dtype=torch.float32)
        AD, RT = torch.as_tensor(adv), torch.as_tensor(ret, dtype=torch.float32); stats = []
        for _ in range(int(h["epochs"])):
            perm = torch.randperm(n, generator=self.gen)
            for i in range(0, n, self.MINIBATCH):
                j = perm[i:i + self.MINIBATCH]; d = self.net.dist(O[j], R[j]); lp = d.log_prob(A[j]); ratio = torch.exp(lp - L[j]); a = AD[j]; a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - self.CLIP, 1 + self.CLIP) * a).mean(); vl = ((self.net.value(O[j], R[j]) - RT[j]) ** 2).mean(); en = d.entropy().mean()
                loss = pg + self.VF * vl - h["ent_coef"] * en
                self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), self.MAXNORM); self.opt.step(); stats.append((pg.item(), vl.item(), en.item()))
        self.buf = {k: [] for k in self.buf}; return np.mean(stats, 0)

    def save(self, path): torch.save(self.net.state_dict(), path)
    def load(self, path): self.net.load_state_dict(torch.load(path))
