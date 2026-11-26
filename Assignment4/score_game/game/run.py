"""Training and test games, the same for the tuning and for A-D."""
from __future__ import annotations

import time

import numpy as np
import torch

from .agents import PPOAgent, QAgent
from .env import SETTINGS, ScoreGame
from dataclasses import replace

Q_HP = ["alpha", "gamma", "lam", "eps_end", "eps_frac", "q0"]
PPO_HP = ["lr", "gamma", "ent_coef", "epochs", "gae_lambda", "reward_scale"]
TRAIN_BASE = 0                       # training game seed = run seed * 1,000,000 + game number


def guides(hp): return {"life_penalty": hp["life_penalty"], "ghost_value": hp["ghost_value"]}


def play(game, agent, algo, seed, greedy=True):
    """One test game without learning: score, decisions, steps, outcome."""
    game.reset(seed)
    while True:
        if algo == "q":
            a = agent.greedy(agent.state(game)) if greedy else agent.act(agent.state(game))[0]
        else: a = agent.act(game, "argmax" if greedy else "sampled")[0]
        r, done, tr, info = game.step(a)
        if done or tr: return dict(score=info["score"], decisions=game.decisions, steps=game.env_steps, outcome=info["outcome"], ghosts=game.ghosts_eaten, mazes=game.layout + 1, stuck=bool(info["stuck"]))


def train(algo, setting, hp, seed, budget, log=None, curve=None):
    """Train one agent for `budget` environment steps (all steps of the game count). Returns the agent and the training games."""
    torch.set_num_threads(1)
    g = ScoreGame(setting, guides(hp), training=True)
    agent = QAgent({k: hp[k] for k in Q_HP}, seed, budget) if algo == "q" else PPOAgent({k: hp[k] for k in PPO_HP}, seed)
    steps, n, games, t0 = 0, 0, [], time.time()
    while steps < budget:
        g.reset(TRAIN_BASE + seed * 1_000_000 + n); n += 1; steps += g.env_steps; last = g.env_steps
        if algo == "q": agent.steps = steps; s = agent.state(g); agent.end_game()
        while True:
            if algo == "q":
                a, greedy = agent.act(s); r, done, tr, info = g.step(a); s2 = agent.state(g)
                steps += g.env_steps - last; last = g.env_steps; agent.steps = steps
                agent.update(s, a, r, s2, done, greedy); s = s2
            else:
                a, view = agent.act(g); r, done, tr, info = g.step(a); steps += g.env_steps - last; last = g.env_steps
                agent.store(view, a, r, done or tr, agent.value_now(g) if (tr and not done) else 0.0)
                if agent.ready():
                    agent.update(0.0 if (done or tr) else agent.value_now(g))
            if done or tr or steps >= budget: break
        if algo == "q": agent.end_game()
        games.append((steps, info["score"], info.get("outcome", "budget end"), g.decisions, g.lives, g.layout, bool(info.get("held_out"))))
        if curve is not None: curve(steps, agent)
    g.close()
    return agent, games, time.time() - t0


def test_games(algo, agent, setting, seeds, maze="0", greedy=True, stop_at_maze4=False):
    """stop_at_maze4: used in the tuning, so that the held-out maze never influences a choice."""
    g = ScoreGame(replace(setting, start_maze=maze), training=stop_at_maze4)
    out = [play(g, agent, algo, s, greedy) for s in seeds]; g.close()
    return out
