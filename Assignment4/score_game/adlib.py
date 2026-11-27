"""Training, test games and recordings of A-D and Part 3 (the game and the agents are the ones the tuning uses)."""
from __future__ import annotations

import csv
import json
import os
import platform
import time
from dataclasses import replace

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

from game.agents import PPOAgent, QAgent
from game.env import ACTION_NAMES, SETTINGS, ScoreGame
from game.run import PPO_HP, Q_HP, guides

HERE = os.path.dirname(os.path.abspath(__file__))
# Lists of games. They never overlap with the tuning (100,000,000+ and 300,000,000+) or with each other.
CURVE_BASE, TEST_BASE, DIAG_BASE, N_TEST, CURVE_POINTS, CURVE_GAMES = 400_000_000, 500_000_000, 700_000_000, 200, 12, 5
TRAIN_SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]      # SEEDS=5,6,7,8,9 adds agents; finished seeds are skipped
NAMES = {"q": "Q-learning", "ppo": "PPO"}


def winners():
    w = json.load(open(os.path.join(HERE, "tuning", "winners.json")))
    return {a: (w[a]["hp"], w[a]["budget"]) for a in w}


def success_line():
    """X (decisions 2.2). Read from tuning/x.json (now 6,952 by decision); without that file: 6,952, or 15,693 when more than 90% of
    the tuning test games of either winner reach 6,952."""
    f = os.path.join(HERE, "tuning", "x.json")
    if os.path.exists(f): return json.load(open(f))["x"]
    w = json.load(open(os.path.join(HERE, "tuning", "winners.json")))
    share = {a: w[a]["share_at_6952"] for a in w}
    x = 15693 if max(share.values()) > 0.9 else 6952
    json.dump(dict(x=x, share_at_6952=share), open(f, "w"), indent=1); return x


def software():
    import ale_py, gymnasium
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f: cpu = next(l.split(":")[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration): pass
    return dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, gymnasium=gymnasium.__version__, ale_py=ale_py.__version__, platform=platform.platform(), cpu=cpu, cpu_threads=os.cpu_count(), workers=os.environ.get("AD_WORKERS"))


SUCCESS_BONUS = 1600.0       # decisions 2.6: the largest single reward in the game (the fourth ghost)


def setting_of(name, x, bonus):
    """A-D. The only difference between A and B is the sticky actions, and between B and C the step penalty (training reward)."""
    return replace(SETTINGS[name], success_score=float(x), success_bonus=float(bonus))


def new_agent(algo, hp, seed, budget):
    return QAgent({k: hp[k] for k in Q_HP}, seed, budget) if algo == "q" else PPOAgent({k: hp[k] for k in PPO_HP}, seed)


def values_of(agent, algo, game): return agent.values(game) if algo == "q" else agent.probs(game)


class Recorder:
    """One row per decision: what she saw, the 4 values (Q) or probabilities (PPO), the choice, positions and what happened."""
    INT = ["episode", "t", "action", "heading", "open", "dot", "danger", "blue", "power", "fruit", "lives", "dots_left", "maze", "x", "y", "gx0", "gx1", "gx2", "gx3", "gy0", "gy1", "gy2", "gy3", "dangerous", "edible", "points", "lost"]

    def __init__(self): self.i, self.v = [], []

    def before(self, ep, t, game, agent, algo, action):
        r = game.ale.getRAM(); f = game.f; v = values_of(agent, algo, game); b = [1, 2, 4, 8]
        self.i.append([ep, t, action, game.direction, int(f["op"] @ b), f["dot"], int(sum(int(x) * 3 ** k for k, x in enumerate(f["dg"]))), f["blue"], f["power"], f["fruit"], game.lives, game.dots_left, game.layout + 1, int(r[10]), int(r[16])] + [int(r[6 + k]) for k in range(4)] + [int(r[12 + k]) for k in range(4)] + [int(sum((not game.edible[k] and not game.eyes[k]) * b[k] for k in range(4))), int(sum(game.edible[k] * b[k] for k in range(4))), 0, 0])
        self.v.append(v)

    def after(self, info): self.i[-1][-2], self.i[-1][-1] = int(info["points"]), int(info["lost"])

    def save(self, path): np.savez_compressed(path, data=np.array(self.i, np.int32 if max(abs(x) for r in self.i for x in r) > 32000 else np.int16), values=np.array(self.v, np.float32), columns=np.array(self.INT))


def test_run(agent, algo, setting, seeds, greedy=True, rec=None):
    """Test games without learning. Ties and sampled actions use a generator seeded by the game, so every game can be replayed alone."""
    g = ScoreGame(setting, training=False); out = []
    for ep, s in enumerate(seeds):
        g.reset(s); t = 0
        if algo == "q": agent.rng = np.random.default_rng(s)
        else: agent.gen.manual_seed(s % (2 ** 31 - 1))
        pts = 0.0
        while True:
            a = agent.greedy(agent.state(g)) if algo == "q" else agent.act(g, "argmax" if greedy else "sampled")[0]
            if rec: rec.before(ep, t, g, agent, algo, a)
            r, done, tr, info = g.step(a); t += 1
            if rec: rec.after(info)
            if done or tr: break
        out.append(dict(episode=ep, seed=s, score=info["score"], outcome=info["outcome"], decisions=g.decisions, env_steps=g.env_steps, lives_left=g.lives, ghosts=g.ghosts_eaten, start_maze=g.start_layout + 1, end_maze=g.layout + 1, stuck=int(info["stuck"])))
    g.close(); return out


def train_run(algo, setting, hp, seed, budget, outdir, curve=True):
    """Train one agent, save it, its training games and its learning curve (CURVE_GAMES test games at CURVE_POINTS places)."""
    torch.set_num_threads(1); os.makedirs(outdir, exist_ok=True); g = ScoreGame(setting, guides(hp), training=True); agent = new_agent(algo, hp, seed, budget)
    steps = n = 0; t0 = time.time(); games = []; pts = []; nxt = 0; every = budget / CURVE_POINTS; ctime = 0.0

    def point():
        nonlocal ctime
        a = time.time(); state = (agent.rng, agent.q.copy()) if algo == "q" else agent.gen.get_state()
        r = test_run(agent, algo, replace(setting, start_maze="0", sticky=setting.sticky, step_penalty=0.0, success_bonus=0.0), [CURVE_BASE + i for i in range(CURVE_GAMES)]) if curve else []
        if algo == "q": agent.rng = state[0]
        else: agent.gen.set_state(state)
        ctime += time.time() - a
        if r: pts.append(dict(env_steps=steps, score=float(np.mean([x["score"] for x in r])), success=float(np.mean([x["outcome"] == "success" for x in r])), decisions=float(np.mean([x["decisions"] for x in r]))))
    while steps < budget:
        if steps >= nxt: point(); nxt += every
        g.reset(seed * 1_000_000 + n); n += 1; steps += g.env_steps; last = g.env_steps; ret = 0.0
        if algo == "q": agent.steps = steps; s = agent.state(g); agent.end_game()
        while True:
            if algo == "q":
                a, gr = agent.act(s); r, done, tr, info = g.step(a); s2 = agent.state(g); steps += g.env_steps - last; last = g.env_steps; agent.steps = steps; agent.update(s, a, r, s2, done, gr); s = s2
            else:
                a, view = agent.act(g); r, done, tr, info = g.step(a); steps += g.env_steps - last; last = g.env_steps
                agent.store(view, a, r, done or tr, agent.value_now(g) if (tr and not done) else 0.0)
                if agent.ready(): agent.update(0.0 if (done or tr) else agent.value_now(g))
            ret += r
            if done or tr or steps >= budget: break
        if algo == "q": agent.end_game()
        games.append([n, steps, round(ret, 2), info["score"], info.get("outcome", "budget end"), g.decisions, g.start_layout + 1, int(bool(info.get("held_out")))])
    point(); g.close(); dt = time.time() - t0 - ctime
    agent.save(os.path.join(outdir, "model.pkl" if algo == "q" else "model.pt"))
    with open(os.path.join(outdir, "train_games.csv"), "w", newline="") as f: w = csv.writer(f); w.writerow(["game", "env_steps", "training_return", "score", "outcome", "decisions", "start_maze", "stopped_at_maze4"]); w.writerows(games)
    with open(os.path.join(outdir, "curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env_steps", "score", "success", "decisions"]); w.writeheader(); w.writerows(pts)
    cfg = dict(algo=algo, setting=setting.to_dict(), seed=seed, budget=budget, env_steps=steps, hp=hp, train_minutes=dt / 60, curve_minutes=ctime / 60, games=n, rows=len(getattr(agent, "ids", {})), curve=dict(points=CURVE_POINTS, games=CURVE_GAMES, smoothing="none"), software=software(), finished=time.strftime("%F %T"))
    json.dump(cfg, open(os.path.join(outdir, "config.json"), "w"), indent=1); return cfg


def load_agent(outdir):
    cfg = json.load(open(os.path.join(outdir, "config.json"))); algo = cfg["algo"]; a = new_agent(algo, cfg["hp"], cfg["seed"], cfg["budget"]); a.load(os.path.join(outdir, "model.pkl" if algo == "q" else "model.pt")); return a, cfg


def eval_run(outdir, name, mode, x, bonus, episodes=N_TEST, record=True):
    """`episodes` held-out test games of a saved agent in setting `name` ('A'..'D'). mode: greedy / argmax (the main policy) or sampled (PPO)."""
    torch.set_num_threads(1); agent, cfg = load_agent(outdir); algo = cfg["algo"]; st = replace(setting_of(name, x, bonus), start_maze="3" if name == "D" else "0", step_penalty=0.0)
    rec = Recorder() if record and mode != "sampled" else None; t = time.time()
    out = test_run(agent, algo, st, [TEST_BASE + i for i in range(episodes)], greedy=mode != "sampled", rec=rec)
    pen = SETTINGS[name].step_penalty      # the return of this setting: real points, the success reward and (in C) the step penalty, without the training guides
    for o in out: o["return"] = o["score"] + (bonus if o["outcome"] == "success" else 0.0) - pen * o["decisions"]
    with open(os.path.join(outdir, f"eval_{name}_{mode}.csv"), "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    if rec: rec.save(os.path.join(outdir, f"traces_{name}_{mode}.npz"))
    sc = np.array([o["score"] for o in out]); ok = np.array([o["outcome"] == "success" for o in out]); bad = np.array([o["outcome"] == "failure" for o in out])
    json.dump(dict(setting=name, mode=mode, episodes=episodes, x=x, minutes=(time.time() - t) / 60, score=float(sc.mean()), return_mean=float(np.mean([o["return"] for o in out])), score_sd_games=float(sc.std()), success=float(ok.mean()), failure=float(bad.mean()), timeout=float(1 - ok.mean() - bad.mean()), reach_2311=float((sc >= 2311).mean()), reach_6952=float((sc >= 6952).mean()), reach_15693=float((sc >= 15693).mean()), len_success=float(np.mean([o["decisions"] for o in out if o["outcome"] == "success"])) if ok.any() else None, len_failure=float(np.mean([o["decisions"] for o in out if o["outcome"] == "failure"])) if bad.any() else None, ghosts=float(np.mean([o["ghosts"] for o in out]))), open(os.path.join(outdir, f"eval_{name}_{mode}.json"), "w"), indent=1)
