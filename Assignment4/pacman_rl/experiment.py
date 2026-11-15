"""Training and evaluation loops shared by both algorithms."""
from __future__ import annotations

import csv
import gzip
import json
import os
import platform
import time
from dataclasses import replace

import numpy as np
import torch

from .agents import PPO, QLearning
from .env import ACTION_NAMES, SETTINGS, PacmanTask

# Seed lists. Training, learning-curve and final evaluation episodes never share seeds.
TRAIN_SEED_BASE = 0            # episode seed = run_seed * 1_000_000 + episode index
CURVE_SEED_BASE = 20_000_000   # periodic evaluation for learning curves (used for monitoring)
EVAL_SEED_BASE = 10_000_000    # held-out final evaluation (never used for tuning)

DEFAULTS = {
    "q": dict(alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.5),
    "ppo": dict(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5,
                epochs=4, rollout=2048, minibatch=256, hidden=64),
}


def software_info():
    import ale_py, gymnasium
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":")[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "gymnasium": gymnasium.__version__, "ale_py": ale_py.__version__,
            "platform": platform.platform(), "cpu": cpu, "torch_threads": torch.get_num_threads()}


def make_agent(algo, hp, seed, budget):
    if algo == "q":
        return QLearning(alpha=hp["alpha"], gamma=hp["gamma"], eps_start=hp["eps_start"],
                         eps_end=hp["eps_end"], eps_decay_steps=int(hp["eps_decay_frac"] * budget), seed=seed)
    return PPO(seed=seed, **{k: hp[k] for k in DEFAULTS["ppo"]})


def action_mask(obs):
    """Only directions without a wall can be chosen (pressing into a wall makes
    Ms. Pac-Man stand still). If no direction looks open, allow all."""
    m = obs[:4] > 0.5
    return m if m.any() else np.ones(4, dtype=bool)


def choose(agent, algo, sid, obs, mode):
    """Returns (action, values) where values are Q-values or action probabilities."""
    mask = action_mask(obs)
    if algo == "q":
        return agent.greedy(sid, mask), agent.action_values(sid)
    probs = agent.probs(obs, mask)
    if mode == "argmax":
        return int(np.argmax(probs)), probs
    a, _, _ = agent.act(obs, mask, "sampled")
    return a, probs


def run_episode(task, agent, algo, seed, mode, trace=None, episode_idx=0):
    sid, obs, info = task.reset(seed)
    env_steps = info["env_steps"]
    ret, outcome = 0.0, None
    while True:
        a, vals = choose(agent, algo, sid, obs, mode)
        if trace is not None:
            f = info["features"]
            trace.append([episode_idx, task.t, ACTION_NAMES[a], *np.round(vals, 4), f["open"], f["dot_dir"],
                          f["danger"], f["chase_dir"], f["power_dir"], f["pac_node"], int(f["frightened"]),
                          *task.ale.getRAM()[[10, 16, 6, 7, 8, 9, 12, 13, 14, 15]].tolist()])
        sid, obs, r, term, trunc, info = task.step(a)
        env_steps += 1
        ret += r
        if term or trunc:
            outcome = info["outcome"]
            break
    return {"seed": seed, "outcome": outcome, "return": ret, "length": task.t,
            "dots_eaten": info["dots_eaten"], "env_steps": env_steps}


def evaluate_agent(task, agent, algo, seeds, mode, trace=None):
    rows = [run_episode(task, agent, algo, s, mode, trace, i) for i, s in enumerate(seeds)]
    for i, r in enumerate(rows):
        r["episode"] = i
    return rows


def summarize_rows(rows):
    out = {k: float(np.mean([r["outcome"] == k for r in rows])) for k in ["success", "failure", "timeout"]}
    out["mean_return"] = float(np.mean([r["return"] for r in rows]))
    out["mean_length"] = float(np.mean([r["length"] for r in rows]))
    out["mean_dots"] = float(np.mean([r["dots_eaten"] for r in rows]))
    return out


# ---------------------------------------------------------------------------
def train(algo, setting_name, seed, budget, out_root="records/part2/runs", eval_every=25_000,
          curve_episodes=10, hp_overrides=None, setting_overrides=None):
    torch.set_num_threads(1)
    hp = dict(DEFAULTS[algo], **(hp_overrides or {}))
    setting = replace(SETTINGS[setting_name], **(setting_overrides or {}))
    run_dir = os.path.join(out_root, setting_name, algo, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    task = PacmanTask(setting)
    curve_task = PacmanTask(setting)
    agent = make_agent(algo, hp, seed, budget)
    eval_mode = "greedy" if algo == "q" else "argmax"

    config = {"algo": algo, "setting": setting.to_dict(), "seed": seed, "budget_env_steps": budget,
              "hyperparameters": hp, "eval_every": eval_every, "curve_episodes": curve_episodes,
              "curve_eval_mode": eval_mode, "out_root": out_root, "software": software_info(),
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    ep_f = open(os.path.join(run_dir, "train_episodes.csv"), "w", newline="")
    ep_w = csv.writer(ep_f)
    ep_w.writerow(["episode", "env_steps_total", "return", "outcome", "length", "dots_eaten", "epsilon"])
    curve = []
    t0 = time.time()
    env_steps, episode, next_eval = 0, 0, 0
    curve_seeds = [CURVE_SEED_BASE + i for i in range(curve_episodes)]
    ppo_stats = []

    def do_curve_eval():
        rows = evaluate_agent(curve_task, agent, algo, curve_seeds, eval_mode)
        curve.append({"env_steps": env_steps, "wall_time": time.time() - t0, **summarize_rows(rows)})

    while env_steps < budget:
        if env_steps >= next_eval:
            do_curve_eval()
            next_eval += eval_every
        ep_seed = TRAIN_SEED_BASE + seed * 1_000_000 + episode
        sid, obs, info = task.reset(ep_seed)
        env_steps += info["env_steps"]
        ret = 0.0
        while True:
            mask = action_mask(obs)
            if algo == "q":
                a = agent.act(sid, env_steps, mask)
            else:
                a, logp, val = agent.act(obs, mask, "sampled")
            nsid, nobs, r, term, trunc, info = task.step(a)
            env_steps += 1
            ret += r
            if algo == "q":
                agent.update(sid, a, r, nsid, term, action_mask(nobs))   # bootstrap on timeout, not on success/failure
            else:
                r_store = r + (agent.gamma * agent.value(nobs) if trunc and not term else 0.0)
                agent.store(obs, mask, a, logp, val, r_store, float(term or trunc))
                if agent.ready():
                    ppo_stats.append({"env_steps": env_steps, **agent.update(nobs, term or trunc)})
            sid, obs = nsid, nobs
            if term or trunc or env_steps >= budget:
                break
        eps = agent.epsilon(env_steps) if algo == "q" else ""
        ep_w.writerow([episode, env_steps, round(ret, 3), info["outcome"] or "budget_end", task.t,
                       info["dots_eaten"], eps])
        ep_f.flush()
        episode += 1
    do_curve_eval()
    ep_f.close()

    runtime = time.time() - t0
    agent.save(os.path.join(run_dir, "model.npy" if algo == "q" else "model.pt"))
    with open(os.path.join(run_dir, "curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader(); w.writerows(curve)
    if ppo_stats:
        with open(os.path.join(run_dir, "ppo_updates.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ppo_stats[0]))
            w.writeheader(); w.writerows(ppo_stats)
    curve_steps = len(curve) * curve_episodes
    config.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime_seconds": round(runtime, 1),
                   "train_env_steps": env_steps, "train_episodes": episode,
                   "note_runtime": "runtime includes the periodic learning-curve evaluations"})
    if algo == "q":
        config["q_states_visited"] = int((agent.q != 0).any(1).sum())
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    task.close(); curve_task.close()
    return run_dir


# ---------------------------------------------------------------------------
TRACE_HEADER = ["episode", "t", "action", "v_UP", "v_RIGHT", "v_LEFT", "v_DOWN", "open_bits", "dot_dir",
                "danger_bits", "chase_dir", "power_dir", "pac_node", "frightened",
                "ram_pac_x", "ram_pac_y", "ram_g0_x", "ram_g1_x", "ram_g2_x", "ram_g3_x",
                "ram_g0_y", "ram_g1_y", "ram_g2_y", "ram_g3_y"]


def evaluate_run(run_dir, setting_name=None, episodes=200, mode=None, save_traces=True):
    """Held-out evaluation of a trained agent. No parameter updates happen here."""
    torch.set_num_threads(1)
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    algo = cfg["algo"]
    train_setting = SETTINGS[cfg["setting"]["name"]]
    base = replace(train_setting, **{k: v for k, v in cfg["setting"].items() if k != "name"})
    setting_name = setting_name or base.name
    setting = base if setting_name == base.name else replace(
        SETTINGS[setting_name], **{k: v for k, v in cfg["setting"].items()
                                   if k not in ("name", "maze", "sticky", "step_penalty")})
    mode = mode or ("greedy" if algo == "q" else "argmax")

    agent = make_agent(algo, cfg["hyperparameters"], cfg["seed"], cfg["budget_env_steps"])
    agent.load(os.path.join(run_dir, "model.npy" if algo == "q" else "model.pt"))
    if algo == "ppo":
        agent.gen.manual_seed(EVAL_SEED_BASE + cfg["seed"])
    task = PacmanTask(setting)
    seeds = [EVAL_SEED_BASE + i for i in range(episodes)]
    trace = [] if save_traces else None
    t0 = time.time()
    rows = evaluate_agent(task, agent, algo, seeds, mode, trace)
    runtime = time.time() - t0
    task.close()

    name = f"eval_{setting_name}_{mode}"
    with open(os.path.join(run_dir, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "seed", "outcome", "return", "length", "dots_eaten", "env_steps"])
        w.writeheader(); w.writerows(rows)
    if trace is not None:
        with gzip.open(os.path.join(run_dir, f"traces_{setting_name}_{mode}.csv.gz"), "wt", newline="") as f:
            w = csv.writer(f); w.writerow(TRACE_HEADER); w.writerows(trace)
    meta = {"setting": setting.to_dict(), "mode": mode, "episodes": episodes, "seeds": [seeds[0], seeds[-1]],
            "runtime_seconds": round(runtime, 1), "software": software_info(), **summarize_rows(rows)}
    with open(os.path.join(run_dir, name + ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
