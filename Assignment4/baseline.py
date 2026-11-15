"""Hand-written reference policy (no learning), evaluated on the same 200
held-out episodes as the agents. It shows what is possible with the features
both agents get: go towards the nearest dot, avoid directions with a dangerous
ghost, chase frightened ghosts.

    python baseline.py            # -> records/part2/results/baseline.csv
"""
import argparse
import csv
import os

import numpy as np

from pacman_rl.env import SETTINGS, PacmanTask
from pacman_rl.experiment import EVAL_SEED_BASE, summarize_rows


def policy(f, rng):
    open_ = [(f["open"] >> k) & 1 for k in range(4)]
    danger = [(f["danger"] >> k) & 1 for k in range(4)]
    if f["chase_dir"] < 4:
        return f["chase_dir"]
    if open_[f["dot_dir"]] and not danger[f["dot_dir"]]:
        return f["dot_dir"]
    safe = [k for k in range(4) if open_[k] and not danger[k]]
    return int(rng.choice(safe)) if safe else f["dot_dir"]


p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=200)
p.add_argument("--out", default="records/part2/results")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
out = []
for name in ["A", "B", "C", "D"]:
    task, rng, rows = PacmanTask(SETTINGS[name]), np.random.default_rng(0), []
    for i in range(a.episodes):
        _, _, info = task.reset(EVAL_SEED_BASE + i)
        ret = 0.0
        while True:
            _, _, r, term, trunc, info = task.step(policy(info["features"], rng))
            ret += r
            if term or trunc:
                break
        rows.append({"outcome": info["outcome"], "return": ret, "length": task.t, "dots_eaten": info["dots_eaten"]})
    out.append({"setting": name, **{k: round(v, 3) for k, v in summarize_rows(rows).items()}})
    print(out[-1], flush=True)
with open(os.path.join(a.out, "baseline.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)
