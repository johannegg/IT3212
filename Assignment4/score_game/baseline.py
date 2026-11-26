"""Players that do not learn, on the same 200 held-out games: the hand-written rule and random moves, per setting.   python baseline.py"""
import csv, os
from multiprocessing import Pool
import numpy as np
import adlib
from adlib import *


def rule(g, rng):
    f = g.f; safe = [k for k in range(4) if f["op"][k] and not f["dg"][k]]
    if f["blue"] < 4: return f["blue"]
    if f["dot"] < 4 and not f["dg"][f["dot"]]: return f["dot"]
    return int(rng.choice(safe)) if safe else int(rng.integers(4))


def go(a):
    name, player, x, lo, hi = a; st = replace(setting_of(name, x, 0.0), start_maze="3" if name == "D" else "0", step_penalty=0.0); g = ScoreGame(st, training=False); rows = []
    for i in range(lo, hi):
        g.reset(TEST_BASE + i); rng = np.random.default_rng(TEST_BASE + i)
        while True:
            r, d, t, info = g.step(rule(g, rng) if player == "hand-written" else int(rng.integers(4)))
            if d or t: break
        rows.append((info["score"], g.decisions, info["outcome"]))
    return name, player, rows


if __name__ == "__main__":
    x = success_line(); out = os.path.join(HERE, "records", "ad", "results"); os.makedirs(out, exist_ok=True); T = {}
    with Pool(14) as p:
        for n, pl, rows in p.imap_unordered(go, [(n, pl, x, lo, lo + 20) for n in "ABCD" for pl in ("hand-written", "random") for lo in range(0, 200, 20)]): T.setdefault((n, pl), []).extend(rows)
    with open(os.path.join(out, "baseline.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["setting", "player", "games", "score", "score_sd", "success", "reach_2311", "reach_6952", "reach_15693", "decisions"])
        for (n, pl), r in sorted(T.items()):
            s = np.array([z[0] for z in r]); w.writerow([n, pl, len(r), round(s.mean(), 1), round(s.std(), 1), round((s >= x).mean(), 3), round((s >= 2311).mean(), 3), round((s >= 6952).mean(), 3), round((s >= 15693).mean(), 3), round(np.mean([z[1] for z in r]), 1)])
    print(open(os.path.join(out, "baseline.csv")).read())
