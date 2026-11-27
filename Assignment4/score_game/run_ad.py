"""Part 2: Settings A-D with the settings the tuning found.   python run_ad.py [--workers 14]

Trains one agent per training seed (TRAIN_SEEDS, 5 by default) per model in A, B and C, tests each with 200 held-out games (the B agents also in D), writes records/ad/.
Finished runs and tests are skipped, so it can be stopped and started again."""
import argparse
import glob
import os
import time
import traceback
from multiprocessing import Pool

import adlib
from adlib import *

ROOT = os.path.join(HERE, "records", "ad")


def train_job(a):
    algo, name, seed, hp, budget, x, bonus, root, curve = a
    d = os.path.join(root, "runs", name, algo, f"seed{seed}")
    try: train_run(algo, replace(setting_of(name, x, bonus), start_maze="train"), hp, seed, budget, d, curve); return True, d, ""
    except Exception: return False, d, traceback.format_exc()[-600:]


def eval_job(a):
    d, name, mode, x, bonus, n, record = a
    try: eval_run(d, name, mode, x, bonus, n, record); return True, f"{d} {name} {mode}", ""
    except Exception: return False, f"{d} {name} {mode}", traceback.format_exc()[-600:]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=14); p.add_argument("--quick", action="store_true"); p.add_argument("--bonus", type=float, default=SUCCESS_BONUS); a = p.parse_args()
    os.environ["AD_WORKERS"] = str(a.workers); root = os.path.join(HERE, "records", "ad_quick") if a.quick else ROOT
    if a.quick: hp = {"q": dict(life_penalty=100, ghost_value=1, gamma=.95, alpha=.1, eps_end=.03, eps_frac=.5, lam=.6, q0=0.), "ppo": dict(life_penalty=100, ghost_value=1, gamma=.97, lr=3e-4, ent_coef=.01, epochs=4, gae_lambda=.9, reward_scale=.1)}; budget, seeds, n, x = {"q": 8000, "ppo": 8000}, [0, 1], 4, 2000
    else: w = winners(); hp = {k: w[k][0] for k in w}; budget = {k: w[k][1] for k in w}; seeds, n, x = TRAIN_SEEDS, N_TEST, success_line()
    print(f"X = {x}, budget {budget}, seeds {seeds}, {n} test games, {a.workers} workers", flush=True); t = time.time()
    jobs = [(al, name, s, hp[al], budget[al], x, a.bonus, root, True) for name in ("A", "B", "C") for al in ("ppo", "q") for s in seeds if not os.path.exists(os.path.join(root, "runs", name, al, f"seed{s}", "config.json"))]
    print(len(jobs), "training runs"); bad = 0
    with Pool(a.workers, maxtasksperchild=1) as po:
        for ok, d, err in po.imap_unordered(train_job, jobs): print("ok" if ok else "FAILED", d.replace(HERE, ""), err, flush=True); bad += not ok
    ev = []
    for name in ("A", "B", "C"):
        for al in ("q", "ppo"):
            for s in seeds:
                d = os.path.join(root, "runs", name, al, f"seed{s}")
                for where, mode in [(w2, m) for w2 in ([name] + (["D"] if name == "B" else [])) for m in (["greedy"] if al == "q" else ["argmax", "sampled"])]:
                    if not os.path.exists(os.path.join(d, f"eval_{where}_{mode}.json")): ev.append((d, where, mode, x, a.bonus, n, True))
    print(len(ev), "tests")
    with Pool(a.workers, maxtasksperchild=1) as po:
        for ok, d, err in po.imap_unordered(eval_job, ev): print("ok" if ok else "FAILED", d.replace(HERE, ""), err, flush=True); bad += not ok
    print(f"done in {(time.time() - t) / 60:.1f} min, {bad} failed", flush=True)


if __name__ == "__main__": main()
