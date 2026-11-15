"""Reproduce the main Part 2 experiment with ONE command:

    python run_experiments.py --workers 4

It (1) trains Q-learning and PPO in Settings A, B and C for every training seed,
(2) evaluates every agent for 200 held-out episodes in its own setting,
(3) evaluates the Setting-B agents on the unseen maze (Setting D) without training,
(4) also evaluates PPO with sampled actions (reported separately),
(5) evaluates a hand-written reference policy (baseline.py), and
(6) writes the summary tables and learning-curve figures to records/part2/results/.

Use --quick for a short smoke test with tiny budgets (written to smoke_test/, not to records/).
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from multiprocessing import Pool

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--budget", type=int, default=300_000)
p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
p.add_argument("--out", default="records/part2", help="runs go to <out>/runs, tables/figures to <out>/results")
p.add_argument("--episodes", type=int, default=200)
p.add_argument("--settings", default="ABC")
p.add_argument("--algos", default="q,ppo")
p.add_argument("--quick", action="store_true")
a = p.parse_args()
if a.quick:   # tiny smoke test, kept out of the real records
    a.seeds, a.budget, a.episodes = 2, 6000, 5
    if a.out == "records/part2":
        a.out = "smoke_test"
extra = ["--eval-every", "3000", "--curve-episodes", "2"] if a.quick else []


def sh(cmd):
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"[{status} {time.time() - t0:6.0f}s] {' '.join(cmd)}", flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], flush=True)
    return r.returncode


algos = a.algos.split(",")
runs = list(itertools.product(list(a.settings), algos, range(a.seeds)))
train_jobs = [["train.py", "--algo", al, "--setting", s, "--seed", str(sd), "--budget", str(a.budget),
               "--out", os.path.join(a.out, "runs")] + extra for s, al, sd in runs]
eval_jobs = []
for s, al, sd in runs:
    d = os.path.join(a.out, "runs", s, al, f"seed{sd}")
    ep = ["--episodes", str(a.episodes)]
    eval_jobs.append(["evaluate.py", d] + ep)
    if al == "ppo":
        eval_jobs.append(["evaluate.py", d, "--mode", "sampled"] + ep)
    if s == "B":
        eval_jobs.append(["evaluate.py", d, "--setting", "D"] + ep)
        if al == "ppo":
            eval_jobs.append(["evaluate.py", d, "--setting", "D", "--mode", "sampled"] + ep)

if __name__ == "__main__":
    t0 = time.time()
    print(f"{len(train_jobs)} training runs, {len(eval_jobs)} evaluations, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        # longest jobs (PPO) first so the workers stay busy
        fails = sum(pool.map(sh, sorted(train_jobs, key=lambda j: j[2] != "ppo"), chunksize=1))
        fails += sum(pool.map(sh, eval_jobs, chunksize=1))
    fails += sh(["baseline.py", "--episodes", str(a.episodes), "--out", os.path.join(a.out, "results")])
    fails += sh(["summarize.py", "--runs", os.path.join(a.out, "runs"), "--out", os.path.join(a.out, "results")])
    print(f"done in {(time.time() - t0) / 60:.1f} min, {fails} failed jobs")
