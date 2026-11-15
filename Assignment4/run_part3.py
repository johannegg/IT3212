"""Part 3: change ONE factor at a time (in Setting B) and compare.

    python run_part3.py --workers 4                    # all investigations
    python run_part3.py --only epochs --workers 4      # one investigation

Everything not listed for an investigation stays at the defaults in
pacman_rl/experiment.py (DEFAULTS), with the same budget, seeds and setting
as the main experiment. The default value of each factor is NOT retrained:
it is read from the Part 2 runs (records/part2/runs/B), so run run_experiments.py first.
Runs go to records/part3/runs/<investigation>_<algo>_<value>/, results to records/part3/results/.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# investigation -> (algorithms, command-line flag, values to test, default value)
INVESTIGATIONS = {
    "exploration": (["q"], "eps-decay-frac", [0.1, 0.9], 0.5),
    "horizon": (["q", "ppo"], "gamma", [0.9, 0.999], 0.99),
    "entropy": (["ppo"], "ent-coef", [0.0, 0.05], 0.01),
    "update_size_q": (["q"], "alpha", [0.02, 0.5], 0.1),
    "update_size_ppo": (["ppo"], "lr", [1e-4, 1e-3], 3e-4),
    "epochs": (["ppo"], "epochs", [1, 10], 4),
}

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--budget", type=int, default=300_000)
p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
p.add_argument("--episodes", type=int, default=200)
p.add_argument("--only", default=None, help="comma-separated investigation names")
p.add_argument("--summarize-only", action="store_true")
a = p.parse_args()
chosen = a.only.split(",") if a.only else list(INVESTIGATIONS)


PART2_RUNS = os.path.join("records", "part2", "runs")
PART3 = os.path.join("records", "part3")


def tag(inv, algo, value):
    return f"{inv}_{algo}_{value}"


def run_dir(t, algo, seed):
    root = PART2_RUNS if t is None else os.path.join(PART3, "runs", t)
    return os.path.join(root, "B", algo, f"seed{seed}")


def sh(cmd):
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    print(f"[{'ok' if r.returncode == 0 else 'FAILED'} {time.time() - t0:5.0f}s] {' '.join(cmd)}", flush=True)
    if r.returncode:
        print(r.stderr[-1500:], flush=True)
    return r.returncode


def summarize():
    os.makedirs(os.path.join(PART3, "results"), exist_ok=True)
    rows = []
    for inv in chosen:
        algos, flag, values, default = INVESTIGATIONS[inv]
        fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 3.6), squeeze=False)
        sources = []
        for ax, algo in zip(axes[0], algos):
            mode = "greedy" if algo == "q" else "argmax"
            for value in sorted(values + [default]):
                t = None if value == default else tag(inv, algo, value)
                curves = []
                for s in range(a.seeds):
                    d = run_dir(t, algo, s)
                    ev = os.path.join(d, f"eval_B_{mode}.csv")
                    if not os.path.exists(ev):
                        continue
                    e = pd.read_csv(ev)
                    rows.append({"investigation": inv, "algo": algo, "factor": flag, "value": value,
                                 "is_default": value == default, "seed": s, "run_dir": d,
                                 "success": (e.outcome == "success").mean(), "failure": (e.outcome == "failure").mean(),
                                 "dots": e.dots_eaten.mean(),
                                 "runtime_min": json.load(open(os.path.join(d, "config.json")))["runtime_seconds"] / 60})
                    curves.append(pd.read_csv(os.path.join(d, "curve.csv")))
                if curves:
                    n = min(len(c) for c in curves)
                    ax.plot(curves[0].env_steps[:n], np.mean([c["mean_dots"].values[:n] for c in curves], 0),
                            lw=2, label=f"{flag} = {value}{' (default)' if value == default else ''} ({len(curves)} seeds)")
                    sources.append(os.path.dirname(run_dir(t, algo, 0)))
            ax.set_title(f"{inv}: {'Q-learning' if algo == 'q' else 'PPO'} (Setting B)")
            ax.set_xlabel("Environment interactions"); ax.set_ylabel("Dots eaten (curve evaluation)")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.text(0.01, 0.005, "Source runs: " + "; ".join(sources) + " (curve.csv, mean over seeds, no smoothing)",
                 fontsize=6, color="0.3")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(os.path.join(PART3, "results", f"curves_{inv}.png"), dpi=150)
        plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PART3, "results", "per_agent.csv"), index=False)
    s = df.groupby(["investigation", "algo", "factor", "value"])[["success", "failure", "dots", "runtime_min"]].agg(["mean", "std"])
    s["n_seeds"] = df.groupby(["investigation", "algo", "factor", "value"]).size()
    s.round(3).to_csv(os.path.join(PART3, "results", "summary.csv"))
    print(s.round(3).to_string())


if __name__ == "__main__":
    if not a.summarize_only:
        train_jobs, eval_jobs = [], []
        for inv in chosen:
            algos, flag, values, _ = INVESTIGATIONS[inv]
            for algo in algos:
                for value in values:
                    for s in range(a.seeds):
                        train_jobs.append(["train.py", "--algo", algo, "--setting", "B", "--seed", str(s),
                                           "--budget", str(a.budget), "--out", os.path.join(PART3, "runs", tag(inv, algo, value)),
                                           f"--{flag}", str(value)])
                        eval_jobs.append(["evaluate.py", run_dir(tag(inv, algo, value), algo, s), "--episodes", str(a.episodes)])
        missing = [run_dir(None, al, s) for al in ["q", "ppo"] for s in range(a.seeds)
                   if not glob.glob(os.path.join(run_dir(None, al, s), "eval_B_*.csv"))]
        if missing:
            print("warning: default runs missing (run run_experiments.py first):", missing)
        print(f"{len(train_jobs)} training runs, {a.workers} workers", flush=True)
        with Pool(a.workers) as pool:
            pool.map(sh, sorted(train_jobs, key=lambda j: j[2] != "ppo"), chunksize=1)
            pool.map(sh, eval_jobs, chunksize=1)
    summarize()
