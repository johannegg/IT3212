"""Part 3: change one factor at a time from the winner's settings, in Setting B (one agent per training seed, the same budget and seeds as A-D).

    python run_part3.py [--workers 14] [--only exploration,...] [--summarize-only]

Each factor gets one lower and one higher value than the winner's (x 1/3 and x 3; gamma by its distance to 1), kept inside wide limits.
The winner itself is not trained again: its results are the Setting B runs of A-D, so run run_ad.py first."""
import argparse, glob, json, os, time, traceback
from multiprocessing import Pool
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from adlib import *

P3 = os.path.join(HERE, "records", "part3"); AD = os.path.join(HERE, "records", "ad", "runs", "B")
# investigation -> (models, factor, lower limit, upper limit)
INV = {"exploration": (["q"], "eps_frac", 0.02, 0.98), "horizon": (["q", "ppo"], "gamma", 0.5, 0.9999), "entropy": (["ppo"], "ent_coef", 0.0, 0.3),
       "update_size_q": (["q"], "alpha", 0.001, 1.0), "update_size_ppo": (["ppo"], "lr", 1e-6, 1e-2), "epochs": (["ppo"], "epochs", 1, 40)}


def pair(inv, factor, v, lo, hi):
    if factor == "gamma": a, b = 1 - (1 - v) * 3, 1 - (1 - v) / 3                  # shorter and longer horizon
    elif factor == "epochs": a, b = max(1, int(round(v / 3))), int(round(v * 3))
    elif factor == "eps_frac": a, b = v / 3, min(.95, v * 3)                         # exploration falls faster and slower
    elif factor == "ent_coef": a, b = 0.0, v * 3 if v > 0 else 0.01                 # no entropy bonus and a larger one
    else: a, b = v / 3, v * 3
    a, b = min(max(a, lo), hi), min(max(b, lo), hi)
    if abs(b - v) < 1e-9 * max(1, abs(v)): b = min(hi, v * 1.5) if factor != "gamma" else v
    return [x for x in (a, b) if abs(x - v) > 1e-9 * max(1.0, abs(v))]


def name_of(inv, al, v): return f"{inv}_{al}_{v:.6g}"


def plan(x, bonus):
    W = winners(); jobs = {}
    for inv, (als, fac, lo, hi) in INV.items():
        for al in als:
            hp, budget = W[al]
            for v in pair(inv, fac, hp[fac], lo, hi): jobs[name_of(inv, al, v)] = (al, inv, fac, v, {**hp, fac: v}, budget)
    return W, jobs


def train_job(a):
    al, hp, seed, budget, x, bonus, d = a
    try: train_run(al, replace(setting_of("B", x, bonus), start_maze="train"), hp, seed, budget, d, True); return True, d, ""
    except Exception: return False, d, traceback.format_exc()[-600:]


def eval_job(a):
    d, mode, x, bonus = a
    try: eval_run(d, "B", mode, x, bonus, N_TEST, False); return True, d, ""
    except Exception: return False, d, traceback.format_exc()[-600:]


def summarize(W, jobs):
    rows = []; out = os.path.join(P3, "results"); os.makedirs(out, exist_ok=True)
    def add(inv, al, fac, v, default, dirs):
        for d in dirs:
            mode = "greedy" if al == "q" else "argmax"; f = os.path.join(d, f"eval_B_{mode}.json")
            if os.path.exists(f): e = json.load(open(f)); c = json.load(open(os.path.join(d, "config.json"))); rows.append(dict(investigation=inv, algo=al, factor=fac, value=v, is_winner=default, seed=c["seed"], score=e["score"], success=e["success"], failure=e["failure"], len_success=e["len_success"], len_failure=e["len_failure"], train_env_steps=c["env_steps"], train_minutes=c["train_minutes"], run_dir=os.path.relpath(d, HERE)))
    for inv, (als, fac, lo, hi) in INV.items():
        for al in als:
            add(inv, al, fac, W[al][0][fac], True, sorted(glob.glob(os.path.join(AD, al, "seed*"))))
            for nm, (a2, i2, f2, v, hp, b) in jobs.items():
                if i2 == inv and a2 == al: add(inv, al, fac, v, False, sorted(glob.glob(os.path.join(P3, "runs", nm, al, "seed*"))))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(out, "per_agent.csv"), index=False); k = ["investigation", "algo", "factor", "value", "is_winner"]
    s = d.groupby(k)[["score", "success", "failure", "len_success", "train_minutes"]].agg(["mean", "std", "min", "max"]); s.columns = [f"{a}_{b}" for a, b in s.columns]; s["agents"] = d.groupby(k).size(); s.round(4).to_csv(os.path.join(out, "summary.csv")); print(s[["score_mean", "score_std", "success_mean", "agents"]].round(2).to_string())
    for inv, (als, fac, lo, hi) in INV.items():
        fig, axes = plt.subplots(1, len(als), figsize=(6.5 * len(als), 4.4), squeeze=False); src = []
        for ax, al in zip(axes[0], als):
            V = sorted({r["value"] for r in rows if r["investigation"] == inv and r["algo"] == al})
            for v in V:
                dirs = [os.path.join(HERE, r["run_dir"]) for r in rows if r["investigation"] == inv and r["algo"] == al and r["value"] == v]; cv = [pd.read_csv(os.path.join(x, "curve.csv")) for x in dirs]; win = v == W[al][0][fac]
                if not cv: continue
                n = min(map(len, cv)); ax.plot(cv[0].env_steps[:n], np.mean([c.score.values[:n] for c in cv], 0), lw=2, label=f"{fac} = {v:.4g}{' (winner)' if win else ''} ({len(cv)} agents)"); src += [os.path.relpath(x, HERE).rsplit("/", 1)[0] for x in dirs]
            ax.set_title(f"{inv}: {NAMES[al]} (Setting B)"); ax.set_xlabel("Environment interactions"); ax.set_ylabel(f"Score ({CURVE_GAMES} curve games)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.text(.01, .005, "Source runs (curve.csv, mean over agents, no smoothing): " + ", ".join(sorted(set(src))), fontsize=5.5, color=".35", wrap=True); fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(os.path.join(out, f"curves_{inv}.png"), dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=14); p.add_argument("--only", default=None); p.add_argument("--summarize-only", action="store_true"); p.add_argument("--plan", action="store_true"); a = p.parse_args(); os.environ["AD_WORKERS"] = str(a.workers); x = success_line(); bonus = SUCCESS_BONUS; W, jobs = plan(x, bonus)
    if a.only: jobs = {k: v for k, v in jobs.items() if v[1] in a.only.split(",")}
    for nm, (al, inv, fac, v, hp, b) in jobs.items(): print(f"{nm:40s} {fac} = {v:.5g} (winner {W[al][0][fac]:.5g})")
    if a.plan: return
    if not a.summarize_only:
        T = [(al, hp, s, b, x, bonus, os.path.join(P3, "runs", nm, al, f"seed{s}")) for nm, (al, inv, fac, v, hp, b) in jobs.items() for s in TRAIN_SEEDS if not os.path.exists(os.path.join(P3, "runs", nm, al, f"seed{s}", "config.json"))]; print(len(T), "training runs"); bad = 0
        with Pool(a.workers, maxtasksperchild=1) as po:
            for ok, d, err in po.imap_unordered(train_job, T): print("ok" if ok else "FAILED", d.replace(HERE, ""), err, flush=True); bad += not ok
            ev = [(os.path.join(P3, "runs", nm, al, f"seed{s}"), "greedy" if al == "q" else "argmax", x, bonus) for nm, (al, *_r) in jobs.items() for s in TRAIN_SEEDS]
            for ok, d, err in po.imap_unordered(eval_job, [e for e in ev if not os.path.exists(os.path.join(e[0], f"eval_B_{e[1]}.json"))]): print("ok" if ok else "FAILED", d.replace(HERE, ""), err, flush=True); bad += not ok
        print(bad, "failed")
    summarize(W, jobs)


if __name__ == "__main__": main()
