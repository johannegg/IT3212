"""Summarize evaluation results ACROSS TRAINED AGENTS (one value per seed),
and plot learning curves against environment interactions.

  python summarize.py --runs records/part2/runs --out records/part2/results

Outputs:
  <out>/per_agent.csv      one row per trained agent and evaluation (setting, mode)
  <out>/summary.csv/.md/.tex   mean, standard deviation and range across seeds
  <out>/curves_<setting>.png   learning curves (source runs printed on the figure)
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--runs", default="records/part2/runs")
p.add_argument("--out", default="records/part2/results")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

rows = []
for cfg_path in sorted(glob.glob(os.path.join(a.runs, "*", "*", "seed*", "config.json"))):
    d = os.path.dirname(cfg_path)
    cfg = json.load(open(cfg_path))
    for ev in sorted(glob.glob(os.path.join(d, "eval_*.csv"))):
        _, eval_setting, mode = os.path.basename(ev)[:-4].split("_")
        e = pd.read_csv(ev)
        meta = json.load(open(ev[:-4] + ".json"))
        rows.append({
            "train_setting": cfg["setting"]["name"], "eval_setting": eval_setting, "algo": cfg["algo"],
            "mode": mode, "seed": cfg["seed"], "run_dir": d,
            "success_rate": (e.outcome == "success").mean(), "failure_rate": (e.outcome == "failure").mean(),
            "timeout_rate": (e.outcome == "timeout").mean(), "mean_return": e["return"].mean(),
            "mean_dots": e.dots_eaten.mean(),
            "len_success": e.length[e.outcome == "success"].mean(),
            "len_failure": e.length[e.outcome == "failure"].mean(),
            "train_env_steps": cfg["train_env_steps"], "train_runtime_min": cfg["runtime_seconds"] / 60,
            "eval_runtime_min": meta["runtime_seconds"] / 60, "cpu": cfg["software"]["cpu"],
            "eval_episodes": len(e)})
if not rows:
    raise SystemExit(f"no evaluated runs found under {a.runs}")
per = pd.DataFrame(rows)
per.to_csv(os.path.join(a.out, "per_agent.csv"), index=False)

metrics = ["success_rate", "failure_rate", "timeout_rate", "mean_return", "mean_dots", "len_success",
           "len_failure", "train_env_steps", "train_runtime_min"]
keys = ["eval_setting", "train_setting", "algo", "mode"]
g = per.groupby(keys)
summ = g[metrics].agg(["mean", "std", "min", "max"])
summ.columns = [f"{m}_{s}" for m, s in summ.columns]
summ["n_seeds"] = g.size()
summ = summ.reset_index()
summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)


def fmt(r, m, digits=2):
    mean, sd = r[f"{m}_mean"], r[f"{m}_std"]
    if pd.isna(mean):
        return "-"
    return f"{mean:.{digits}f} ± {0 if pd.isna(sd) else sd:.{digits}f}"


cols = [("Success", "success_rate", 2), ("Failure", "failure_rate", 2), ("Timeout", "timeout_rate", 2),
        ("Return", "mean_return", 1), ("Length (success)", "len_success", 0), ("Length (failure)", "len_failure", 0),
        ("Train steps", "train_env_steps", 0), ("Train time (min)", "train_runtime_min", 1)]
header = ["Eval", "Trained in", "Algorithm", "Policy", "Seeds"] + [c[0] for c in cols]
lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
tex = []
names = {"q": "Q-learning", "ppo": "PPO"}
for _, r in summ.iterrows():
    vals = [r.eval_setting, r.train_setting, names[r.algo], r["mode"], str(r.n_seeds)] + [fmt(r, m, d) for _, m, d in cols]
    lines.append("| " + " | ".join(vals) + " |")
    tex.append(" & ".join(v.replace("±", "$\\pm$") for v in vals) + r" \\")
cpu = per.cpu.iloc[0]
note = (f"Values are mean ± standard deviation across trained agents (one value per training seed); each agent "
        f"was evaluated for {int(per.eval_episodes.iloc[0])} held-out episodes. Returns in Setting C include the "
        f"step penalty and are not comparable with A/B; compare success and failure rates instead. Hardware: {cpu}.")
open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n\n" + note + "\n")
open(os.path.join(a.out, "summary.tex"), "w").write(
    "% " + note + "\n\\begin{tabular}{lllll" + "r" * len(cols) + "}\n\\hline\n"
    + " & ".join(header) + r" \\" + "\n\\hline\n" + "\n".join(tex) + "\n\\hline\n\\end{tabular}\n")

# ---------------------------------------------------------------- curves
colors = {"q": "#1f77b4", "ppo": "#d62728"}
for setting in sorted({os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(c))))
                       for c in glob.glob(os.path.join(a.runs, "*", "*", "seed*", "curve.csv"))}):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    sources = []
    for algo in ["q", "ppo"]:
        files = sorted(glob.glob(os.path.join(a.runs, setting, algo, "seed*", "curve.csv")))
        if not files:
            continue
        sources.append(f"{os.path.join(a.runs, setting, algo)}/seed{{{','.join(f.split('seed')[-1].split('/')[0] for f in files)}}}")
        curves = [pd.read_csv(f) for f in files]
        for ax, m, label in zip(axes, ["success", "failure", "mean_dots"], ["Success rate", "Failure rate", "Dots eaten per episode"]):
            for c in curves:
                ax.plot(c.env_steps, c[m], color=colors[algo], alpha=0.25, lw=0.8)
            n = min(len(c) for c in curves)
            mean = np.mean([c[m].values[:n] for c in curves], 0)
            ax.plot(curves[0].env_steps.values[:n], mean, color=colors[algo], lw=2.2,
                    label=f"{names[algo]} (mean of {len(curves)} seeds)")
            ax.set_title(label); ax.set_xlabel("Environment interactions"); ax.grid(alpha=0.3)
    cfg = json.load(open(glob.glob(os.path.join(a.runs, setting, "*", "seed*", "config.json"))[0]))
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Setting {setting}: learning curves (evaluation every {cfg['eval_every']:,} interactions, "
                 f"{cfg['curve_episodes']} episodes each, no smoothing; thin lines = single seeds)", fontsize=10)
    fig.text(0.01, 0.005, "Source runs: " + "; ".join(sources) + " (curve.csv, config.json)", fontsize=7, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(a.out, f"curves_{setting}.png"), dpi=150)
    plt.close(fig)

print(open(os.path.join(a.out, "summary.md")).read())
