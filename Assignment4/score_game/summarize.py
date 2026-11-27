"""Tables and figures of A-D.   python summarize.py [--runs records/ad]"""
import argparse, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); NAMES = {"q": "Q-learning", "ppo": "PPO"}


def collect(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "runs", "*", "*", "seed*", "eval_*.json"))):
        d = os.path.dirname(f); e = json.load(open(f)); c = json.load(open(os.path.join(d, "config.json"))); g = pd.read_csv(os.path.join(d, f.split("/")[-1].replace(".json", ".csv")))
        ok, bad = g.outcome == "success", g.outcome == "failure"
        tg = pd.read_csv(os.path.join(d, "train_games.csv")); last = tg[tg.env_steps > .9 * c["budget"]]
        rows.append(dict(train_setting=c["setting"]["name"], eval_setting=e["setting"], algo=c["algo"], mode=e["mode"], seed=c["seed"], score=e["score"], ret=e["return_mean"], success=e["success"], failure=e["failure"], timeout=e["timeout"], x=e["x"], reach_2311=e["reach_2311"], reach_6952=e["reach_6952"], reach_15693=e["reach_15693"],
                         len_success=g.decisions[ok].mean(), len_failure=g.decisions[bad].mean(), decisions=g.decisions.mean(), ghosts=g.ghosts.mean(), train_env_steps=c["env_steps"], train_minutes=c["train_minutes"], train_return_end=last.training_return.mean(), train_score_end=last.score.mean(), train_games=c["games"], run_dir=os.path.relpath(d, HERE)))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--runs", default=os.path.join(HERE, "records", "ad")); a = p.parse_args(); out = os.path.join(a.runs, "results"); os.makedirs(out, exist_ok=True)
    d = collect(a.runs); d.to_csv(os.path.join(out, "per_agent.csv"), index=False)
    keys = ["eval_setting", "train_setting", "algo", "mode"]; m = ["score", "ret", "success", "failure", "timeout", "len_success", "len_failure", "reach_2311", "reach_6952", "reach_15693", "ghosts", "train_env_steps", "train_minutes", "train_score_end"]
    s = d.groupby(keys)[m].agg(["mean", "std", "min", "max"]); s.columns = [f"{x}_{y}" for x, y in s.columns]; s["agents"] = d.groupby(keys).size(); s.round(3).to_csv(os.path.join(out, "summary.csv"))
    L = ["| Test | Trained in | Model | Policy | Agents | Score | Return | Success | Failure | Timeout | Length (success) | Length (failure) | Training steps | Training time (min) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda r, k, n=2: f"{r[k + '_mean']:.{n}f} ± {0 if np.isnan(r[k + '_std']) else r[k + '_std']:.{n}f}"
    for k, r in s.iterrows(): L.append(f"| {k[0]} | {k[1]} | {NAMES[k[2]]} | {k[3]} | {int(r['agents'])} | {f(r, 'score', 0)} | {f(r, 'ret', 0)} | {f(r, 'success')} | {f(r, 'failure')} | {f(r, 'timeout')} | {'-' if np.isnan(r['len_success_mean']) else f(r, 'len_success', 0)} | {'-' if np.isnan(r['len_failure_mean']) else f(r, 'len_failure', 0)} | {r['train_env_steps_mean']:.0f} | {f(r, 'train_minutes', 1)} |")
    L.append(f"\nMean ± standard deviation across trained agents (one value per agent, 200 held-out games each). Score and success use the game's real points. Return = points, +1,600 once when the score reaches X, and -0.5 per decision in C; it is not comparable between B and C. Success rate = share of an agent's 200 games with at least X points, then the mean over agents; success = at least {int(d.x.iloc[0])} points. Training returns use the guides of each model and are not comparable between models.")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L)); print("\n".join(L))
    for name in ("A", "B", "C"):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.3)); src = []
        for al, col in (("q", "tab:blue"), ("ppo", "tab:red")):
            dirs = sorted(glob.glob(os.path.join(a.runs, "runs", name, al, "seed*"))); cv = [pd.read_csv(os.path.join(x, "curve.csv")) for x in dirs if os.path.exists(os.path.join(x, "curve.csv"))]
            if not cv: continue
            n = min(map(len, cv)); src += [os.path.relpath(x, HERE) for x in dirs]
            for i, key in enumerate(("score", "success")):
                for c in cv: ax[i].plot(c.env_steps, c[key], color=col, lw=.6, alpha=.35)
                ax[i].plot(cv[0].env_steps[:n], np.mean([c[key].values[:n] for c in cv], 0), color=col, lw=2.2, label=f"{NAMES[al]} (mean of {len(cv)} agents)")
        for i, t in enumerate(("Score per test game", "Success rate")): ax[i].set_xlabel("Environment interactions"); ax[i].set_ylabel(t); ax[i].grid(alpha=.3); ax[i].legend(fontsize=8)
        fig.suptitle(f"Setting {name}: learning curves (every {int(d[d.train_setting == name].train_env_steps.mean() // 12 // 1000 * 1000)} interactions, 5 test games, no smoothing; thin lines = single agents)", fontsize=10)
        sd = sorted({int(x.rsplit('seed', 1)[1]) for x in src}); fig.text(.01, .005, "Source runs: " + ", ".join(sorted({x.rsplit('/', 1)[0] + f'/seed{sd[0]}-{sd[-1]}' for x in src})), fontsize=6.5, color="0.35"); fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(os.path.join(out, f"curves_{name}.png"), dpi=150); plt.close(fig)


if __name__ == "__main__": main()
