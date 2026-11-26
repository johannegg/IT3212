"""Tuning of Q-learning and PPO, each on its own, with the same number of rounds, settings, seeds and steps.

    python tune.py                          # keeps on with the next unfinished round until it stops
    python tune.py --rounds 1               # at most this many rounds in this call
    python tune.py --final                  # the 4 best of each model: 5 new seeds and new games -> winners.json
    python tune.py --quick                  # tiny run of the same loop (folder tuning_quick), only to check that it works
    touch tuning/STOP                       # stops after the runs that are in progress

Everything is saved in tuning/: results.jsonl (one line per run), round files and decisions.log. A stopped run continues from there.
Method (see the decisions file): round 1 = 18 random settings and 2 known ones. Later rounds = the 4 best with new seeds, 6 fine,
4 medium, 3 mix and 3 big changes of them. Score = average real score of the test games. Only Setting B, mazes 1-3 for training."""
import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from game.env import SETTINGS
from game.run import test_games, train

# ---- the 8 values of each model: limits are fixed ------------------------------------------------------------------------------
SPACE = {
    "q": [("life_penalty", 0, 500, "lin"), ("ghost_value", 0.25, 4, "log"), ("gamma", 0.8, 0.999, "gap"), ("alpha", 0.005, 0.5, "log"),
          ("eps_end", 0.005, 0.1, "log"), ("eps_frac", 0.1, 0.9, "lin"), ("lam", 0.0, 0.95, "lin"), ("q0", 0.0, 100.0, "lin")],
    "ppo": [("life_penalty", 0, 500, "lin"), ("ghost_value", 0.25, 4, "log"), ("gamma", 0.8, 0.999, "gap"), ("lr", 5e-5, 2e-3, "log"),
            ("ent_coef", 5e-4, 0.05, "log"), ("epochs", 2, 10, "int"), ("gae_lambda", 0.8, 0.99, "lin"), ("reward_scale", 1e-3, 0.1, "log")],
}
KNOWN = {   # Part 2's settings, and the setup of the earlier tuning (a ghost worth little), adapted to the new game
    "q": [dict(life_penalty=0, ghost_value=1.0, gamma=0.99, alpha=0.1, eps_end=0.05, eps_frac=0.5, lam=0.0, q0=0.0),
          dict(life_penalty=0, ghost_value=0.25, gamma=0.99, alpha=0.1, eps_end=0.05, eps_frac=0.5, lam=0.0, q0=0.0)],
    "ppo": [dict(life_penalty=0, ghost_value=1.0, gamma=0.99, lr=3e-4, ent_coef=0.01, epochs=4, gae_lambda=0.95, reward_scale=0.1),
            dict(life_penalty=0, ghost_value=0.25, gamma=0.99, lr=3e-4, ent_coef=0.01, epochs=4, gae_lambda=0.95, reward_scale=0.1)],
}
ALGOS = ["q", "ppo"]
WIDTH, KEEP, FINAL = 20, 4, 4
LEVELS = [("fine", 6), ("medium", 4), ("mix", 3), ("big", 3)]       # with the 4 kept: 20
BUDGET_MIN, BUDGET_MAX, BUDGET_STEP = 300_000, 300_000, 100_000       # kept at 300,000 (decision after round 1: a higher budget doubles the time)


def to_value(spec, u):
    name, lo, hi, kind = spec; u = float(min(1.0, max(0.0, u)))
    if kind == "lin": return lo + u * (hi - lo)
    if kind == "int": return int(round(lo + u * (hi - lo)))
    if kind == "log": return float(lo * (hi / lo) ** u)
    return float(1.0 - (1.0 - lo) * ((1.0 - hi) / (1.0 - lo)) ** u)      # "gap": the distance to 1 moves on a log scale, u = 0 is lo, u = 1 is hi


def to_unit(spec, v):
    name, lo, hi, kind = spec
    if kind in ("lin", "int"): return (v - lo) / (hi - lo)
    if kind == "log": return float(np.log(v / lo) / np.log(hi / lo))
    return float(np.log((1.0 - v) / (1.0 - lo)) / np.log((1.0 - hi) / (1.0 - lo)))


def hp_of(algo, u): return {s[0]: to_value(s, x) for s, x in zip(SPACE[algo], u)}
def unit_of(algo, hp): return [min(1.0, max(0.0, to_unit(s, hp[s[0]]))) for s in SPACE[algo]]


class Store:
    def __init__(self, folder): self.dir = folder; os.makedirs(folder, exist_ok=True); self.results = os.path.join(folder, "results.jsonl")
    def say(self, *a):
        s = time.strftime("%F %T ") + " ".join(str(x) for x in a); print(s, flush=True); open(os.path.join(self.dir, "decisions.log"), "a").write(s + "\n")
    def load(self): return [json.loads(l) for l in open(self.results)] if os.path.exists(self.results) else []
    def path(self, n): return os.path.join(self.dir, n)
    def read(self, n): return json.load(open(self.path(n))) if os.path.exists(self.path(n)) else None
    def write(self, n, d): json.dump(d, open(self.path(n), "w"), indent=1)
    def stopped(self): return os.path.exists(self.path("STOP"))


def job(a):
    algo, rnd, sid, hp, seed, budget, n_games, base = a
    for attempt in (1, 2):
        try:
            t = time.time(); agent, games, dt = train(algo, SETTINGS["B"], hp, seed, budget)
            r = test_games(algo, agent, SETTINGS["B"], [base + i for i in range(n_games)], stop_at_maze4=True); sc = [x["score"] for x in r]
            tr = np.array([[g[0], g[1]] for g in games]); q = [float(tr[(tr[:, 0] > budget * k / 4) & (tr[:, 0] <= budget * (k + 1) / 4), 1].mean()) if ((tr[:, 0] > budget * k / 4) & (tr[:, 0] <= budget * (k + 1) / 4)).any() else 0.0 for k in range(4)]
            return dict(algo=algo, round=rnd, sid=sid, seed=seed, budget=budget, hp=hp, score=float(np.mean(sc)), scores=sc, success=float(np.mean([s >= 6952 for s in sc])),
                        stuck=sum(x["stuck"] for x in r), decisions=float(np.mean([x["decisions"] for x in r])), ghosts=float(np.mean([x["ghosts"] for x in r])),
                        train_quarters=q, train_games=len(games), rows=len(getattr(agent, "ids", {})), minutes=(time.time() - t) / 60, train_minutes=dt / 60, when=time.strftime("%F %T"))
        except Exception: err = traceback.format_exc()
    return dict(algo=algo, round=rnd, sid=sid, seed=seed, budget=budget, hp=hp, error=err)


def run_batch(st, jobs, workers, check_random=False):
    done = {(r["algo"], r["round"], r["sid"], r["seed"]) for r in st.load() if "error" not in r}
    todo = [j for j in jobs if (j[0], j[1], j[2], j[4]) not in done]
    st.say(f"runs: {len(jobs)}, to do {len(todo)}"); n, ok = 0, 0
    if not todo: return
    with Pool(workers, maxtasksperchild=1) as p:
        for r in p.imap_unordered(job, todo, chunksize=1):
            open(st.results, "a").write(json.dumps(r) + "\n"); n += 1
            if "error" in r: st.say("ERROR", r["algo"], r["sid"], r["seed"], r["error"][-400:])
            if n % 40 == 0: st.say(f"{n} of {len(todo)} runs finished")
            if check_random and n == 10 and all(x.get("score", 0) < 350 for x in st.load()[-10:]):
                st.say("STOP: the first 10 runs are near random play (208 points), something is wrong"); p.terminate(); raise SystemExit(1)
            if st.stopped(): st.say("STOP file found: no new runs"); p.terminate(); raise SystemExit(2)


def new_settings(algo, rnd, kept, rng):
    """Units of the round's 40 settings: [(level, units)]."""
    n = len(SPACE[algo])
    if rnd == 1: return [("known", unit_of(algo, k)) for k in KNOWN[algo]] + [("random", list(rng.random(n))) for _ in range(WIDTH - len(KNOWN[algo]))]
    out = [("same", list(k)) for k in kept]
    pick = lambda: np.array(kept[rng.integers(len(kept))])
    for level, count in LEVELS:
        for _ in range(count):
            x = pick().copy()
            if level == "fine": x += rng.choice([-1, 1], n) * rng.uniform(.05, .15, n)
            elif level == "medium":
                k = rng.choice(n, int(rng.integers(2, 4)), replace=False); x[k] += rng.choice([-1, 1], len(k)) * rng.uniform(.10, .30, len(k))
            elif level == "mix":
                y = pick(); m = rng.random(n) < .5; x = np.where(m, x, y)
            elif level == "big": k = rng.choice(n, int(rng.integers(1, 4)), replace=False); x[k] = rng.random(len(k))
            out.append((level, list(np.clip(x, 0, 1))))
    return out


def sid_of(algo, rnd, i): return f"{algo}-r{rnd}-{i:02d}"


def analyse(st, algo, rnd, plan, cfg):
    """Average score of each setting of a round (its earlier results at the same budget are included for kept ones), ranking and the 8 best."""
    rows = [r for r in st.load() if "error" not in r and r["algo"] == algo]
    out = []
    for i, p in enumerate(plan):
        sid = p["sid"]; own = [r["score"] for r in rows if r["round"] == rnd and r["sid"] == sid and r["budget"] == cfg["budget"]]
        old = [r["score"] for r in rows if r["sid"] in p.get("history", []) and r["round"] < rnd and r["budget"] == cfg["budget"]]
        if own: out.append(dict(i=i, sid=sid, new=float(np.mean(own)), score=float(np.mean(own + old)), n=len(own + old), min=float(np.min(own)), level=p["level"], units=p["units"], history=p.get("history", []) + [sid], succ=float(np.mean([r["success"] for r in rows if r["sid"] == sid and r["round"] == rnd]))))
    out.sort(key=lambda d: -d["score"]); kept = []
    for d in out:
        if len(kept) < cfg["keep"] and all((np.abs(np.array(d["units"]) - np.array(k["units"])) >= .10).sum() >= 2 for k in kept): kept.append(d)
    for d in out:
        if len(kept) < cfg["keep"] and d not in kept: kept.append(d)
    return out, kept


def plan_round(st, cfg, rnd):
    f = f"plan_round{rnd}.json"; p = st.read(f)
    if p: return p
    prev = st.read(f"kept_round{rnd - 1}.json") if rnd > 1 else None; plan = {}
    for k, algo in enumerate(ALGOS):
        rng = np.random.default_rng(10_000 * rnd + k + (1 if cfg["quick"] else 0)); kept = [d["units"] for d in prev[algo]] if prev else None
        S = new_settings(algo, rnd, kept, rng); rows = []
        for i, (level, u) in enumerate(S): rows.append(dict(sid=sid_of(algo, rnd, i), level=level, units=u, hp=hp_of(algo, u), history=prev[algo][i]["history"] if (prev and level == "same") else []))
        plan[algo] = rows[:cfg["width"]] if not prev else rows
    st.write(f, plan); return plan


def seeds_of(rnd, cfg): return [1000 + 5 * (rnd - 1) + k for k in range(cfg["seeds"])]
def games_base(rnd): return 100_000_000 + 10_000 * rnd


def next_budget(st, kept, cfg):
    """300,000 and up: more steps when the 8 best are still improving in the last quarter of training (same for both models)."""
    imp = []
    for algo in ALGOS:
        rows = [r for r in st.load() if "error" not in r and r["algo"] == algo and r["sid"] in {k["sid"] for k in kept[algo]} and r["budget"] == cfg["budget"]]
        a, b = np.mean([r["train_quarters"][2] for r in rows]), np.mean([r["train_quarters"][3] for r in rows]); imp.append((b - a) / max(a, 1.0))
    m = max(imp); add = 3 * BUDGET_STEP if m > .15 else BUDGET_STEP if m >= .05 else 0
    nb = min(BUDGET_MAX, cfg["budget"] + add); st.say("improvement in the last quarter of training (q, ppo):", [round(100 * float(x), 1) for x in imp], "% -> next budget", nb)
    return nb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--rounds", type=int, default=99); ap.add_argument("--workers", type=int, default=15); ap.add_argument("--final", action="store_true")
    ap.add_argument("--until", default=None, help="HH:MM, do not start a new round after this time"); a = ap.parse_args()
    cfg = dict(quick=a.quick, width=WIDTH, keep=KEEP, seeds=1 if a.quick else 5, budget=5_000 if a.quick else BUDGET_MIN, games=2 if a.quick else 20)
    st = Store(os.path.join(HERE, "tuning_quick" if a.quick else "tuning"))
    state = st.read("state.json") or dict(round=1, budget=cfg["budget"], last_avg=None, finished=False)
    cfg["budget"] = state["budget"]
    if a.final: return final(st, cfg, a.workers, state)
    if state["finished"]: st.say("already finished (stop rule). Delete finished in state.json to continue."); return
    ran = 0
    while ran < a.rounds and not state["finished"]:
        if st.stopped(): st.say("STOP file found"); return
        if a.until and time.strftime("%H:%M") > a.until and ran: st.say("the time limit is reached: no new round"); return
        rnd = state["round"]; plan = plan_round(st, cfg, rnd); st.say(f"ROUND {rnd}: {len(plan['q'])} settings per model, {cfg['seeds']} seeds, budget {cfg['budget']}")
        jobs = [(al, rnd, p["sid"], p["hp"], s, cfg["budget"], cfg["games"], games_base(rnd)) for s in seeds_of(rnd, cfg) for i in range(WIDTH) for al in ALGOS for p in [plan[al][i]]]
        run_batch(st, jobs, a.workers, check_random=(rnd == 1 and not a.quick)); kept = {}; avg = {}
        for al in ALGOS:
            rank, kept[al] = analyse(st, al, rnd, plan[al], cfg); same = [d["new"] for d in rank if d["level"] == "same"]; avg[al] = float(np.mean(same)) if same else None
            st.say(f"round {rnd} {al}: the kept now average {np.mean([d["new"] for d in kept[al]]):.0f}; last round's 8 re-tested on new seeds: {"-" if avg[al] is None else f"{avg[al]:.0f}"}; best {rank[0]['score']:.0f}, median {np.median([d['score'] for d in rank]):.0f}, worst {rank[-1]['score']:.0f}"); st.write(f"ranking_round{rnd}_{al}.json", rank)
            for d in kept[al]: st.say(f"   {al} kept {d['sid']} ({d['level']}) score {d['score']:.0f} (n {d['n']}, lowest seed {d['min']:.0f}) success {d['succ']:.2f}", {k: round(v, 4) if isinstance(v, float) else v for k, v in hp_of(al, d['units']).items()})
        st.write(f"kept_round{rnd}.json", kept); ran += 1
        stop = False
        if state["last_avg"] and all(avg[al] is not None for al in ALGOS):      # fair comparison: re-tests of the kept settings on new seeds, in two rounds in a row
            gains = {al: (avg[al] - state["last_avg"][al]) / max(1.0, state["last_avg"][al]) for al in ALGOS}; st.say("improvement of the kept settings (re-tested on new seeds):", {k: f"{100 * v:.1f}%" for k, v in gains.items()}); stop = all(v < .05 for v in gains.values())
        nb = next_budget(st, kept, cfg) if not a.quick else cfg["budget"]
        state = dict(round=rnd + 1, budget=nb, last_avg=(avg if (nb == cfg["budget"] and all(v is not None for v in avg.values())) else None), finished=bool(stop)); cfg["budget"] = nb; st.write("state.json", state)
        if stop: st.say("STOP RULE: less than 5% improvement for both models"); break
    st.say("end of this call")


def final(st, cfg, workers, state):
    """The 4 best of each model by all results so far, 5 new seeds and new games -> the winners."""
    last = state["round"] - 1; kept = st.read(f"kept_round{last}.json"); st.say(f"FINAL: the best {FINAL} of each model after round {last}")
    jobs = [(al, 0, "final-" + k["sid"], hp_of(al, k["units"]), 9000 + s, cfg["budget"], cfg["games"], 300_000_000) for al in ALGOS for k in kept[al][:FINAL] for s in range(cfg["seeds"])]
    run_batch(st, jobs, workers); win = {}; rows = st.load()
    for al in ALGOS:
        tab = []
        for k in kept[al][:FINAL]:
            r = [x for x in rows if x["sid"] == "final-" + k["sid"] and "error" not in x]; tab.append((float(np.mean([x["score"] for x in r])), k["sid"], k["units"], float(np.std([x["score"] for x in r])), float(np.mean([x["success"] for x in r])), len(r)))
        tab.sort(reverse=True)
        for t in tab: st.say(f"   final {al} {t[1]}: score {t[0]:.0f} (sd between agents {t[3]:.0f}), success {t[4]:.2f}, agents {t[5]}")
        win[al] = dict(sid=tab[0][1], hp=hp_of(al, tab[0][2]), score=tab[0][0], budget=cfg["budget"])
    st.write("winners.json", win); st.say("WINNERS", json.dumps(win))


if __name__ == "__main__": main()
