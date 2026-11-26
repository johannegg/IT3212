"""Tuning of Q-learning and PPO, each on its own, with the same number of rounds, settings, seeds and steps.

    python tune.py                          # keeps on with the next unfinished round until it stops
    python tune.py --rounds 1               # at most this many rounds in this call
    python tune.py --final                  # picks the winner of each model: the best average score in the last round -> winners.json
    python tune.py --quick                  # tiny run of the same loop (folder tuning_quick), only to check that it works
    touch tuning/STOP                       # stops after the runs that are in progress

Everything is saved in tuning/: results.jsonl (one line per run), round files and decisions.log. A stopped run continues from there.
Method (see the decisions file): round 1 = 18 random settings and 2 known ones. Rounds 2-3 = the 4 best with new seeds, 6 fine,
4 medium, 3 mix and 3 big changes of them, within the first limits (SPACE_OLD). From round 4 (CMA_FROM): the 4 best with new seeds
and 16 settings drawn by CMA-ES (Hansen 2016, arXiv:1604.00772) within wider limits (SPACE). The CMA-ES cloud starts from the best
half of round 3, only the 16 drawn settings update it, a draw outside the limits is drawn again (Hansen, appendix B.5), and a
setting whose runs all failed is ranked last. Score = average real score of the test games. Only Setting B, mazes 1-3 for training."""
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

# ---- the 8 values of each model ---------------------------------------------------------------------------------------------------
# Rounds 1-3 used these limits. Their plan and kept files store positions ("units") in this space.
SPACE_OLD = {
    "q": [("life_penalty", 0, 500, "lin"), ("ghost_value", 0.25, 4, "log"), ("gamma", 0.8, 0.999, "gap"), ("alpha", 0.005, 0.5, "log"),
          ("eps_end", 0.005, 0.1, "log"), ("eps_frac", 0.1, 0.9, "lin"), ("lam", 0.0, 0.95, "lin"), ("q0", 0.0, 100.0, "lin")],
    "ppo": [("life_penalty", 0, 500, "lin"), ("ghost_value", 0.25, 4, "log"), ("gamma", 0.8, 0.999, "gap"), ("lr", 5e-5, 2e-3, "log"),
            ("ent_coef", 5e-4, 0.05, "log"), ("epochs", 2, 10, "int"), ("gae_lambda", 0.8, 0.99, "lin"), ("reward_scale", 1e-3, 0.1, "log")],
}
# From round 4. Several best settings sat on a limit after round 3, so the limits were widened: hard limits where a value has no
# meaning past them, otherwise a buffer (5-10 times past the old edge) where a run would only crash or waste time.
SPACE = {
    "q": [("life_penalty", 1, 5000, "log"), ("ghost_value", 0.01, 20, "log"), ("gamma", 0.5, 0.9999, "gap"), ("alpha", 1e-4, 1.0, "log"),
          ("eps_end", 1e-4, 0.3, "log"), ("eps_frac", 0.01, 1.0, "lin"), ("lam", 0.0, 0.99, "lin"), ("q0", 0.0, 1000.0, "lin")],
    "ppo": [("life_penalty", 1, 5000, "log"), ("ghost_value", 0.01, 20, "log"), ("gamma", 0.5, 0.9999, "gap"), ("lr", 1e-6, 5e-3, "log"),
            ("ent_coef", 1e-5, 0.1, "log"), ("epochs", 1, 15, "int"), ("gae_lambda", 0.5, 0.999, "gap"), ("reward_scale", 1e-4, 1.0, "log")],
}
KNOWN = {   # Part 2's settings, and the setup of the earlier tuning (a ghost worth little), adapted to the new game
    "q": [dict(life_penalty=0, ghost_value=1.0, gamma=0.99, alpha=0.1, eps_end=0.05, eps_frac=0.5, lam=0.0, q0=0.0),
          dict(life_penalty=0, ghost_value=0.25, gamma=0.99, alpha=0.1, eps_end=0.05, eps_frac=0.5, lam=0.0, q0=0.0)],
    "ppo": [dict(life_penalty=0, ghost_value=1.0, gamma=0.99, lr=3e-4, ent_coef=0.01, epochs=4, gae_lambda=0.95, reward_scale=0.1),
            dict(life_penalty=0, ghost_value=0.25, gamma=0.99, lr=3e-4, ent_coef=0.01, epochs=4, gae_lambda=0.95, reward_scale=0.1)],
}
ALGOS = ["q", "ppo"]
WIDTH, KEEP = 20, 4
CMA_FROM = 4            # first round drawn by CMA-ES
STOP_GAIN = 0.10        # stop when the re-tested 4 kept improve by less than this for both models
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


def space(rnd): return SPACE_OLD if rnd < CMA_FROM else SPACE
def hp_of(algo, u, rnd=1): return {s[0]: to_value(s, x) for s, x in zip(space(rnd)[algo], u)}
def unit_of(algo, hp, rnd=1): return [min(1.0, max(0.0, to_unit(s, hp[s[0]]))) for s in space(rnd)[algo]]
def hp_entry(algo, d, rnd):
    """The real values of a ranked or kept setting. Rounds before CMA_FROM stored only units in the old space."""
    return dict(d["hp"]) if "hp" in d else hp_of(algo, d["units"], rnd)


# ---- CMA-ES (Hansen 2016, arXiv:1604.00772, default parameters for 16 drawn settings) ---------------------------------------
DRAW = WIDTH - KEEP


def cma_weights(n): w = np.log(n + .5) - np.log(np.arange(1, n + 1)); return w / w.sum()


def cma_consts(n_dim):
    mu = DRAW // 2; w = cma_weights(mu); mueff = 1 / float((w ** 2).sum()); D = n_dim
    cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D); cs = (mueff + 2) / (D + mueff + 5)
    c1 = 2 / ((D + 1.3) ** 2 + mueff); cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
    ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs; chi = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D * D))
    return dict(mu=mu, w=w, mueff=mueff, cc=cc, cs=cs, c1=c1, cmu=cmu, ds=ds, chi=chi)


def cma_min_spread(algo):
    """At least one whole step for whole-number values (epochs), so rounding does not freeze them."""
    return np.array([1.0 / (hi - lo) if kind == "int" else 0.0 for _, lo, hi, kind in SPACE[algo]])


def cma_floor(algo, cl):
    C = np.array(cl["C"]); m = cma_min_spread(algo)
    for j in range(len(m)):
        cur = cl["s"] * np.sqrt(C[j, j])
        if m[j] > 0 and cur < m[j]: f = m[j] / cur; C[j, :] *= f; C[:, j] *= f
    cl["C"] = C.tolist(); return cl


def cma_start(algo, ranking_prev, rnd_prev):
    """Warm start from the best half (10) of the previous round, in the new units: weighted centre and spread."""
    seen, E = set(), []
    for d in ranking_prev:
        hp = hp_entry(algo, d, rnd_prev); key = json.dumps({k: round(float(v), 10) for k, v in hp.items()}, sort_keys=True)
        if key in seen: continue
        seen.add(key); E.append(unit_of(algo, hp, CMA_FROM))
        if len(E) == WIDTH // 2: break
    E = np.array(E); w = cma_weights(len(E)); m = (w[:, None] * E).sum(0); sd = np.sqrt((w[:, None] * (E - m) ** 2).sum(0))
    s = max(float(sd.mean()), .05); C = np.diag((np.maximum(sd, .02) / s) ** 2); n = len(m)
    return cma_floor(algo, dict(m=m.tolist(), s=s, C=C.tolist(), pc=[0.0] * n, ps=[0.0] * n, g=0, Y=None, sids=None))


def cma_update(algo, cl, scores):
    """One CMA-ES update from the drawn settings of the last round. scores: same order as cl['Y']; a failed setting is -inf."""
    k = cma_consts(len(cl["m"])); Y = np.array(cl["Y"]); o = np.argsort(-np.array(scores), kind="stable")[:k["mu"]]; Yb = Y[o]
    m, s, C, pc, ps = np.array(cl["m"]), cl["s"], np.array(cl["C"]), np.array(cl["pc"]), np.array(cl["ps"]); g = cl["g"] + 1
    yw = (k["w"][:, None] * Yb).sum(0); m = m + s * yw
    ev, B = np.linalg.eigh(C); ev = np.maximum(ev, 1e-12); Cis = B @ np.diag(ev ** -.5) @ B.T
    ps = (1 - k["cs"]) * ps + np.sqrt(k["cs"] * (2 - k["cs"]) * k["mueff"]) * Cis @ yw
    hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - k["cs"]) ** (2 * g)) < (1.4 + 2 / (len(m) + 1)) * k["chi"]
    pc = (1 - k["cc"]) * pc + hs * np.sqrt(k["cc"] * (2 - k["cc"]) * k["mueff"]) * yw
    C = (1 - k["c1"] - k["cmu"]) * C + k["c1"] * np.outer(pc, pc) + k["cmu"] * (k["w"][:, None, None] * np.einsum("ni,nj->nij", Yb, Yb)).sum(0)
    s = float(s * np.exp((k["cs"] / k["ds"]) * (np.linalg.norm(ps) / k["chi"] - 1)))
    return cma_floor(algo, dict(m=m.tolist(), s=s, C=C.tolist(), pc=pc.tolist(), ps=ps.tolist(), g=g, Y=None, sids=None))


def cma_draw(cl, rng, n):
    """n settings from the cloud. A draw outside 0..1 in any value is drawn again (Hansen, appendix B.5)."""
    m, s, C = np.array(cl["m"]), cl["s"], np.array(cl["C"]); ev, B = np.linalg.eigh(C); A = B @ np.diag(np.sqrt(np.maximum(ev, 0)))
    Y, X, redraws = [], [], 0
    for _ in range(n):
        for t in range(1000):
            y = A @ rng.standard_normal(len(m)); x = m + s * y
            if np.all((x >= 0) & (x <= 1)): break
            redraws += 1
        else: x = np.clip(x, 0, 1); y = (x - m) / s      # never seen in the checks; logged by the caller through redraws
        Y.append(y.tolist()); X.append(x.tolist())
    return Y, X, redraws


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
    """Average score of each setting of a round (its earlier results at the same budget are included for kept ones), ranking and the 4 kept.
    Before CMA_FROM the kept also had to differ from each other (rule 6.12), because they were the parents of the next round.
    From CMA_FROM the cloud makes the new settings, so the kept are simply the 4 best by average score."""
    rows = [r for r in st.load() if "error" not in r and r["algo"] == algo]
    out = []
    for i, p in enumerate(plan):
        sid = p["sid"]; own = [r["score"] for r in rows if r["round"] == rnd and r["sid"] == sid and r["budget"] == cfg["budget"]]
        old = [r["score"] for r in rows if r["sid"] in p.get("history", []) and r["round"] < rnd and r["budget"] == cfg["budget"]]
        if own: out.append(dict(i=i, sid=sid, new=float(np.mean(own)), score=float(np.mean(own + old)), n=len(own + old), min=float(np.min(own)), level=p["level"], units=p["units"], hp=p["hp"], history=p.get("history", []) + [sid], succ=float(np.mean([r["success"] for r in rows if r["sid"] == sid and r["round"] == rnd]))))
        elif rnd >= CMA_FROM: out.append(dict(i=i, sid=sid, new=float("-inf"), score=float("-inf"), n=0, min=float("-inf"), level=p["level"], units=p["units"], hp=p["hp"], history=p.get("history", []) + [sid], succ=0.0, failed=True))
    out.sort(key=lambda d: -d["score"]); kept = []
    for d in out:
        if rnd >= CMA_FROM: break
        if len(kept) < cfg["keep"] and all((np.abs(np.array(d["units"]) - np.array(k["units"])) >= .10).sum() >= 2 for k in kept): kept.append(d)
    for d in out:
        if len(kept) < cfg["keep"] and d not in kept and not d.get("failed"): kept.append(d)
    return out, kept


def plan_round_cma(st, cfg, rnd):
    """The 4 kept with their exact values, and 16 settings drawn from the CMA-ES cloud. The cloud is saved as cma_round{rnd}.json."""
    prev = st.read(f"kept_round{rnd - 1}.json"); plan = {}; clouds = {}
    for k, algo in enumerate(ALGOS):
        rng = np.random.default_rng(10_000 * rnd + k + (1 if cfg["quick"] else 0))
        if rnd == CMA_FROM: cl = cma_start(algo, st.read(f"ranking_round{rnd - 1}_{algo}.json"), rnd - 1)
        else:
            old = st.read(f"cma_round{rnd - 1}.json")[algo]; rank = {d["sid"]: d["score"] for d in st.read(f"ranking_round{rnd - 1}_{algo}.json")}
            cl = cma_update(algo, old, [rank.get(sid, float("-inf")) for sid in old["sids"]])
        rows = []
        for i, d in enumerate(prev[algo]):
            hp = hp_entry(algo, d, rnd - 1); rows.append(dict(sid=sid_of(algo, rnd, i), level="same", units=unit_of(algo, hp, rnd), hp=hp, history=d["history"]))
        Y, X, redraws = cma_draw(cl, rng, DRAW)
        for y, x in zip(Y, X):
            i = len(rows); rows.append(dict(sid=sid_of(algo, rnd, i), level="cma", units=x, hp=hp_of(algo, x, rnd), history=[]))
        cl.update(Y=Y, sids=[r["sid"] for r in rows[KEEP:]], redraws=redraws); clouds[algo] = cl; plan[algo] = rows
        st.say(f"   {algo} cloud: overall step {cl['s']:.3f}, spread per value", [round(cl["s"] * float(np.sqrt(c)), 3) for c in np.diag(np.array(cl["C"]))], f"draws again at the limits: {redraws}")
    st.write(f"cma_round{rnd}.json", clouds); st.write(f"plan_round{rnd}.json", plan); return plan


def plan_round(st, cfg, rnd):
    f = f"plan_round{rnd}.json"; p = st.read(f)
    if p: return p
    if rnd >= CMA_FROM: return plan_round_cma(st, cfg, rnd)
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
            st.say(f"round {rnd} {al}: the kept now average {np.mean([d["new"] for d in kept[al]]):.0f}; last round's 4 re-tested on new seeds: {"-" if avg[al] is None else f"{avg[al]:.0f}"}; best {rank[0]['score']:.0f}, median {np.median([d['score'] for d in rank]):.0f}, worst {rank[-1]['score']:.0f}"); st.write(f"ranking_round{rnd}_{al}.json", rank)
            for d in kept[al]: st.say(f"   {al} kept {d['sid']} ({d['level']}) score {d['score']:.0f} (n {d['n']}, lowest seed {d['min']:.0f}) success {d['succ']:.2f}", {k: round(v, 4) if isinstance(v, float) else v for k, v in d['hp'].items()})
        st.write(f"kept_round{rnd}.json", kept); ran += 1
        stop = False
        if state["last_avg"] and all(avg[al] is not None for al in ALGOS):      # fair comparison: re-tests of the kept settings on new seeds, in two rounds in a row
            gains = {al: (avg[al] - state["last_avg"][al]) / max(1.0, state["last_avg"][al]) for al in ALGOS}; st.say("improvement of the kept settings (re-tested on new seeds):", {k: f"{100 * v:.1f}%" for k, v in gains.items()}); stop = all(v < STOP_GAIN for v in gains.values())
        nb = next_budget(st, kept, cfg) if not a.quick else cfg["budget"]
        state = dict(round=rnd + 1, budget=nb, last_avg=(avg if (nb == cfg["budget"] and all(v is not None for v in avg.values())) else None), finished=bool(stop)); cfg["budget"] = nb; st.write("state.json", state)
        if stop: st.say(f"STOP RULE: less than {100 * STOP_GAIN:.0f}% improvement for both models"); break
    st.say("end of this call")


def final(st, cfg, workers, state):
    """The winner of each model = the setting with the highest average score in the last round (kept ones include their earlier
    seeds). No runs: its honest score comes from Setting B in A-D, on new seeds and held-out games."""
    last = state["round"] - 1; st.say(f"WINNER: the best average score of each model in round {last}"); rows = [r for r in st.load() if "error" not in r]; win = {}
    for al in ALGOS:
        d = [x for x in st.read(f"ranking_round{last}_{al}.json") if not x.get("failed")][0]; hp = hp_entry(al, d, last)
        games = [g for r in rows if r["algo"] == al and r["sid"] in d["history"] and r["budget"] == cfg["budget"] for g in r["scores"]]
        win[al] = dict(sid=d["sid"], hp=hp, score=d["score"], seeds=d["n"], budget=cfg["budget"], share_at_6952=float(np.mean([g >= 6952 for g in games])), games=len(games))
        st.say(f"   winner {al} {d['sid']}: score {d['score']:.0f} over {d['n']} seeds, {100 * win[al]['share_at_6952']:.0f}% of {len(games)} tuning test games reach 6,952", hp)
    st.write("winners.json", win); st.say("WINNERS", json.dumps(win))


if __name__ == "__main__": main()
