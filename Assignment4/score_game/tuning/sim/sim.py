"""Simulated tuning loop: 8 values in [0,1] (share of each range), 40 settings, keep 8, noisy scores. Compare ways of making the next 40. Result = TRUE score (0-1) of the final winner."""
import numpy as np, sys
from multiprocessing import Pool
D = 8
def problem(r, kind):
    opt = r.uniform(.1, .9, D); w = r.dirichlet(np.ones(D) * .7) ; 
    if kind == 'smooth':      # one good area; a few values matter a lot, and some values work together
        A = np.diag(w * 8); i, j = r.choice(D, 2, False); A[i, j] = A[j, i] = .6 * np.sqrt(A[i, i] * A[j, j]); width = .35
        return lambda X: np.exp(-np.einsum('ni,ij,nj->n', X - opt, A, X - opt) / (2 * width ** 2 * 8 / 4))
    c = r.uniform(.05, .95, (5, D)); h = np.r_[1., r.uniform(.5, .85, 4)]; wd = r.uniform(.12, .3, 5)   # several areas, only one is best
    return lambda X: np.max(h * np.exp(-((((X[:, None] - c) ** 2) * (w * 8)).sum(2)) / (2 * (wd * 2.0) ** 2)), 1)
def clip(x): return np.clip(x, 0, 1)
def vary(r, p, size, nval):     # nval = number of values changed (or D), size = share of the range, random direction
    x = p.copy(); k = r.choice(D, min(nval, D), False); x[k] += r.choice([-1, 1], len(k)) * size; return clip(x)
def big(r, p, n): x = p.copy(); k = r.choice(D, n, False); x[k] = r.random(len(k)); return x
def make(r, best, plan):
    out = [b for b in best]; pick = lambda: best[r.integers(len(best))]
    for kind, n, *a in plan:
        for _ in range(n):
            if kind == 'fine': out.append(vary(r, pick(), a[0], D))
            elif kind == 'med': out.append(vary(r, pick(), a[0], r.integers(a[1], a[2] + 1)))
            elif kind == 'big': out.append(big(r, pick(), r.integers(1, 4)))
            elif kind == 'new': out.append(r.random(D))
            elif kind == 'mix': b, c = best[r.choice(len(best), 2, False)]; out.append(np.where(r.random(D) < .5, b, c))
            elif kind == 'de': b, c, d = best[r.choice(len(best), 3, False)]; out.append(clip(np.where(r.random(D) < .5, b + .5 * (c - d), b)))
    return np.array(out)
def run(a):
    seed, kind, noise, plan, rounds = a; r = np.random.default_rng(seed); f = problem(r, kind); X = r.random((40, D)); H = {}
    res = []
    def obs(X, n=5): return f(X) + r.normal(0, noise / np.sqrt(n), len(X))
    for t in range(rounds):
        y = obs(X)
        key = [x.tobytes() for x in X]; 
        for k, v in zip(key, y): H.setdefault(k, []).append(v)
        score = np.array([np.mean(H[k]) for k in key]); o = np.argsort(-score); best = []
        for i in o:
            if all((np.abs(X[i] - X[j]) >= .10).sum() >= 2 for j in best): best.append(i)
            if len(best) == 8: break
        for i in o:
            if len(best) == 8: break
            if i not in best: best.append(i)
        B = X[best]
        fin = B[:4]; s = np.array([np.mean(H[x.tobytes()]) for x in fin]); z = (obs(fin) * 5 + s * 5) / 10; res.append(float(f(fin[[np.argmax(z)]])[0]))     # final: 5 new seeds each, combined
        if t < rounds - 1: X = make(r, B, plan)
    return res
PLANS = {
 'A suggested: PBT steps (fine .02, med .05)': [('fine', 10, .02), ('med', 12, .045, 2, 3), ('big', 8), ('new', 2)],
 'B wider steps (fine .05, med .10)': [('fine', 10, .05), ('med', 12, .10, 2, 3), ('big', 8), ('new', 2)],
 'C wider (fine .10, med .20)': [('fine', 10, .10), ('med', 12, .20, 2, 3), ('big', 8), ('new', 2)],
 'D very wide (fine .15, med .30)': [('fine', 10, .15), ('med', 12, .30, 2, 3), ('big', 8), ('new', 2)],
 'E as C, med changes 1-2 values': [('fine', 10, .10), ('med', 12, .20, 1, 2), ('big', 8), ('new', 2)],
 'F as C, med changes 3-5 values': [('fine', 10, .10), ('med', 12, .20, 3, 5), ('big', 8), ('new', 2)],
 'G more fine: fine 16, med 8, big 6, new 2': [('fine', 16, .10), ('med', 8, .20, 2, 3), ('big', 6), ('new', 2)],
 'H more medium: fine 4, med 20, big 6, new 2': [('fine', 4, .10), ('med', 20, .20, 2, 3), ('big', 6), ('new', 2)],
 'I more random: fine 8, med 8, big 8, new 8': [('fine', 8, .10), ('med', 8, .20, 2, 3), ('big', 8), ('new', 8)],
 'J no fully random: fine 10, med 14, big 8': [('fine', 10, .10), ('med', 14, .20, 2, 3), ('big', 8)],
 'K mix in: fine 8, med 10, mix 6, big 6, new 2': [('fine', 8, .10), ('med', 10, .20, 2, 3), ('mix', 6), ('big', 6), ('new', 2)],
 'L only random (no loop)': [('new', 32)],
 'M differential evolution 24 + 8 new': [('de', 24), ('new', 8)],
 'N mix 8, med 16, big 8': [('mix', 8), ('med', 16, .20, 2, 3), ('big', 8)],
}
if __name__ == '__main__':
    N = 1500
    for kind in ('smooth', 'several areas'):
        for noise in (.15, .35):        # sd of one seed, share of the best score
            print(f'== {kind}, noise per seed {noise}', flush=True); rows = []
            for nm, pl in PLANS.items():
                with Pool(15) as p: R = np.array(p.map(run, [(1000 + i, 'smooth' if kind == 'smooth' else 'multi', noise, pl, 4) for i in range(N)], chunksize=20))
                rows.append((nm, R.mean(0), R[:, 1].std() / np.sqrt(N)))
            for nm, m, se in sorted(rows, key=lambda x: -x[1][1]): print(f'  {nm:45s} after round 1/2/3/4: ' + ' / '.join(f'{v:.3f}' for v in m) + f'   (±{se:.3f})')
