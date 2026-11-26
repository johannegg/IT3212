import numpy as np, sim
from sim import *
from multiprocessing import Pool
def vary_r(r, p, lo, hi, nval):     # each changed value gets its own random step between lo and hi
    x = p.copy(); k = r.choice(D, min(nval, D), False); x[k] += r.choice([-1, 1], len(k)) * r.uniform(lo, hi, len(k)); return clip(x)
def make2(r, best, plan):
    out = [b for b in best]; pick = lambda: best[r.integers(len(best))]
    for kind, n, *a in plan:
        for _ in range(n):
            if kind == 'fineR': out.append(vary_r(r, pick(), a[0], a[1], D))
            elif kind == 'medR': out.append(vary_r(r, pick(), a[0], a[1], r.integers(2, 4)))
            elif kind == 'gauss': x = pick().copy(); out.append(clip(x + r.normal(0, a[0], D)))
            elif kind == 'largeR': out.append(vary_r(r, pick(), a[0], a[1], D))     # all values, big steps from a good one
            elif kind == 'half': out.append(big(r, pick(), r.integers(4, 7)))        # 4-6 values new
            else: out += list(make(r, best, [(kind, 1, *a)])[len(best):])
    return np.array(out)
sim.make = make2
P2 = {
 '1 last suggestion (K)': [('fine', 8, .10), ('med', 10, .20, 2, 3), ('mix', 6), ('big', 6), ('new', 2)],
 '2 random steps: fine 5-15%, med 10-30%': [('fineR', 8, .05, .15), ('medR', 10, .10, .30), ('mix', 6), ('big', 6), ('new', 2)],
 '3 random steps: fine 2-10%, med 10-30%': [('fineR', 8, .02, .10), ('medR', 10, .10, .30), ('mix', 6), ('big', 6), ('new', 2)],
 '4 as 2, all-new replaced by all values +-20-50%': [('fineR', 8, .05, .15), ('medR', 10, .10, .30), ('mix', 6), ('big', 6), ('largeR', 2, .2, .5)],
 '5 as 2, all-new replaced by 4-6 new values': [('fineR', 8, .05, .15), ('medR', 10, .10, .30), ('mix', 6), ('big', 6), ('half', 2)],
 '6 as 2, all-new replaced by 2 more big': [('fineR', 8, .05, .15), ('medR', 10, .10, .30), ('mix', 6), ('big', 8)],
 '7 as 6, more fine: fine 12, med 8, mix 6, big 6': [('fineR', 12, .05, .15), ('medR', 8, .10, .30), ('mix', 6), ('big', 6)],
 '8 as 6, more fine 5-10%: fine 12': [('fineR', 12, .03, .10), ('medR', 8, .10, .30), ('mix', 6), ('big', 6)],
 '9 as 6, two fine levels (2-5% x4, 5-15% x8)': [('fineR', 4, .02, .05), ('fineR', 8, .05, .15), ('medR', 8, .10, .30), ('mix', 6), ('big', 6)],
 '10 normal-distributed steps instead (sd 10%) x12': [('gauss', 12, .10), ('medR', 8, .10, .30), ('mix', 6), ('big', 6)],
 '11 no big at all: fine 12, med 14, mix 6': [('fineR', 12, .05, .15), ('medR', 14, .10, .30), ('mix', 6)],
}
def job(a): return sim.run(a)
if __name__ == '__main__':
    N = 3000
    for kind in ('smooth', 'multi'):
        for noise in (.15, .35):
            base = None; rows = []
            for nm, pl in P2.items():
                with Pool(15) as p: R = np.array(p.map(job, [(1000 + i, kind, noise, pl, 6) for i in range(N)], chunksize=30))
                base = R if base is None else base; rows.append((nm, R.mean(0), (R[:, 1:] - base[:, 1:]).mean(0), (R[:, 3] - base[:, 3]).std() / np.sqrt(N)))
            print(f'== {kind}, noise {noise}: true score after round 2/3/4/6, and difference to plan 1 at round 4')
            for nm, m, d, se in rows: print(f'  {nm:50s} {m[1]:.3f} {m[2]:.3f} {m[3]:.3f} {m[5]:.3f}   diff {d[2]:+.4f} (+-{se:.4f})', flush=True)
