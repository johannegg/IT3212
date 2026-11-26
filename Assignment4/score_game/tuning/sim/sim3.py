import numpy as np, sys
from sim import problem, clip, D
from multiprocessing import Pool
def vary_r(r, p, lo, hi, n): x = p.copy(); k = r.choice(D, min(n, D), False); x[k] += r.choice([-1, 1], len(k)) * r.uniform(lo, hi, len(k)); return clip(x)
def levels(W, K):      # the decided mix (12 fine, 8 medium, 6 mix, 6 big of 32), scaled to W - K new ones
    n = W - K; f = round(n * 12 / 32); m = round(n * 8 / 32); x = round(n * 6 / 32); return f, m, x, n - f - m - x
def run(a):
    seed, kind, noise, W, K, rounds = a; r = np.random.default_rng(seed); f = problem(r, kind); X = r.random((W, D)); H = {}; res = []
    obs = lambda X, n=5: f(X) + r.normal(0, noise / np.sqrt(n), len(X))
    for t in range(rounds):
        y = obs(X); key = [x.tobytes() for x in X]
        for k, v in zip(key, y): H.setdefault(k, []).append(v)
        sc = np.array([np.mean(H[k]) for k in key]); o = np.argsort(-sc); best = []
        for i in o:
            if all((np.abs(X[i] - X[j]) >= .10).sum() >= 2 for j in best): best.append(i)
            if len(best) == K: break
        for i in o:
            if len(best) == K: break
            if i not in best: best.append(i)
        B = X[best]; n = min(4, K); fin = B[:n]; s = np.array([np.mean(H[x.tobytes()]) for x in fin]); z = (obs(fin) * 5 + s * 5) / 10; res.append(float(f(fin[[np.argmax(z)]])[0]))
        if t < rounds - 1:
            nf, nm, nx, nb = levels(W, K); out = list(B); p = lambda: B[r.integers(len(B))]
            out += [vary_r(r, p(), .05, .15, D) for _ in range(nf)] + [vary_r(r, p(), .10, .30, r.integers(2, 4)) for _ in range(nm)]
            for _ in range(nx): b, c = B[r.choice(len(B), 2, len(B) < 2)]; out.append(np.where(r.random(D) < .5, b, c))
            for _ in range(nb):
                x = p().copy(); k = r.choice(D, r.integers(1, 4), False); x[k] = r.random(len(k)); out.append(x)
            X = np.array(out[:W])
    return res
if __name__ == '__main__':
    N = 1500; plans = {'40/8': (40, 8, 4), '20/4': (20, 4, 8), '20/5': (20, 5, 8), '30/6': (30, 6, 5), '16/4': (16, 4, 10), '60/10': (60, 10, 3)}
    for kind in ('smooth', 'multi'):
        for noise in (.15, .35):
            print(f'== {kind}, noise {noise}: true score of the winner after a given number of settings tried (each with 5 seeds)')
            for nm, (W, K, R) in plans.items():
                with Pool(3) as p: M = np.array(p.map(run, [(1000 + i, kind, noise, W, K, R) for i in range(N)], chunksize=50)).mean(0)
                print(f'  {nm:6s}', '  '.join(f'{W * (t + 1):3d}: {M[t]:.3f}' for t in range(R)), flush=True)
