"""Hybrid: round 1 wide (40), then 20/4 rounds, against 20/4 from the start and 40/8. Counts settings tried."""
import numpy as np
from sim3 import *
def run2(a):
    seed, kind, noise, seq, Ks, _ = a; r = np.random.default_rng(seed); f = problem(r, kind); X = r.random((seq[0], D)); H = {}; res = []
    obs = lambda X, n=5: f(X) + r.normal(0, noise / np.sqrt(n), len(X))
    for t, W in enumerate(seq):
        K = Ks[t]; y = obs(X); key = [x.tobytes() for x in X]
        for k, v in zip(key, y): H.setdefault(k, []).append(v)
        sc = np.array([np.mean(H[k]) for k in key]); o = np.argsort(-sc); best = []
        for i in o:
            if all((np.abs(X[i] - X[j]) >= .10).sum() >= 2 for j in best): best.append(i)
            if len(best) == K: break
        for i in o:
            if len(best) == K: break
            if i not in best: best.append(i)
        B = X[best]; fin = B[:4]; s = np.array([np.mean(H[x.tobytes()]) for x in fin]); z = (obs(fin) * 5 + s * 5) / 10; res.append(float(f(fin[[np.argmax(z)]])[0]))
        if t < len(seq) - 1:
            W2 = seq[t + 1]; nf, nm, nx, nb = levels(W2, K); out = list(B); p = lambda: B[r.integers(len(B))]
            out += [vary_r(r, p(), .05, .15, D) for _ in range(nf)] + [vary_r(r, p(), .10, .30, r.integers(2, 4)) for _ in range(nm)]
            for _ in range(nx): b, c = B[r.choice(len(B), 2, len(B) < 2)]; out.append(np.where(r.random(D) < .5, b, c))
            for _ in range(nb): x = p().copy(); k = r.choice(D, r.integers(1, 4), False); x[k] = r.random(len(k)); out.append(x)
            X = np.array(out[:W2])
    return res
if __name__ == '__main__':
    N = 1500; P = {'20/4 all': ([20] * 8, [4] * 8), '40/8 all': ([40] * 4, [8] * 4), '40 first, keep 4, then 20/4': ([40] + [20] * 6, [4] * 7), '40 first, keep 8, then 20/4': ([40] + [20] * 6, [8] + [4] * 6)}
    for kind in ('smooth', 'multi'):
        print('==', kind, 'noise .25: winner after n settings tried')
        for nm, (seq, Ks) in P.items():
            with Pool(3) as p: M = np.array(p.map(run2, [(5000 + i, kind, .25, seq, Ks, 0) for i in range(N)], chunksize=50)).mean(0)
            print(f'  {nm:32s}', '  '.join(f'{sum(seq[:t + 1]):3d}: {M[t]:.3f}' for t in range(len(seq))), flush=True)
