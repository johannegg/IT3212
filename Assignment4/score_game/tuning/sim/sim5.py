"""After 3 rounds of our current 20/4 loop, compare ways to run rounds 4-6.
Values are in 'share of the original range' (0..1). Each value has its own width of good area (narrow to wide, x12).
'edge' problems put the best value of 2 of the 8 values outside 0..1 (at -0.35 or 1.35); 'inside' keeps all inside.
Result = TRUE score of the final winner (4 best, 5 new seeds), after rounds 3,4,5,6."""
import numpy as np, sys
from multiprocessing import Pool
D = 8; W, K = 20, 4

def problem(r, kind):
    opt = r.uniform(.1, .9, D)
    if kind in ('edge', 'multi-edge'):
        k = r.choice(D, 2, False); opt[k] = r.choice([-.2, 1.2], 2)
    w = np.exp(r.uniform(np.log(.08), np.log(1.0), D))          # width of the good area per value
    if kind.startswith('multi'):                                  # 3 other good areas, lower (0.5-0.85 of the best)
        c = r.uniform(0, 1, (3, D)); h = r.uniform(.5, .85, 3)
        return lambda X: np.maximum(np.exp(-0.5 * (((X - opt) / w) ** 2).sum(-1) / 2),
                                    np.max(h * np.exp(-0.5 * (((X[..., None, :] - c) / w) ** 2).sum(-1) / 2), -1))
    return lambda X: np.exp(-0.5 * (((X - opt) / w) ** 2).sum(-1) / 2)

def select(X, sc, n):
    o = np.argsort(-sc); best = []
    for i in o:
        if all((np.abs(X[i] - X[j]) >= .10).sum() >= 2 for j in best): best.append(i)
        if len(best) == n: break
    for i in o:
        if len(best) == n: break
        if i not in best: best.append(i)
    return best

def current_gen(r, B, lo, hi):
    out = list(B); p = lambda: B[r.integers(len(B))]; span = hi - lo
    def vary(x, a, b, n):
        x = x.copy(); k = r.choice(D, n, False); x[k] += r.choice([-1, 1], n) * r.uniform(a, b, n) * span[k]; return np.clip(x, lo, hi)
    out += [vary(p(), .05, .15, D) for _ in range(6)] + [vary(p(), .10, .30, r.integers(2, 4)) for _ in range(4)]
    for _ in range(3): b, c = B[r.choice(len(B), 2, False)]; out.append(np.where(r.random(D) < .5, b, c))
    for _ in range(3): x = p().copy(); k = r.choice(D, r.integers(1, 4), False); x[k] = r.uniform(lo[k], hi[k]); out.append(x)
    return np.array(out)

def logw(n): w = np.log(n + .5) - np.log(np.arange(1, n + 1)); return w / w.sum()

def run(a):
    seed, kind, method, factor = a; r = np.random.default_rng(seed); f = problem(r, kind); H = {}
    noise = float(__import__("os").environ.get("NOISE", ".25")); obs = lambda X: f(X) + r.normal(0, noise / np.sqrt(5), len(X))
    lo, hi = np.zeros(D), np.ones(D)
    X = r.random((W, D)); res = []; sig = None; prev_mean = None; prev_dir = None
    # CMA state
    cm = None
    for t in range(6):
        y = obs(X); key = [x.tobytes() for x in X]
        for k, v in zip(key, y): H.setdefault(k, []).append(v)
        sc = np.array([np.mean(H[k]) for k in key])
        best = select(X, sc, K); B = X[best]
        fin = B[:4]; s = np.array([np.mean(H[x.tobytes()]) for x in fin]); z = (obs(fin) * 5 + s * 5) / 10
        if t >= 2: res.append(float(f(fin[[np.argmax(z)]])[0]))
        if t == 5: break
        # unique settings of this round, best half
        u, ui = np.unique(X, axis=0, return_index=True); su = sc[ui]; o = np.argsort(-su)[:W // 2]; E = u[o]; wts = logw(len(E))
        if t >= 2 and method != 'current_old':
            if t == 2:     # widen the edges that a kept setting sits on
                for j in range(D):
                    if np.any(B[:, j] <= lo[j] + 1e-9): lo[j] = -1.0
                    if np.any(B[:, j] >= hi[j] - 1e-9): hi[j] = 2.0
        if t < 2 or method in ('current_old', 'current_wide'):
            X = current_gen(r, B, lo, hi) if (t < 2 or method == 'current_wide') else current_gen(r, B, np.zeros(D), np.ones(D)); continue
        mean = (wts[:, None] * E).sum(0); sd = np.sqrt((wts[:, None] * (E - mean) ** 2).sum(0))
        if method == 'spread':
            new = np.clip(factor * sd, .03, .5)
            sig = new if sig is None else .5 * sig + .5 * new
            if prev_mean is not None:
                d = np.sign(mean - prev_mean)
                if prev_dir is not None: sig = np.where((d == prev_dir) & (d != 0), np.minimum(sig * 1.5, .5), sig)
                prev_dir = d
            prev_mean = mean
            out = list(B); p = lambda: B[r.integers(len(B))]
            for _ in range(6): out.append(np.clip(p() + r.normal(0, .5 * sig), lo, hi))
            for _ in range(4):
                x = p().copy(); k = r.choice(D, r.integers(2, 4), False); x[k] += r.normal(0, sig[k]); out.append(np.clip(x, lo, hi))
            for _ in range(3): b, c = B[r.choice(len(B), 2, False)]; out.append(np.where(r.random(D) < .5, b, c))
            for _ in range(3): x = p().copy(); k = r.choice(D, r.integers(1, 4), False); x[k] = r.uniform(lo[k], hi[k]); out.append(x)
            X = np.array(out)
        elif method in ('cma', 'cma_wide'):       # standard CMA-ES (Hansen 2016 defaults for lam=16), warm started from round 3
            lam = W - K; mu = lam // 2; w = logw(mu); mueff = 1 / (w ** 2).sum()
            if cm is None:
                s0 = max(sd.mean(), .05); C = np.diag((np.maximum(sd, .02) / s0) ** 2)
                cm = dict(m=mean, s=s0, C=C, pc=np.zeros(D), ps=np.zeros(D))
                cm['cc'] = (4 + mueff / D) / (D + 4 + 2 * mueff / D); cm['cs'] = (mueff + 2) / (D + mueff + 5)
                cm['c1'] = 2 / ((D + 1.3) ** 2 + mueff); cm['cmu'] = min(1 - cm['c1'], 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
                cm['ds'] = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cm['cs']; cm['chi'] = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D * D))
            else:   # update from the last generation's samples (the 16 new ones)
                Y = cm['Y']; ys = sc[K:]; o2 = np.argsort(-ys)[:mu]; Yb = Y[o2]
                m, s, C = cm['m'], cm['s'], cm['C']; yw = (w[:, None] * Yb).sum(0); m = m + s * yw
                ev, Bv = np.linalg.eigh(C); ev = np.maximum(ev, 1e-12); Cis = Bv @ np.diag(ev ** -.5) @ Bv.T
                cm['ps'] = (1 - cm['cs']) * cm['ps'] + np.sqrt(cm['cs'] * (2 - cm['cs']) * mueff) * Cis @ yw
                hs = np.linalg.norm(cm['ps']) / np.sqrt(1 - (1 - cm['cs']) ** (2 * cm['g'])) < (1.4 + 2 / (D + 1)) * cm['chi']
                cm['pc'] = (1 - cm['cc']) * cm['pc'] + hs * np.sqrt(cm['cc'] * (2 - cm['cc']) * mueff) * yw
                C = (1 - cm['c1'] - cm['cmu']) * C + cm['c1'] * np.outer(cm['pc'], cm['pc']) + cm['cmu'] * (w[:, None, None] * np.einsum('ni,nj->nij', Yb, Yb)).sum(0)
                s = s * np.exp((cm['cs'] / cm['ds']) * (np.linalg.norm(cm['ps']) / cm['chi'] - 1))
                cm.update(m=m, s=s, C=C)
            cm['g'] = cm.get('g', 0) + 1
            ev, Bv = np.linalg.eigh(cm['C']); A = Bv @ np.diag(np.sqrt(np.maximum(ev, 0)))
            Y = r.normal(size=(lam, D)) @ A.T
            if method == 'cma_wide': Y[r.choice(lam, 3, False)] *= 2.0     # 3 of the 16 drawn twice as far out
            Xn = np.clip(cm['m'] + cm['s'] * Y, lo, hi); cm['Y'] = (Xn - cm['m']) / cm['s']
            X = np.vstack([B, Xn])
    return res

M = {'current loop, old limits': ('current_old', 0), 'current loop, wider limits': ('current_wide', 0),
     'spread rule x1.0': ('spread', 1.0), 'spread rule x1.3': ('spread', 1.3), 'spread rule x1.7': ('spread', 1.7), 'CMA-ES copy': ('cma', 0), 'CMA-ES, 3 of 16 twice as far': ('cma_wide', 0)}
if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    for kind in sys.argv[2].split(','):
        print(f'== {kind}: true score of the winner after round 3 / 4 / 5 / 6', flush=True)
        for nm, (m, fac) in M.items():
            with Pool(15) as p: R = np.array(p.map(run, [(7000 + i, kind, m, fac) for i in range(N)], chunksize=20))
            print(f'  {nm:28s} ' + ' / '.join(f'{v:.3f}' for v in R.mean(0)) + f'   (±{R[:, 1].std() / np.sqrt(N):.3f})', flush=True)
