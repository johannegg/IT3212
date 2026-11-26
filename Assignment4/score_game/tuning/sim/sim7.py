"""Final check of the chosen design: after 3 old rounds, 1 or 2 more rounds with (a) the old loop or (b) CMA-ES 16 drawn + 4 kept,
draw again at wide limits, kept = 4 best by average. Winner = best average score of the last round (no final runs)."""
import numpy as np, sys, os
from multiprocessing import Pool
import sim6
from sim5 import problem, select, current_gen, D

def run(a):
    seed, kind, method, extra = a; r = np.random.default_rng(seed); f = problem(r, kind); H = {}
    obs = lambda X: f(X) + r.normal(0, float(os.environ.get('NOISE', '.12')) / np.sqrt(5), len(X))
    lo, hi = np.zeros(D), np.ones(D); X = r.random((20, D))
    def score(X):
        y = obs(X); key = [x.tobytes() for x in X]
        for k, v in zip(key, y): H.setdefault(k, []).append(v)
        return np.array([np.mean(H[k]) for k in key])
    for t in range(3):
        sc = score(X); B = X[select(X, sc, 4)]
        if t < 2: X = current_gen(r, B, lo, hi)
    if method == 'old':
        for _ in range(extra):
            X = current_gen(r, B, lo, hi); sc = score(X); B = X[select(X, sc, 4)]
        return float(f(X[[np.argmax(sc)]])[0])
    lo, hi = -np.ones(D), 2 * np.ones(D); u, ui = np.unique(X, axis=0, return_index=True); E = u[np.argsort(-sc[ui])[:10]]; w0 = sim6.logw(10)
    m = (w0[:, None] * E).sum(0); sd = np.sqrt((w0[:, None] * (E - m) ** 2).sum(0)); lam, mu = 16, 8; w = sim6.logw(mu); mueff = 1 / (w ** 2).sum()
    cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D); cs = (mueff + 2) / (D + mueff + 5); c1 = 2 / ((D + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff)); ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs; chi = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D * D))
    s = max(sd.mean(), .05); C = np.diag((np.maximum(sd, .02) / s) ** 2); pc = np.zeros(D); ps = np.zeros(D); kept = X[np.argsort(-sc)[:4]]
    for g in range(1, extra + 1):
        ev, Bv = np.linalg.eigh(C); ev = np.maximum(ev, 1e-12); A = Bv @ np.diag(np.sqrt(ev)); Cis = Bv @ np.diag(ev ** -.5) @ Bv.T; Y = []
        for i in range(lam):
            for _ in range(1000):
                y = A @ r.normal(size=D)
                if np.all((m + s * y >= lo) & (m + s * y <= hi)): break
            Y.append(y)
        Y = np.array(Y); X = np.vstack([kept, m + s * Y]); sc = score(X); o = np.argsort(-sc[4:])[:mu]; yw = (w[:, None] * Y[o]).sum(0); m = m + s * yw
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cis @ yw; hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * g)) < (1.4 + 2 / (D + 1)) * chi
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * yw
        C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * (w[:, None, None] * np.einsum('ni,nj->nij', Y[o], Y[o])).sum(0); s = s * np.exp((cs / ds) * (np.linalg.norm(ps) / chi - 1))
        kept = X[np.argsort(-sc)[:4]]
    return float(f(X[[np.argmax(sc)]])[0])

if __name__ == '__main__':
    N = 2000
    for extra in (1, 2):
        for kind in ('inside', 'edge', 'multi', 'multi-edge'):
            out = []
            for meth in ('old', 'cma'):
                with Pool(15) as p: R = np.array(p.map(run, [(9000 + i, kind, meth, extra) for i in range(N)], chunksize=20))
                out.append(f'{meth} {R.mean():.3f}')
            print(f'rounds after 3: {extra}  {kind:11s} ' + '   '.join(out) + f'  (±{R.std() / np.sqrt(N):.3f})', flush=True)
