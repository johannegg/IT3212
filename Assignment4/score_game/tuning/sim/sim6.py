"""After 3 rounds of the old 20/4 loop: CMA-ES rounds with K kept re-tested + lam drawn, 'draw again' at the limits,
equal total number of new settings tried. Compares lam (round width) and whether the 4 kept are needed.
Result: TRUE score of the final winner (best re-tested of the kept, or CMA centre when K=0) after ~N settings."""
import numpy as np, sys, os
from multiprocessing import Pool
from sim5 import problem, select, current_gen, logw, D

def run(a):
    seed, kind, lam, K, total = a; r = np.random.default_rng(seed); f = problem(r, kind); H = {}
    noise = float(os.environ.get('NOISE', '.12')); obs = lambda X: f(X) + r.normal(0, noise / np.sqrt(5), len(X))
    lo, hi = np.zeros(D), np.ones(D); X = r.random((20, D))
    for t in range(3):                                   # rounds 1-3, old loop (as done)
        y = obs(X); key = [x.tobytes() for x in X]
        for k, v in zip(key, y): H.setdefault(k, []).append(v)
        sc = np.array([np.mean(H[k]) for k in key]); B = X[select(X, sc, 4)]
        if t < 2: X = current_gen(r, B, lo, hi)
    for j in range(D):                                   # wide limits around the old ones
        lo[j], hi[j] = -1.0, 2.0
    u, ui = np.unique(X, axis=0, return_index=True); su = sc[ui]; E = u[np.argsort(-su)[:10]]; w0 = logw(len(E))
    mean = (w0[:, None] * E).sum(0); sd = np.sqrt((w0[:, None] * (E - mean) ** 2).sum(0))
    mu = lam // 2; w = logw(mu); mueff = 1 / (w ** 2).sum()
    cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D); cs = (mueff + 2) / (D + mueff + 5)
    c1 = 2 / ((D + 1.3) ** 2 + mueff); cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
    ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs; chi = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D * D))
    m = mean; s = max(sd.mean(), .05); C = np.diag((np.maximum(sd, .02) / s) ** 2); pc = np.zeros(D); ps = np.zeros(D)
    allX, allS = list(X), list(sc); kept = B[:K]; tried = 0; g = 0
    while tried + lam + K <= total:
        g += 1; ev, Bv = np.linalg.eigh(C); ev = np.maximum(ev, 1e-12); A = Bv @ np.diag(np.sqrt(ev)); Cis = Bv @ np.diag(ev ** -.5) @ Bv.T
        Y = np.empty((lam, D))
        for i in range(lam):                             # draw again until inside the limits (Hansen B.5)
            for _ in range(100):
                y = A @ r.normal(size=D)
                if np.all((m + s * y >= lo) & (m + s * y <= hi)): break
            Y[i] = y
        Xn = m + s * Y; tried += lam + K
        X = np.vstack([kept, Xn]) if K else Xn
        yv = obs(X); key = [x.tobytes() for x in X]
        for k, v in zip(key, yv): H.setdefault(k, []).append(v)
        sc = np.array([np.mean(H[k]) for k in key]); allX += list(X); allS += list(sc)
        yn = sc[K:]; o = np.argsort(-yn)[:mu]; yw = (w[:, None] * Y[o]).sum(0); m = m + s * yw     # only the drawn ones update the cloud
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cis @ yw
        hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * g)) < (1.4 + 2 / (D + 1)) * chi
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * yw
        C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * (w[:, None, None] * np.einsum('ni,nj->nij', Y[o], Y[o])).sum(0)
        s = s * np.exp((cs / ds) * (np.linalg.norm(ps) / chi - 1))
        if K: kept = X[np.argsort(-sc)[:K]]
    if K and os.environ.get('CENTRE'):   # final: the 4 kept + the centre, 10 new seeds each
        fin = np.vstack([kept[:4], m]); z = (obs(fin) + obs(fin)) / 2; return float(f(fin[[np.argmax(z)]])[0])
    if K:   # final: the 4 kept, 5 new seeds each, combined with what they had
        fin = kept[:4]; z = (obs(fin) * 5 + np.array([np.mean(H[x.tobytes()]) for x in fin]) * 5) / 10; return float(f(fin[[np.argmax(z)]])[0])
    fin = np.vstack([m, np.array(allX)[np.argsort(-np.array(allS))[:3]]]); z = (obs(fin) + obs(fin)) / 2 if os.environ.get('CENTRE') else obs(fin); return float(f(fin[[np.argmax(z)]])[0])

if __name__ == '__main__':
    N = 2000; kinds = sys.argv[1].split(',')
    P = {'16 drawn + 4 kept (20/round)': (16, 4), '10 drawn + 4 kept (14/round, Hansen default)': (10, 4),
         '26 drawn + 4 kept (30/round)': (26, 4), '20 drawn, no kept (20/round)': (20, 0)}
    for total in (40, 60):
        for kind in kinds:
            print(f'== {kind}, {total} runs after round 3 (about 2 or 3 of our rounds)', flush=True)
            for nm, (lam, K) in P.items():
                with Pool(15) as p: R = np.array(p.map(run, [(7000 + i, kind, lam, K, total) for i in range(N)], chunksize=20))
                runs = (total // (lam + K)) * (lam + K)
                print(f'  {nm:46s} {R.mean():.3f} (±{R.std() / np.sqrt(N):.3f})  rounds {total // (lam + K)}, runs {runs}', flush=True)
