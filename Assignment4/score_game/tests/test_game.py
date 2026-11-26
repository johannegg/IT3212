"""Inputs against the points, on all four mazes: random play and a nearest-dot player.   python tests/test_game.py"""
import collections, os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from dataclasses import replace
from multiprocessing import Pool
from game.env import *


def rule(g, rng):
    f = g.f; safe = [k for k in range(4) if f['op'][k] and not f['dg'][k]]
    if f['blue'] < 4: return f['blue']
    if f['dot'] < 4 and not f['dg'][f['dot']]: return f['dot']
    return int(rng.choice(safe)) if safe else int(rng.integers(4))


def go(a):
    maze, seed, kind = a
    g = ScoreGame(replace(SETTINGS['B'], start_maze=str(maze)), training=False); rng = np.random.default_rng(seed); C = collections.Counter(); out = []
    for i in range(6):
        g.reset(91_000_000 + 1000 * maze + 100 * seed + i); prev_lay = g.layout; pres = g.present.copy(); pel = None; score = g.score; gp = g.ghost_points; sc_prev = g.score; lay_start = g.layout; mazes = {g.layout}
        # every raw step is checked: wrap the emulator step
        real = g._raw; state = {}
        def raw(act, g=g, real=real, C=C, state=state):
            ram0 = g.ale.getRAM().copy(); lay0 = int(ram0[0]) & 3; lives0 = g.ale.lives(); m0 = g.maps[lay0]; pr0 = (ram0[m0.dot_byte] >> m0.dot_bit & 1).astype(int); pe0 = [int(ram0[117]) >> b & 1 for b in m0.pellets]; gp0 = g.ghost_points
            r = real(act); ram1 = g.ale.getRAM(); lay1 = int(ram1[0]) & 3
            if lay1 == lay0 and ram1[119] >= ram0[119]:
                m = g.maps[lay1]; pr1 = (ram1[m.dot_byte] >> m.dot_bit & 1).astype(int); ate = int((pr0 - pr1).sum()); pell = sum(pe0) - sum(int(ram1[117]) >> b & 1 for b in m.pellets); gpts = g.ghost_points - gp0
                C['raw steps'] += 1; C['dots match'] += int((ate >= 0) and (int(r) % 50 // 10 * 10 == 10 * ate)); rest = r - 10 * ate - 50 * pell - gpts
                C['unexplained points'] += int(rest != 0); C['points that are fruit or other: %d' % rest] += int(rest != 0)
                if int(r) >= 200: C['ghost events'] += 1
            C['lost a life'] += int(g.ale.lives() < lives0)
            return r
        g._raw = raw
        steps = 0
        while True:
            m = g.maps[g.layout]
            a = rule(g, rng) if (kind == 'rule' and rng.random() > .1) else int(rng.integers(4))
            r, done, tr, info = g.step(a); steps += 1; mazes.add(g.layout)
            ram = g.ale.getRAM(); C['decisions'] += 1
            C['inside the maze'] += int(g.maze.dist[g.pac].min() >= 0 and g.maze.coords is not None)
            if done or tr or steps > 4000: break
        out.append((info['score'], g.decisions, g.env_steps, info.get('outcome'), sorted(mazes), g.ghosts_eaten, g.ghost_points)); g._raw = real
        C['finished'] += 1
    return maze, kind, C, out


if __name__ == '__main__':
    jobs = [(m, s, k) for m in range(4) for s in range(3) for k in ('random', 'rule')]
    T = collections.defaultdict(collections.Counter); R = collections.defaultdict(list)
    with Pool(15) as p:
        for m, k, c, o in p.imap_unordered(go, jobs): T[m].update(c); R[m, k] += o
    for m in sorted(T):
        c = T[m]; print(f"maze {m + 1}: raw steps {c['raw steps']}, dots by memory = dots by points in {c['dots match'] / c['raw steps'] * 100:.3f}% of steps, unexplained points in {c['unexplained points']} steps ->", {k: v for k, v in c.items() if 'fruit' in k}, '| lives lost', c['lost a life'])
    for k, o in sorted(R.items()): print(k, 'games', len(o), 'mean score', np.mean([x[0] for x in o]).round(0), 'decisions', np.mean([x[1] for x in o]).round(0), 'steps', np.mean([x[2] for x in o]).round(0), 'outcomes', dict(collections.Counter(x[3] for x in o)), 'mazes seen', dict(collections.Counter(tuple(x[4]) for x in o)))
