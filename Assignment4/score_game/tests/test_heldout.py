"""D stays on maze 4 (walls, dots and points), and training never sees maze 4.   python tests/test_heldout.py"""
import collections, os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from game.env import *
from test_game import rule


def d(seed):
    g = ScoreGame(SETTINGS['D'], training=False); rng = np.random.default_rng(seed); C = collections.Counter()
    real = g._raw
    def raw(act):
        r0 = g.ale.getRAM().copy(); m = g.maps[3]; a0 = (r0[m.dot_byte] >> m.dot_bit & 1).astype(int); pe0 = sum(int(r0[117]) >> b & 1 for b in m.pellets)
        r = real(act); r1 = g.ale.getRAM(); a1 = (r1[m.dot_byte] >> m.dot_bit & 1).astype(int); pe1 = sum(int(r1[117]) >> b & 1 for b in m.pellets); C['raw'] += 1
        C['layout is maze 4'] += int((int(r1[0]) & 3) == 3)
        if a1.sum() > a0.sum() + 50: C['new level (dots refilled)'] += 1
        else:
            C['dots match points'] += int(int(r) % 50 // 10 == int((a0 - a1).sum())); C['dots checked'] += 1
        return r
    g._raw = raw
    for i in range(6):
        g.reset(96_000_000 + 100 * seed + i); lv = 0
        while True:
            r, dn, tr, info = g.step(rule(g, rng) if rng.random() > .05 else int(rng.integers(4)))
            x = float(g.ale.getRAM()[10]) + PAC_X_OFF; y = float(g.ale.getRAM()[16]) + PAC_Y_OFF; nr, nc = g.maze.coords[g.pac]; C['pos'] += 1; C['position on the maze-4 walls'] += int(abs(y - (7.5 + 12 * nr)) <= 8 and (abs(x - (4 * nc + 2)) <= 6 or abs(x - (4 * nc + 2)) >= 150)); C['layout seen by agent %d' % (g.layout + 1)] += 1
            if dn or tr: break
        C['games'] += 1; C['score'] += int(info['score'])
    return C


def t(seed):      # training: play on from maze 3 with cleared levels forced (dots removed) until maze 4 would come; the agent must not see maze 4
    from dataclasses import replace
    g = ScoreGame(replace(SETTINGS['B'], start_maze='2'), training=True); g.reset(seed); C = collections.Counter(); seen = set()
    for k in range(60):
        g.ale.setRAM(0, 3)     # the game shows maze 4
        r, dn, tr, info = g.step(int(np.random.default_rng(seed + k).integers(4)))
        seen.add(g.layout); C['held out flag'] += int(info['held_out']); 
        if tr: C['game stopped'] += 1; break
    C['agent saw layouts ' + str(sorted(x + 1 for x in seen))] += 1
    return C


if __name__ == '__main__':
    T = collections.Counter()
    with Pool(14) as p:
        for c in p.imap_unordered(d, range(14)): T.update(c)
        U = collections.Counter()
        for c in p.imap_unordered(t, range(10)): U.update(c)
    for k in sorted(T): print('D:', k, T[k])
    print('training:', dict(U))
