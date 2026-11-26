"""Which memory bit belongs to which dot and power pellet, for all four mazes. Found by clearing one bit at a time and looking at what disappears.

    python make_maps.py      # writes game/maps/map0.json ... map3.json
"""
import json, os
import ale_py, gymnasium as gym, numpy as np
from scipy import ndimage
from game.maze import Maze, PINK, MAZE_HEIGHT, BACKGROUND
gym.register_envs(ale_py)
HERE = os.path.dirname(os.path.abspath(__file__))
DOT_BYTES, PELLET_BYTE = range(59, 105), 117


def pellets(frame, walls):
    pink = (frame[:MAZE_HEIGHT] == PINK).all(-1) & ~walls
    lab, n = ndimage.label(pink); sizes = ndimage.sum(pink, lab, range(1, n + 1)) if n else []
    return [tuple(np.round(ndimage.center_of_mass(pink, lab, i + 1), 1)) for i in range(n) if 12 < sizes[i] <= 60]


def build(v):
    e = gym.make('ALE/MsPacman-v5', repeat_action_probability=0., frameskip=4, full_action_space=False); e.reset(seed=0); ale = e.unwrapped.ale
    ale.setRAM(0, v); f, *_ = e.step(0); m = Maze(f); st0 = e.unwrapped.clone_state()
    def frames_after(setter, n=14):       # the picture blinks: look at several frames and take everything that is shown in one of them
        e.unwrapped.restore_state(st0); setter(); fr = []
        for _ in range(n): f, *_ = e.step(0); fr.append(f)
        return fr
    base = frames_after(lambda: None); shown = np.array([[m.dots_present(f)] for f in base])[:, 0]
    ref = shown.any(0); P0 = sorted({p for f in base for p in pellets(f, m.walls)})
    dots = {}; ram = ale.getRAM()
    e.unwrapped.restore_state(st0); r0 = ale.getRAM().copy()
    for b in DOT_BYTES:
        for bit in range(8):
            if not r0[b] >> bit & 1: continue
            fr = frames_after(lambda: ale.setRAM(b, int(r0[b]) & ~(1 << bit) & 255)); vis = np.any([m.dots_present(f) for f in fr], 0)
            gone = np.flatnonzero(ref & ~vis)
            if len(gone) == 1: dots[int(gone[0])] = [int(b), int(bit)]
            elif len(gone) > 1: print('maze', v, 'byte', b, 'bit', bit, 'hides', len(gone))
    power = {}
    for bit in range(2, 6):
        fr = frames_after(lambda: ale.setRAM(PELLET_BYTE, int(r0[PELLET_BYTE]) & ~(1 << bit) & 255)); P1 = {p for f in fr for p in pellets(f, m.walls)}
        lost = [p for p in P0 if not any(abs(p[0] - q[0]) < 3 and abs(p[1] - q[1]) < 3 for q in P1)]
        power[bit] = [int(m.node_of_pixel(p[1], p[0])) for p in lost]
    e.close(); return dict(dots={str(k): v2 for k, v2 in sorted(dots.items())}, power={str(k): v2 for k, v2 in power.items()}, dot_nodes=[int(x) for x in m.dot_nodes]), m
if __name__ == '__main__':
    for v in range(4):
        d, m = build(v); n = len(m.dot_nodes)
        print('maze', v + 1, 'dots', n, 'found', len(d['dots']), 'distinct bits', len({tuple(x) for x in d['dots'].values()}), 'pellet bits', d['power'])
        json.dump(d, open(os.path.join(HERE, 'game', 'maps', f'map{v}.json'), 'w'))
