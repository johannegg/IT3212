"""The full Ms. Pac-Man game (three lives, real points) as Q-learning and PPO see it.

Everything is read from the game's memory, but only what a player can see on the screen: the memory replaces reading
the picture, which was wrong in up to 30% of the steps. A game ends at game over, after STUCK_STEPS decisions without points
or a lost life, or (in training) when the held-out maze 4 is reached.
One decision = 4 frames. Moments in which nothing moves (start, death, eaten ghost) are skipped and counted as steps.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import ale_py
import gymnasium as gym
import numpy as np

from .maze import GHOST_X_OFF, GHOST_Y_OFF, PAC_X_OFF, PAC_Y_OFF, Maze

gym.register_envs(ale_py)

ACTIONS = [1, 2, 3, 4]                   # ALE ids for UP, RIGHT, LEFT, DOWN
ACTION_NAMES = ["UP", "RIGHT", "LEFT", "DOWN"]
OPPOSITE = [3, 2, 1, 0]
CW = [0, 1, 3, 2]                        # place of UP, RIGHT, LEFT, DOWN on the clock
CWINV = [0, 1, 3, 2]
HERE = os.path.dirname(os.path.abspath(__file__))

# memory addresses
RAM_MAZE, RAM_GHOST_STATE, RAM_DIRECTION, RAM_PELLETS = 0, 1, 56, 117
RAM_GHOST_X, RAM_PAC_X, RAM_FRUIT_X, RAM_GHOST_Y, RAM_PAC_Y, RAM_FRUIT_Y = 6, 10, 11, 12, 16, 17
DIRECTION = {0: 0, 1: 1, 2: 3, 3: 2}     # game direction -> action index; anything else = none (4)

# fixed choices (graph units: 1 unit = 4 pixels)
DANGER_FAR, DANGER_NEAR = 16, 6          # a dangerous ghost this near that does not get farther away
BLUE_RADIUS, POWER_RADIUS = 40, 40
START_MOVES, STUCK_STEPS, MAX_SKIP = 30, 5000, 200
HELD_OUT_MAZE = 3                        # maze 4


@dataclass
class Setting:
    name: str
    sticky: float = 0.0                  # probability that the previous action is repeated, per frame
    step_penalty: float = 0.0            # training reward only
    start_maze: str = "train"            # "train" = maze 1, 2 or 3 at random, or the number of a maze (0-3)
    keep_maze: bool = False              # Setting D: after a clear the game stays on the same maze (the layout byte is written back)
    success_score: float = 6952.0        # counted as success when the game ends with at least this score
    success_bonus: float = 0.0           # added to the training reward once, at the step where the score first reaches success_score

    def to_dict(self): return asdict(self)


SETTINGS = {
    "A": Setting("A", 0.0), "B": Setting("B", 0.25), "C": Setting("C", 0.25, step_penalty=0.5),
    "D": Setting("D", 0.25, start_maze="3", keep_maze=True),
}


class Maps:
    """Walls (from the first picture of each maze), and which memory bit belongs to which dot and power pellet."""
    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is None:
            e = gym.make("ALE/MsPacman-v5", repeat_action_probability=0.0, frameskip=4, full_action_space=False)
            out = []
            for v in range(4):
                e.reset(seed=0); e.unwrapped.ale.setRAM(RAM_MAZE, v); f, *_ = e.step(0); m = Maze(f)
                j = json.load(open(os.path.join(HERE, "maps", f"map{v}.json")))
                assert list(m.dot_nodes) == j["dot_nodes"], "the first picture was read differently"
                b = np.array([j["dots"][str(i)] for i in range(len(m.dot_nodes))])
                m.dot_byte, m.dot_bit = b[:, 0], b[:, 1]
                m.pellets = {int(k): int(n[0]) for k, n in j["power"].items()}
                m.dist32 = m.dist.astype(np.float32); out.append(m)
            e.close(); cls._cache = out
        return cls._cache


class ScoreGame:
    """reset(seed, maze) -> info; step(action) -> (training reward, done, truncated, info)."""

    def __init__(self, setting: Setting, guides: dict | None = None, training: bool = True):
        g = {"life_penalty": 0.0, "ghost_value": 1.0, **(guides or {})}
        self.s, self.life_penalty, self.ghost_value, self.training = setting, g["life_penalty"], g["ghost_value"], training
        self.env = gym.make("ALE/MsPacman-v5", repeat_action_probability=setting.sticky, frameskip=4, full_action_space=False)
        self.ale = self.env.unwrapped.ale
        self.maps = Maps.get()
        if setting.sticky == 0:
            pass

    # ------------------------------------------------------------------ game
    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.env.reset(seed=int(seed) % (2 ** 31 - 1))
        layout = int(self.rng.integers(0, 3)) if self.s.start_maze == "train" else int(self.s.start_maze)
        self.start_layout = layout
        if layout: self.ale.setRAM(RAM_MAZE, layout)
        self.env_steps = self.decisions = self.idle = 0
        self.score = 0.0; self.lives = 3; self.lost = 0; self.over = False; self.edible = [False] * 4; self.eyes = [False] * 4; self.chain = 0; self.ghost_points = 0.0; self.ghosts_eaten = 0; self.bonus_paid = False
        self._raw(0)                                        # the game draws the maze
        self.layout = layout
        for _ in range(70):                                 # the intro in which nobody moves
            p = self._positions()
            self._raw(0)
            if self._positions() != p: break
        for _ in range(int(self.rng.integers(0, START_MOVES + 1))):     # 0-30 random moves
            if self.over: break
            self._raw(ACTIONS[int(self.rng.integers(0, 4))])
        self.start_score = self.score
        self._see()
        return {"env_steps": self.env_steps, "layout": self.layout}

    def _positions(self):
        r = self.ale.getRAM()
        return tuple(int(x) for x in r[RAM_GHOST_X:RAM_GHOST_X + 5]) + tuple(int(x) for x in r[RAM_GHOST_Y:RAM_GHOST_Y + 5])

    def _raw(self, act):
        """One emulator step (4 frames): score, lives, ghosts."""
        _, r, term, trunc, info = self.env.step(act)
        self.env_steps += 1; self.score += r; self._last_points = float(r)
        ram = self.ale.getRAM()
        if self.s.keep_maze and (int(ram[RAM_MAZE]) & 3) != self.start_layout: self.ale.setRAM(RAM_MAZE, self.start_layout); ram = self.ale.getRAM()
        edible = [bool(ram[RAM_GHOST_STATE + k] & 128) for k in range(4)]
        house = [bool(ram[RAM_GHOST_STATE + k] & 32) for k in range(4)]
        new_fright = any(edible) and not any(self.edible)
        if new_fright: self.chain = 0
        gone = [k for k in range(4) if self.edible[k] and not edible[k]]
        if gone and r >= 200:       # ghosts were eaten in this step: the points tell how many (200, 400, 800, 1600 in a row); others only stopped being blue
            left = int(r) // 200 * 200; n = 0
            while n < len(gone) and self.chain + n < 12 and left >= 200 * 2 ** min(self.chain + n, 3):
                left -= 200 * 2 ** min(self.chain + n, 3); n += 1
            n = max(n, 1); px, py = float(ram[RAM_PAC_X]), float(ram[RAM_PAC_Y])
            near = sorted(gone, key=lambda k: abs(float(ram[RAM_GHOST_X + k]) - px) + abs(float(ram[RAM_GHOST_Y + k]) - py))[:n]
            for j, k in enumerate(near):
                self.eyes[k] = True
            pts = sum(200 * 2 ** min(self.chain + j, 3) for j in range(n)); self.chain += n
            self.ghost_points += min(pts, float(int(r) // 100 * 100)); self.ghosts_eaten += n
        for k in range(4):
            if k not in gone and (house[k] or edible[k]): self.eyes[k] = False
            elif k in gone and house[k] and not self.eyes[k]: self.eyes[k] = False
        self.edible = edible
        lv = int(info["lives"])
        if lv < self.lives: self.lost += self.lives - lv
        self.lives = lv                                     # also goes up: an extra life at 10,000 points
        if term or lv == 0: self.over = True
        if trunc: self.trunc_game = True
        return r

    def step(self, action: int):
        """The agent's action, then skip over moments in which nothing moves. Returns training reward, done, truncated, info."""
        score0, lost0, ghost0 = self.score, self.lost, self.ghost_points
        self.trunc_game = False
        p = self._positions(); self._raw(ACTIONS[action]); n = 0
        while not self.over and not self.trunc_game and n < MAX_SKIP and self._positions() == p:      # nothing moved: frozen
            p = self._positions(); self._raw(ACTIONS[action]); n += 1
        self.decisions += 1
        gained = self.score - score0; lost = self.lost - lost0
        self.idle = 0 if (gained > 0 or lost > 0) else self.idle + 1
        truncated = self.trunc_game or self.idle >= STUCK_STEPS
        held_out = self.training and (int(self.ale.getRAM()[RAM_MAZE]) & 3) == HELD_OUT_MAZE      # in training the agent never sees anything of maze 4: the game is stopped before
        if held_out: truncated = truncated or not self.over
        else: self._see()
        done = self.over
        r = gained + (self.ghost_value - 1.0) * (self.ghost_points - ghost0) - self.life_penalty * lost - self.s.step_penalty
        success = self.score >= self.s.success_score
        if success and not self.bonus_paid: r += self.s.success_bonus; self.bonus_paid = True     # paid when success is decided (the score never drops), not at the last death
        info = {"points": gained, "lost": lost, "score": self.score, "stuck": self.idle >= STUCK_STEPS, "held_out": bool(held_out)}
        if done or truncated:
            info["outcome"] = "success" if success else ("failure" if done else "timeout")
        return float(r), done, truncated, info

    # ------------------------------------------------------------- what she sees
    def _see(self):
        ram = self.ale.getRAM()
        lay = int(ram[RAM_MAZE]) & 3
        self.layout = lay; mz = self.maps[lay]; self.maze = mz
        pac = mz.node_of_pixel(float(ram[RAM_PAC_X]) + PAC_X_OFF, float(ram[RAM_PAC_Y]) + PAC_Y_OFF); self.pac = pac
        self.direction = DIRECTION.get(int(ram[RAM_DIRECTION]), 4)
        present = (ram[mz.dot_byte] >> mz.dot_bit & 1).astype(bool); self.present = present
        dots = mz.dot_nodes[present]; self.dots_left = int(present.sum()); self.dots_total = len(present)
        gh = [mz.node_of_pixel(float(ram[RAM_GHOST_X + k]) + GHOST_X_OFF, float(ram[RAM_GHOST_Y + k]) + GHOST_Y_OFF) for k in range(4)]
        dp = mz.dist[pac]; nb = mz.neighbor[pac]; op = nb >= 0; self.open = op
        bad = [g for k, g in enumerate(gh) if not self.edible[k] and not self.eyes[k] and np.isfinite(dp[g])]
        blue = [g for k, g in enumerate(gh) if self.edible[k] and np.isfinite(dp[g])]
        power = [n for b, n in mz.pellets.items() if ram[RAM_PELLETS] >> b & 1]
        fruit = []
        if ram[RAM_FRUIT_X] != 0:
            fr = mz.node_of_pixel(float(ram[RAM_FRUIT_X]) + GHOST_X_OFF, float(ram[RAM_FRUIT_Y]) + GHOST_Y_OFF)
            if np.isfinite(dp[fr]): fruit = [fr]

        def towards(t, radius=None):
            if len(t) == 0 or not op.any(): return 4
            t = np.asarray(t)
            if radius is not None and dp[t].min() > radius: return 4
            d = np.full(4, np.inf)
            for k in range(4):
                if op[k]: d[k] = mz.dist[nb[k], t].min()
            return int(np.argmin(d)) if np.isfinite(d).any() else 4

        dg = np.zeros(4, int)
        for g in bad:
            for k in range(4):
                if op[k]:
                    x = mz.dist[nb[k], g]
                    if x <= DANGER_FAR and x <= dp[g]: dg[k] = max(dg[k], 2 if x <= DANGER_NEAR else 1)
        self.f = dict(op=op, dot=towards(dots), dg=dg, blue=towards(blue, BLUE_RADIUS), power=towards(power, POWER_RADIUS), fruit=towards(fruit))

    def view(self):
        """Per direction: (open, nearest dot, danger 0-2, blue ghost, power pellet, fruit, way 0 forward / 1 side / 2 back / 3 no heading);
        and for the whole situation: danger anywhere, blue ghost near, dots left (0-2), lives left (1-3)."""
        f = self.f; h = self.direction
        if h < 4: way = [0 if k == h else 2 if k == OPPOSITE[h] else 1 for k in range(4)]
        else: way = [3] * 4
        glob = (int(f["dg"].any()), int(f["blue"] < 4), min(2, 2 - 3 * self.dots_left // (self.dots_total + 1)) if self.dots_total else 0, max(1, min(3, self.lives)))
        rows = [(int(f["op"][k]), int(f["dot"] == k), int(f["dg"][k]), int(f["blue"] == k), int(f["power"] == k), int(f["fruit"] == k), way[k]) for k in range(4)]
        return rows, glob

    def close(self): self.env.close()


# ---- the two agents see the same information -----------------------------------------------------------------------------------
def q_keys(game):
    rows, glob = game.view()
    return tuple(r + glob for r in rows)


N_DIR = 1 + 1 + 2 + 1 + 1 + 1 + 3           # open, dot, two danger levels, blue, power, fruit, way (forward / side / back)
N_GLOB = 1 + 1 + 3 + 3                      # danger anywhere, blue near, dots left, lives
OBS_DIR = N_DIR + N_GLOB


def ppo_view(game):
    """4 x OBS_DIR array, directions in absolute order, and the order in which the critic sees them (from her heading)."""
    rows, glob = game.view()
    g = np.zeros(N_GLOB, np.float32); g[0] = glob[0]; g[1] = glob[1]; g[2 + glob[2]] = 1; g[5 + glob[3] - 1] = 1
    out = np.zeros((4, OBS_DIR), np.float32)
    for k, (o, d, dg, b, p, fr, w) in enumerate(rows):
        x = out[k]; x[0], x[1] = o, d; x[2] = dg > 0; x[3] = dg > 1; x[4], x[5], x[6] = b, p, fr
        if w < 3: x[7 + w] = 1
        x[N_DIR:] = g
    h = game.direction
    order = [CWINV[(i + CW[h]) % 4] for i in range(4)] if h < 4 else list(range(4))
    return out, order
