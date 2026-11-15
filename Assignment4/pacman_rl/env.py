"""MsPacman task used for all settings (A-D).

One episode = one life of Ms. Pac-Man:
  * success: she has eaten `success_dots` dots (a fraction of the maze),
  * failure: she loses her life (caught by a ghost),
  * timeout: neither happened within `max_steps` agent steps.

Both agents see the SAME observation: a small set of egocentric features
(see `features()`), so differences between agents come from the algorithm,
not from the input. Q-learning uses the discrete state id, PPO uses the
same information as a 22-dimensional 0/1 vector.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import ale_py
import gymnasium as gym
import numpy as np

from .maze import GHOST_X_OFF, GHOST_Y_OFF, PAC_X_OFF, PAC_Y_OFF, Maze

gym.register_envs(ale_py)

ACTIONS = [1, 2, 3, 4]                 # ALE ids for UP, RIGHT, LEFT, DOWN
ACTION_NAMES = ["UP", "RIGHT", "LEFT", "DOWN"]
N_ACTIONS = 4
RAM_MAZE = 0                           # which maze layout is used (0-3)
RAM_FRIGHT_TIMER = 116                 # > 0 while ghosts are frightened
INTRO_STEPS = 65                       # agent steps before Ms. Pac-Man can move

# feature thresholds, in graph units (1 unit = 4 pixels)
DANGER_RADIUS = 16
CHASE_RADIUS = 40
POWER_RADIUS = 40

N_STATES = 16 * 4 * 16 * 5 * 5         # open x dot_dir x danger x chase x power
OBS_DIM = 4 + 4 + 4 + 5 + 5


@dataclass
class Setting:
    name: str
    sticky: float = 0.0          # probability that the previous action is repeated
    step_penalty: float = 0.0    # subtracted from the reward on every agent step
    maze: int = 0                # maze layout (0 = the normal first maze)
    success_dots: int = 50       # dots to eat for success (~1/3 of the maze)
    success_bonus: float = 25.0  # terminal success reward
    reward_mode: str = "score+success"   # or "success" (terminal reward only)
    max_steps: int = 2000        # agent steps before timeout
    noop_max: int = 30           # random no-op actions at the start (varied start states)

    def to_dict(self):
        return asdict(self)


SETTINGS = {
    "A": Setting("A", sticky=0.0),
    "B": Setting("B", sticky=0.25),
    "C": Setting("C", sticky=0.25, step_penalty=0.05),
    # D: evaluate Setting-B agents on an unseen maze, no further training
    "D": Setting("D", sticky=0.25, maze=1),
}


class PacmanTask:
    def __init__(self, setting: Setting):
        self.s = setting
        self.env = gym.make("ALE/MsPacman-v5", repeat_action_probability=setting.sticky,
                            frameskip=4, full_action_space=False)
        self.ale = self.env.unwrapped.ale
        self.maze_cache: dict[int, Maze] = {}
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int):
        """Returns (state_id, obs_vector, info). All env.step calls, including
        no-ops, are counted in info['env_steps']."""
        self.rng = np.random.default_rng(seed)
        frame, info = self.env.reset(seed=int(seed))
        steps = 0
        if self.s.maze != 0:
            self.ale.setRAM(RAM_MAZE, self.s.maze)
        frame, _, _, _, info = self.env.step(0)
        steps += 1
        if self.s.maze not in self.maze_cache:
            self.maze_cache[self.s.maze] = Maze(frame)
        self.maze = self.maze_cache[self.s.maze]
        # The game starts with an intro in which nobody can move (66 agent steps).
        # Random no-ops are added AFTER the intro, so the ghosts start in
        # different places. Without this, every episode in Setting A is identical.
        for _ in range(INTRO_STEPS + int(self.rng.integers(0, self.s.noop_max + 1))):
            frame, _, _, _, info = self.env.step(0)
            steps += 1
        self.lives = info["lives"]
        self.n_dots = len(self.maze.dot_nodes)
        self.power = {}                 # node -> pixel center, power pellets not yet eaten
        self.power_eaten = set()
        self.t = 0
        self.dots_eaten = 0
        self.frame = frame
        self._update_power(frame)
        sid, vec, feat = self.features()
        return sid, vec, {"env_steps": steps, "features": feat}

    # ------------------------------------------------------------------- step
    def step(self, action: int):
        frame, score, _term, _trunc, info = self.env.step(ACTIONS[action])
        self.t += 1
        self.frame = frame
        self._update_power(frame, ate_power=(score == 50))
        present = self.maze.dots_present(frame)
        self.dots_eaten = int(self.n_dots - present.sum())

        failure = info["lives"] < self.lives or _term
        success = (not failure) and self.dots_eaten >= self.s.success_dots
        timeout = (not failure and not success) and self.t >= self.s.max_steps

        if self.s.reward_mode == "score+success":
            reward = min(max(score, 0.0), 200.0) / 10.0
        else:
            reward = 0.0
        if success:
            reward += self.s.success_bonus
        reward -= self.s.step_penalty

        outcome = "success" if success else "failure" if failure else "timeout" if timeout else None
        sid, vec, feat = self.features()
        return sid, vec, reward, (success or failure), timeout, {
            "outcome": outcome, "game_score": score, "dots_eaten": self.dots_eaten,
            "features": feat, "env_steps": 1}

    # --------------------------------------------------------------- features
    def _positions(self):
        ram = self.ale.getRAM().astype(float)
        mz = self.maze
        pac = mz.node_of_pixel(ram[10] + PAC_X_OFF, ram[16] + PAC_Y_OFF)
        ghosts = [mz.node_of_pixel(ram[6 + k] + GHOST_X_OFF, ram[12 + k] + GHOST_Y_OFF) for k in range(4)]
        frightened = ram[RAM_FRIGHT_TIMER] > 0
        return pac, ghosts, frightened, ram

    def _update_power(self, frame, ate_power=False):
        mz = self.maze
        pac, *_ = self._positions()
        if self.t < 20:   # power pellets blink, so collect them over the first steps
            for x, y in mz.find_power_pellets(frame):
                node = mz.node_of_pixel(x, y)
                if node not in self.power_eaten:
                    self.power[node] = (x, y)
        if self.power:
            nodes = np.array(list(self.power))
            d = mz.dist[pac, nodes]
            for node in nodes[d <= 2]:
                self.power.pop(int(node)); self.power_eaten.add(int(node))
            if ate_power and self.power and d.min() > 2:
                node = int(nodes[np.argmin(d)])
                self.power.pop(node); self.power_eaten.add(node)

    def features(self):
        mz = self.maze
        pac, ghosts, frightened, _ram = self._positions()
        nb = mz.neighbor[pac]
        open_ = nb >= 0
        present = mz.dots_present(self.frame)
        dot_nodes = mz.dot_nodes[present]

        def best_dir(targets, radius=None):
            """Open direction whose next node is closest to any target; 4 = none."""
            if len(targets) == 0 or not open_.any():
                return 4
            if radius is not None and mz.dist[pac, targets].min() > radius:
                return 4
            d = np.full(4, np.inf)
            for k in range(4):
                if open_[k]:
                    d[k] = mz.dist[nb[k], targets].min()
            return int(np.argmin(d)) if np.isfinite(d).any() else 4

        dot_dir = best_dir(dot_nodes)
        dot_dir = 0 if dot_dir == 4 else dot_dir

        danger = np.zeros(4, dtype=bool)
        live_ghosts = [g for g in ghosts if np.isfinite(mz.dist[pac, g])]   # not in the ghost house
        if not frightened:
            for g in live_ghosts:
                for k in range(4):
                    if open_[k] and mz.dist[nb[k], g] <= DANGER_RADIUS and mz.dist[nb[k], g] <= mz.dist[pac, g]:
                        danger[k] = True
        chase_dir = best_dir(np.array(live_ghosts, dtype=int), CHASE_RADIUS) if frightened else 4
        power_dir = best_dir(np.array(list(self.power), dtype=int), POWER_RADIUS)

        open_bits = int(open_ @ [1, 2, 4, 8])
        danger_bits = int(danger @ [1, 2, 4, 8])
        sid = (((open_bits * 4 + dot_dir) * 16 + danger_bits) * 5 + chase_dir) * 5 + power_dir

        vec = np.zeros(OBS_DIM, dtype=np.float32)
        vec[0:4] = open_
        vec[4 + dot_dir] = 1
        vec[8:12] = danger
        vec[12 + chase_dir] = 1
        vec[17 + power_dir] = 1
        feat = {"open": open_bits, "dot_dir": dot_dir, "danger": danger_bits,
                "chase_dir": chase_dir, "power_dir": power_dir, "pac_node": int(pac),
                "frightened": bool(frightened)}
        return int(sid), vec, feat

    def close(self):
        self.env.close()
