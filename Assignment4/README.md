# Q-learning vs PPO on MsPacman (IT3212 Assignment 4, Group 44)

## Current version: `score_game/`

`score_game/` replaces the setup described in the rest of this file. It plays the full game for score (3 lives, all 4 mazes), tunes both models with the same search allowance, and runs Settings A–D and Part 3 with the winning settings. Every choice and its reason is in [design_decisions.md](design_decisions.md). The rest of this file describes the first Part 2 version, kept unchanged with its results in `records/part2/`.

```bash
pip install -r requirements.txt
bash score_game/run_all.sh      # tuning, winner pick, Settings A-D, A-D summary, Part 3
```

| Path | Content |
|---|---|
| `score_game/game/` | The game (ALE MsPacman-v5 read from memory), the inputs and both agents |
| `score_game/tune.py` | The tuning loop (design_decisions.md section 6) |
| `score_game/tuning/` | Every tuning run (`results.jsonl`), every choice of the loop (`decisions.log`), the plans, rankings and CMA-ES state of each round, and the winners (`winners.json`, `x.json`) |
| `score_game/tuning/sim/` | The simulations behind the tuning design (rows 6.9, 6.18, 6.19) |
| `score_game/run_ad.py`, `summarize.py` | Settings A–D, 5 seeds per model, 200 test games per agent → `score_game/records/ad/` |
| `score_game/run_part3.py` | Part 3: one factor at a time from the winners (×1/3 and ×3), 5 seeds each → `score_game/records/part3/` |
| `score_game/baseline.py` | Hand-written and random players as a reference |

Each step skips work that is already saved, so `run_all.sh` can be stopped and started again. `bash score_game/run_more_seeds.sh` adds training seeds 5–9 and then 10–14 to A–D and Part 3, and redoes the summaries after each block. A tuning run is about 7 minutes (Q-learning) or 13 minutes (PPO) on one CPU core; `run_all.sh` uses 15 workers by default (`WORKERS=4 bash score_game/run_all.sh` for fewer).

## Repository layout (first Part 2 version)

```
pacman_rl/            environment, features and agents (shared by all parts)
train.py, evaluate.py, summarize.py, baseline.py
run_experiments.py    Part 2: one command for the main experiment
run_part3.py          Part 3: one factor at a time
records/
  part2/runs/<setting>/<algo>/seed<n>/      configs, logs, models, episode traces
  part2/results/                            tables and figures
  part3/runs/<investigation>_<algo>_<value>/<setting>/<algo>/seed<n>/
  part3/results/
```

## Reproduce the main experiment (Part 2)

```bash
pip install -r requirements.txt
python run_experiments.py --workers 4        # all of Part 2 (Settings A-D)
python run_experiments.py --quick            # 2-minute smoke test first
```

`run_experiments.py` trains both algorithms in Settings A, B and C with 5 training
seeds each, evaluates every trained agent for 200 held-out episodes, evaluates
the Setting-B agents on an unseen maze (Setting D), and writes tables and
figures to `records/part2/results/`. With 4 CPU cores, it takes roughly 1-2 hours.

Single runs:

```bash
python train.py --algo q --setting A --seed 0              # -> records/part2/runs/A/q/seed0/
python evaluate.py records/part2/runs/A/q/seed0                     # 200 episodes, greedy
python evaluate.py records/part2/runs/B/ppo/seed0 --setting D       # transfer to unseen maze
python summarize.py
```

## Design decisions

These choices are written down before running the experiment. All of them can
be changed from the command line or in `pacman_rl/env.py`.

### Task: what is success and failure?
Clearing a whole maze is far beyond what any agent can learn with our budget,
so success rates would all be 0 and tell us nothing. We therefore use a
scaled-down task. One episode is **one life** of Ms. Pac-Man:

| Outcome | Definition |
|---|---|
| Success | She eats 50 dots (about one third of the maze) without losing her life |
| Failure | A ghost catches her (she loses the life) |
| Timeout | Neither happens within 2000 agent steps |

A simple hand-written policy (`baseline.py`: go to the nearest dot, avoid
ghosts, chase frightened ghosts) reaches success in about 40-60% of the
episodes, so the task is possible but not trivial. Its results are written to
`records/part2/results/baseline.csv` as a reference point for the learned agents.

### Reward
* Game points / 10, capped at 20 per step (dot = 1, power pellet = 5, ghost = 20).
  This keeps the power-pellet prediction (Part 1, State 3) testable.
* **Terminal success reward:** +25 when the episode ends in success.
* No extra penalty for losing a life (losing the life ends the episode, so all
  future reward is lost).
* `--reward-mode success` is available for a purely terminal reward, but agents
  will barely learn from it with our budget.

### Settings

| Setting | Transitions | Reward | Maze | Notes |
|---|---|---|---|---|
| A | deterministic (sticky actions p = 0) | score + success | 1st maze | |
| B | stochastic: **sticky actions, p = 0.25** (Machado et al., 2018) | same as A | 1st maze | only the transitions change |
| C | as B | as B **minus 0.05 per step** | 1st maze | compare with B using success/failure rates only |
| D | as B | as B | **unseen 2nd maze** | Setting-B agents, no further training |

* **Sticky actions:** with probability 0.25 the game repeats the previous
  action instead of the chosen one. This is the standard way to make Atari
  stochastic.
* **Varied start states:** the game starts with an intro of 65 agent steps in which
  nobody moves. After it we add 0-30 random no-op steps, so the ghosts start
  from different places. Without this, every episode in Setting A would be
  identical. Note: in Setting A, the only randomness in an evaluation episode
  is this start (31 possible starts), so many of the 200 episodes repeat.
* **Unseen maze (D):** the Atari game has four maze layouts, but normally you
  only see the 2nd after clearing the 1st. RAM address 0 stores the layout;
  we set it to 1 at reset. All four ALE "modes" use the same maze, so they
  cannot be used for this.

### State representation (the same for both agents)
Both agents get the **same information**, so differences come from the
algorithm, not from the input. Features are computed from the screen (walls,
dots) and RAM (positions of Ms. Pac-Man and the ghosts, frightened timer), and
are *egocentric*: they describe the situation around Ms. Pac-Man, not her
absolute position.

| Feature | Values |
|---|---|
| Which of the 4 directions are open (no wall) | 16 |
| Direction towards the nearest dot (shortest path in the maze) | 4 |
| For each direction: is a dangerous ghost within 16 units (64 px) and not getting farther away? | 16 |
| If ghosts are frightened: direction towards the nearest one within 40 units, else none | 5 |
| Direction towards the nearest power pellet within 40 units, else none | 5 |

* **Q-learning:** a table with one row per combination (25,600 states x 4 actions).
* **PPO:** the same features as a 22-dimensional 0/1 vector, fed into two
  small neural networks (actor and critic, 2 x 64 hidden units).
* **Actions:** UP, RIGHT, LEFT, DOWN for both agents (diagonals and NOOP removed).
* **Wall masking:** both agents can only choose directions without a wall. In
  the game, pressing into a wall makes Ms. Pac-Man stand still, and in tests
  this trapped the agents. Q-learning picks greedy/random actions only among
  open directions; PPO sets the probability of blocked directions to 0.
* **Consequence for Setting D:** because the features are egocentric, a
  Q-table learned on maze 1 still means something on maze 2. A tabular state
  based on *position* would carry no information to a new maze. This is a
  choice we made on purpose and must discuss in the report.

### Interaction budget
* **300,000 environment interactions per training run** (one interaction = one
  `env.step`, which is 4 game frames). This counts *all* steps, including the
  intro and no-op steps at the start of each episode.
* This is 1.2 million frames, about 2.4% of the 50 million frames used for DQN
  (Mnih et al., 2015). It is chosen so that the main experiment (30 runs) plus the
  Part 3 investigations (5 seeds per configuration) fit within a few hours on a laptop.

### Seeds
| Purpose | Seeds |
|---|---|
| Training runs | 0, 1, 2, 3, 4 (same list for both algorithms) |
| Training episodes | `run_seed * 1,000,000 + episode index` |
| Learning-curve evaluation (monitoring only) | 20,000,000 + 0..9 |
| Held-out final evaluation | 10,000,000 + 0..199 (same 200 for every agent) |

### Evaluation
* Parameters are never updated during evaluation.
* Main evaluation policy (as stated in Part 1): **greedy** Q-table actions and
  **highest-probability** (argmax) PPO actions.
* PPO is also evaluated with **sampled** actions; this is reported separately.
* Results are summarized **across trained agents** (mean ± standard deviation of the
  5 per-seed values), never by pooling 1000 episodes as if they were independent.

### Learning curves
* Every 25,000 interactions, the current agent is evaluated for 10 episodes
  (curve seeds, greedy/argmax).
* The curves are **not smoothed**. Thin lines show single seeds, thick lines show
  the mean across seeds.
* These evaluation steps are not counted in the training budget.

### Hyperparameters (same search allowance)
Both algorithms use standard default values from the literature, with **no tuning
for either algorithm** (equal allowance of zero). If you tune, give both the
same number of configurations and use only the curve seeds, never the held-out
evaluation seeds.

| Q-learning | | PPO | |
|---|---|---|---|
| learning rate α | 0.1 | learning rate | 3e-4 |
| discount γ | 0.99 | discount γ | 0.99 |
| ε schedule | 1.0 → 0.05, linear over the first 50% of the budget | GAE λ | 0.95 |
| Q init | 0 | clip | 0.2 |
| | | entropy coefficient | 0.01 |
| | | rollout / minibatch / epochs | 2048 / 256 / 4 |

## Outputs

Each run directory, e.g. `records/part2/runs/B/ppo/seed3/`, contains:

| File | Content |
|---|---|
| `config.json` | all settings, hyperparameters, seed, software versions, CPU, runtime |
| `train_episodes.csv` | every training episode: interactions so far, return, outcome, length |
| `curve.csv` | periodic evaluation for the learning curves |
| `ppo_updates.csv` | PPO losses, entropy, clip fraction per update |
| `model.npy` / `model.pt` | the trained Q-table / network |
| `eval_<setting>_<mode>.csv/.json` | the 200 held-out episodes and their summary |
| `traces_<setting>_<mode>.csv.gz` | every decision: features, chosen action, Q-values or action probabilities, positions (for Part 4) |

`records/part2/results/` contains `summary.md`, `summary.tex` (LaTeX table), `summary.csv`,
`per_agent.csv` and `curves_<setting>.png`. Every figure lists its source runs.

## Part 3 (mechanisms)

```bash
python run_part3.py --workers 4      # after run_experiments.py
```

All investigations run in Setting B with the same budget and 5 seeds. For each one,
only the listed factor changes; the default value is taken from the main runs.

| Investigation | Algorithm | Factor | Values (default in bold) |
|---|---|---|---|
| Exploration | Q-learning | ε decay (fraction of budget) | 0.1, **0.5**, 0.9 |
| Planning horizon | both | discount γ | 0.9, **0.99**, 0.999 |
| Policy exploration | PPO | entropy coefficient | 0.0, **0.01**, 0.05 |
| Update size | Q-learning | α | 0.02, **0.1**, 0.5 |
| Update size | PPO | learning rate | 1e-4, **3e-4**, 1e-3 |
| PPO data reuse | PPO | epochs per rollout | 1, **4**, 10 |

Single variants can also be trained by hand with `--out` (a separate folder), for example:

```bash
python train.py --algo q   --setting B --seed 0 --out records/part3/runs/eps_fast --eps-decay-frac 0.1
python train.py --algo ppo --setting B --seed 0 --out records/part3/runs/ent_0    --ent-coef 0.0
python train.py --algo ppo --setting B --seed 0 --out records/part3/runs/epochs_10 --epochs 10
python train.py --algo q   --setting B --seed 0 --out records/part3/runs/gamma_09 --gamma 0.9
python train.py --algo ppo --setting B --seed 0 --out records/part3/runs/lr_1e3   --lr 1e-3
```

## Known limitations
* Ghost positions come from RAM; the screen shows only one ghost per frame (flicker).
* RAM address 116 is a single frightened timer for all ghosts. A ghost that was
  eaten and came back is still treated as frightened until the timer ends.
* The features are coarse on purpose (the Q-table must stay small). Both agents
  suffer from the same aliasing: different situations can map to the same state.
* RAM addresses and the screen-to-RAM offsets were found by experiment
  (ale-py 0.12). They are listed at the top of `pacman_rl/maze.py` and `pacman_rl/env.py`.
