# Assignment 4: MsPacMan design decisions

The choices behind the game, the agents and the tuning in `score_game/`, with the reason for each. Row numbers stay the same between versions, so a reference such as "5.9" keeps pointing to the same row. A removed row keeps its number and says when it stopped applying.

**Main rule:** play the default MsPacMan as a normal player would, and go for score. A change from the default needs a reason.

**Why:** a simplified game has to be defended from the start. With an easy enough goal, such as eating one dot, every agent reaches 100%, but then it is no longer MsPacMan.

**Status:** *Fact* = how the game is, not a choice. *Decided* = our choice. *Assignment* = required by the assignment text. *Suggested* = proposed, not yet agreed.

## 1. The game

| # | Topic | Default game | Our choice | Status |
|---|---|---|---|---|
| 1.1 | Lives | 3 | 3, the game ends at game over | Decided |
| 1.2 | Goal | Highest score | Highest score | Decided |
| 1.3 | Cleared maze | Continue in the next maze | Same as default | Decided |
| 1.4 | Side tunnels | Yes | Yes | Fact |
| 1.5 | Steps per choice | 4 frames, which is 15 choices per second of game time | Same as default. The game does not run faster, and the speed is an advantage a machine may have | Decided |
| 1.6 | Actions | 9 (standing still and diagonals included) | 4 directions. A normal player does not use diagonals, and more actions would need more training | Decided |
| 1.7 | Walls | Can be pressed into, and she then mostly keeps going | Same as default. No rule blocks any action | Decided |
| 1.8 | Start of a game | Always the same | 0–30 random moves, each in one of the 4 directions, in training and testing. All settings in a round get the same list of starts, and points from these moves count. Gives 335 different starts in 400 games, against 31 for waiting | Decided |
| 1.9 | Randomness during a game | Sticky actions all the time | Setting A: none after the start. Settings B, C and D: sticky actions at the same strength through the whole game, as in the default game | Decided |
| 1.10 | Extra life | One extra life at 10,000 points | Same as default. It counts in the "lives left" input (shown as at most 3) and the life penalty | Fact |

## 2. Success and reward

| # | Topic | Our choice | Status |
|---|---|---|---|
| 2.1 | Success | Reach X points before game over. The assignment needs a success rate, so a limit is required | Decided |
| 2.2 | X | 6,952, a human's score (2.9). If more than 90% of the tuning test games of **either** model's winner reach it, X is the higher published human score, 15,693, for **both** models, so that there are successes and failures for Part 4. X is set by this rule after the tuning and before A–D (`winners.json`, `share_at_6952`), and is otherwise changed only if a better source gives another human number. Nothing in the tuning uses it | Decided |
| 2.3 | Failure | Game over before X points | Decided |
| 2.4 | Does the game stop at X? | No, it is played to game over. Success is counted from the final score | Decided |
| 2.5 | Reward | Testing and choosing the best setting: the game's own points, 1:1 and with no limit per step. Training: the same points plus the two guides in 5.14 and 5.16. PPO can scale the numbers inside its own learning, because large numbers make it learn badly | Decided |
| 2.6 | Success reward | None in tuning, so that X can be set afterwards. In A–D a small fixed reward (250 points) when the game ends with success, because Setting A asks for a terminal success reward | Decided |
| 2.7 | Reported for each agent | Average score, success, failure and game length | Assignment |
| 2.8 | Number used in tuning | The average score over several trained agents, never the best agent | Decided |
| 2.9 | Which human score | 6,952, the lower of the two published numbers. To be checked against the original papers | Decided |

**Reference scores**

| # | Player | Average score |
|---|---|---|
| 2.10 | Random play (20 games, measured on the default game) | 276 |
| 2.11 | Random play, published | 307 |
| 2.12 | DQN (a deep learning agent), published | 2,311 |
| 2.13 | Human, published, two papers | 6,952 and 15,693 |

Published numbers come from [endtoend.ai](https://www.endtoend.ai/envs/gym/atari/ms-pacman/) and must be checked against the original papers before they are cited.

## 3. Mazes

| # | Topic | Our choice | Status |
|---|---|---|---|
| 3.1 | Number of mazes | 4, clearly different: only 51–66% of the walkable cells are the same between any two. In the game, levels 1–2 are maze 1, 3–4 maze 2, 5–6 maze 3, and then maze 4 comes | Fact |
| 3.2 | Training | Each game starts in maze 1, 2 or 3 at random | Decided |
| 3.3 | Held-out maze | Maze 4. A training game that reaches it is stopped before the agent sees anything of it, and the same holds for the test games in the tuning | Decided |
| 3.4 | Setting D | Only maze 4: the game starts there and stays there after every clear. It has not been used for training or tuning, only to check that the game is read correctly. No training | Decided |
| 3.5 | Testing in A–C | Starts in maze 1, the default beginning, and plays on | Decided |

## 4. What the agents see

| # | Topic | Our choice | Status |
|---|---|---|---|
| 4.1 | Source | Read from the game's memory, but only what a player can see on the screen. The memory replaces reading the picture and must not give an edge over a player | Decided |
| 4.2 | Inputs | Fixed and the same for Q-learning and PPO, so that a difference comes from the algorithm: open directions, nearest dot, power pellet, blue ghost, danger in two levels, fruit, dots left, lives left, all seen from where she is heading. The agents must learn to play and not be given the solution | Decided |
| 4.3 | "Safest direction" input | Not used, because it gives the solution | Decided |
| 4.4 | One value per direction | Both models use this form: Q-learning has one table row per direction, and PPO uses the same small network for each direction. It keeps the table small and gives the two models the same representation | Decided |
| 4.5 | New inputs | Where the fruit is, lives left, dots left in the maze, and whether a ghost is blue, not how long it lasts. She sees the same as a player, see 4.1 | Decided |

## 5. Settings A–D and training

| # | Topic | Our choice | Status |
|---|---|---|---|
| 5.1 | A | Nothing random in the game itself | Assignment |
| 5.2 | B | Random: 25% sticky actions per frame. This is the default game | Assignment |
| 5.3 | C | B with a step penalty of 0.5 per step, moving or not. Part 2 used 0.05 when a dot was worth 1, and a dot is now 10. It is not used in tuning, A, B or D, because the only difference between B and C must be this penalty | Assignment (penalty), Decided |
| 5.4 | D | The B agents on the held-out maze | Assignment |
| 5.5 | Budget | 300,000 steps per training run in every round, the same for both models and all settings. A–D and Part 3 use the same. A higher budget (up to 1 million) was considered, but it doubles the training time or more | Decided |
| 5.6 | Seeds | 5 training seeds, and 200 test games on a new list that is not used for anything else | Assignment |
| 5.7 | Tuning | Redone for the new game, see section 6 | Decided |
| 5.8 | Predictions | Part 1 stays unchanged. Changed choices are noted, as section 1.3 of the report says | Assignment |
| 5.9 | States 1–3 | The rule for finding them in the recorded games is written before anyone studies the recordings | Suggested |
| 5.10 | Exploration (random choices while learning) | High at the start of each training run and lower later, counted in percent of that run's steps. Not used in testing. How fast it falls is part of the tuning. It starts over in every run, so the number of tuning rounds does not matter | Decided |
| 5.11 | Guides in training | Only soft guides, meaning penalties and rewards, and none of them in testing. Hard rules that block actions are not used, because the agent would meet actions in testing that it never tried | Decided |
| 5.12 | Penalty for standing still | Not used, see 9.1 | Decided |
| 5.13 | Gamma (how far ahead points count) | Tuned for each model, see 6.15. A cleared maze takes about 600 choices and is almost invisible at 0.99 | Decided |
| 5.14 | Life penalty | Tuned for each model, see 6.15, in training only. A game does not end when a life is lost, and no points or dots are taken back, so the cost of dying is easy to miss | Decided |
| 5.15 | Stuck games | A game stops after 5,000 steps without points and without a lost life. It is counted as a timeout and reported, and gives no extra punishment. In 300 full games the longest gap between points was 679 | Decided |
| 5.16 | Ghost value | Tuned for each model, see 6.15, in training only. A ghost is always worth something, so State 3 in the predictions can be tested | Decided |
| 5.17 | Nothing else is tuned | Pellet and fruit value, maze bonus, danger distance, network and batch sizes stay fixed, to keep the search small. See section 9 | Decided |

## 6. Tuning

Many settings are tried, the best are kept, and new settings are made from them. Every run starts from zero, so nothing learned is carried over between runs. Only settings are kept. Only Setting B is used, and never maze 4. Rounds 1–3 used the method in 6.2; rounds 4–6 used CMA-ES (6.19), because several of the best settings sat on a limit after round 3 and the search needed a method that sets its own step sizes. The stop rule (6.5) ended the tuning after round 6. Every run is in `score_game/tuning/results.jsonl`, and every choice the loop made is in `score_game/tuning/decisions.log`.

| # | Topic | Our choice | Status |
|---|---|---|---|
| 6.1 | Round 1 | 20 settings per model: 18 with every value drawn at random within its limits, and 2 known ones (Part 2's settings and the setup chosen in the earlier tuning, both adapted to the new game) as a floor to beat | Decided |
| 6.2 | Rounds 2–3 | 20 settings again, made from the 4 best: 4 the same with new seeds, 6 **fine** (all 8 values move 5–15% of their range), 4 **medium** (2–3 values move 10–30%), 3 **mix** (two of the best, each value from one of them), 3 **big** (1–3 values completely new within the limits). Up or down with equal chance. No fully random newcomers after round 1. 20 and 4 keep 20%, as in Population Based Training (8.10). From round 4, replaced by 6.19. The exact code that ran rounds 2–3 is `score_game/tuning/tune_before_cma.py` | Decided |
| 6.3 | Why not only the best | Keeping several avoids getting stuck on one | Decided |
| 6.4 | Lucky settings | The 4 kept are trained again with new seeds in every round. In the earlier tuning one setting fell from 83% to 41% with new seeds. Old and new results are averaged, if the budget is the same | Decided |
| 6.5 | When to stop | A new round starts while the 4 kept settings of **either** model, re-tested on new seeds, improve by 10% or more on the round before. The seed noise of that number was about ±6–7% for Q-learning and ±4% for PPO. After round 6 it was 4.2% for Q-learning and −6.5% for PPO, which ended the tuning | Decided |
| 6.6 | Equal search | Each model is tuned on its own, and both always get the same number of rounds, settings, seeds and steps | Assignment (rule 6) |
| 6.7 | Final choice | The winner of each model is the setting with the highest average score in the last round, the kept ones counted with their earlier seeds. No extra final run: its honest score comes from Setting B in A–D, on new seeds and new games. Saved in `score_game/tuning/winners.json` | Decided |
| 6.8 | Continuing later | The tuning can be stopped and continued from the last finished round. Every run, setting and seed is saved, each round's seeds come from its round number, and the CMA-ES state is saved for each round (`cma_round*.json`) | Decided |
| 6.9 | Size of the variations (rounds 2–3) | Random between the limits in 6.2. Chosen by simulation of our own loop (`tuning/sim/sim.py`–`sim4.py`): smaller steps (2–10%) were worse, fully random newcomers did not help, and the good plans differed by less than 1 point of 100. From round 4, CMA-ES sets the step size (6.19) | Decided |
| 6.10 | Continuing after A–D has been run | Needs a new list of test games, because the old results have been seen (rule 7). X from 2.2 stays the same | Suggested |
| 6.11 | Seeds in a round | 5 for every setting. All settings meet the same seeds and games | Decided |
| 6.12 | Not all the same | Rounds 1–3 only: the 4 kept had to differ from each other by at least 10% of the range in at least 2 values. Removed from round 4, because the kept settings no longer steer the search: only the 16 drawn settings update the CMA-ES cloud | Decided |
| 6.13 | What is tuned | 6 learning values and the 2 guides for each model (6.15). Nothing is shared | Decided |
| 6.14 | Score of the shared part | Not used, because nothing is shared (6.13) | Decided |
| 6.15 | Values and limits | See the table below. Rounds 1–3 used the first limits. From round 4 the limits were widened, because several of the best settings sat on a limit: hard limits where a value has no meaning past them, otherwise a buffer about 5–10 times past the old edge, where a run would only crash or waste time | Decided |
| 6.16 | Test games per agent in tuning | 20 for each trained agent, so 100 for each setting | Decided |
| 6.17 | How a value moves | As a share of its range. Values that differ a lot in size move on a log scale, and gamma moves by its distance to 1. Rounds 1–3: a value that would cross a limit stops at the limit. From round 4: a draw outside the limits is drawn again, as Hansen recommends over clipping (8.24, appendix B.5) | Decided |
| 6.18 | 20 settings per round | Rounds 1–3: chosen by simulation of our own loop on made-up problems. For the same number of runs, more rounds of 20 gave a better winner than fewer rounds of 40 in all cases that finished (for example 0.81 against 0.75 after 80 settings, where 1 is perfect). For CMA-ES, 20 per round (16 drawn + 4 kept) also did better than 14 or 30 (`tuning/sim/sim6.py`). The team has limited training resources, so learning after every small round fits | Decided |
| 6.19 | CMA-ES (rounds 4–6) | Each model has a "cloud": a centre, a shape and a size. Each round, 16 settings are drawn from the cloud, and the 4 kept settings are re-tested with their exact values. The 8 best of the 16 drawn move the cloud. The standard constants from Hansen (8.24) for 16 drawn and 8 used are applied. The cloud starts from the best 10 settings of round 3. The 4 kept are the 4 best by average score. A setting whose runs all failed is ranked last. Epochs, a whole number, always varies by at least ±1. In simulations of our loop (`tuning/sim/sim5.py`–`sim7.py`), CMA-ES gave a better winner than continuing with 6.2, and forcing some draws further out made it worse | Decided |

**Values and limits (6.15)**

| # | Value | Model | Rounds 1–3 | From round 4 | Scale |
|---|---|---|---|---|---|
| 6.20 | Life penalty (points) | Both | 0–500 | 1–5,000 | even, log from round 4 |
| 6.21 | Ghost value (× the game's points) | Both | 0.25–4 | 0.01–20 | log |
| 6.22 | Gamma | Both | 0.8–0.999 | 0.5–0.9999 | log, distance to 1 |
| 6.23 | Learning rate | Q-learning | 0.005–0.5 | 0.0001–1 | log |
| 6.24 | Exploration at the end | Q-learning | 0.5%–10% | 0.01%–30% | log |
| 6.25 | How long exploration falls (share of the run) | Q-learning | 10%–90% | 1%–100% | even |
| 6.26 | How far back credit goes (λ) | Q-learning | 0–0.95 | 0–0.99 | even |
| 6.27 | Start value (points) | Q-learning | 0–100 | 0–1,000 | even |
| 6.28 | Learning rate | PPO | 0.00005–0.002 | 0.000001–0.005 | log |
| 6.29 | Entropy | PPO | 0.0005–0.05 | 0.00001–0.1 | log |
| 6.30 | Epochs | PPO | 2–10 | 1–15 | whole numbers |
| 6.31 | GAE λ | PPO | 0.8–0.99 | 0.5–0.999 | even, distance to 1 from round 4 |
| 6.32 | Reward scale | PPO | 0.001–0.1 | 0.0001–1 | log |

**When may a rule be changed?** Freely before the results of that step exist. After that only for a bug, or between tuning rounds, written down and equal for both models. Never to help one model, never by removing seeds, and never after the A–D results are seen, unless as a new version with a new test list. Every change is logged here.

## 8. Sources

| # | Source | Used for | Read |
|---|---|---|---|
| 8.1 | [Machado et al. 2018, Revisiting the ALE (short version)](https://www.ijcai.org/proceedings/2018/0787.pdf) | Sticky actions (1.9), game over (1.1), budget in frames (5.5), best results must be measured again (6.7) | Whole paper |
| 8.2 | Lecture 09-1 Reinforcement Learning (course slides) | Exploration (5.10), reward and real goal (2.5), gamma (5.13), average and failures (2.7) | Whole lecture |
| 8.3 | Lecture 09-2 Reinforcement Learning (course slides) | PPO values work together, so several are changed at once (6.2) | Whole lecture |
| 8.4 | [Eimer et al. 2023, Hyperparameters in RL](https://arxiv.org/html/2306.01324v1) | Tuning seeds (6.11), separate seeds and equal budget (6.6) | Summary of most of the page |
| 8.5 | [Quantity vs. Quality 2020, tuning PPO](https://arxiv.org/abs/2007.14604) | Stopping runs early may hurt (6.1). Two simple tasks only | Search result |
| 8.6 | [Go-Explore 2021, First return, then explore](https://arxiv.org/pdf/2004.12919) | No-op starts (1.8), exploring from later places (considered, not used) | Search result |
| 8.7 | [Raffin, hyperparameter tuning in practice](https://araffin.github.io/post/optuna/) | Testing the best settings again (6.4) | Search result |
| 8.8 | [Blog test of no-op starts](https://towardsdatascience.com/are-the-space-invaders-deterministic-or-stochastic-595a30becae2/) | Waiting at the start gives little variation (1.8) | Search result |
| 8.9 | [endtoend.ai, Ms. Pac-Man scores](https://www.endtoend.ai/envs/gym/atari/ms-pacman/) | Reference scores (2.11–2.13). Secondary source | Search result |
| 8.10 | [Jaderberg et al. 2017, Population Based Training](https://arxiv.org/abs/1711.09846), read through [ar5iv](https://ar5iv.labs.arxiv.org/html/1711.09846) | 20 settings and keeping 20% (6.1, 6.2). "a population size of between 20 and 40 is sufficient to see strong and consistent improvements" and "if the population size is too small (10 or below) we tend to encounter higher variance" (section 4.4); bottom 20% replaced by the "top 20% of the population" (section 4.1.1). It is about neural networks trained in parallel, and says nothing about 20/4 being better than 40/8 | Quoted passages |
| 8.11 | [Hyperband demo (Jamieson)](https://homes.cs.washington.edu/~jamieson/hyperband.html) | Keeping about 1/3 per round (6.2) | Search result |
| 8.12 | [Cross-entropy method, practical guide](https://metricgate.com/docs/cross-entropy-method-optimization/) | Keeping 5–20%; too few risks getting stuck (6.3). Not peer-reviewed | Search result |
| 8.13 | [RiskSpan, tuning machine learning models](https://riskspan.com/tuning-machine-learning-models/) | 60 random tries give about 95% chance of the best 5%. 40 tries give 87%, 20 tries 64% (own calculation) | Search result |
| 8.14 | [Ng, Harada and Russell 1999, reward shaping (quoted in a later paper)](https://arxiv.org/pdf/2505.12611) | Extra rewards can change which play is best, except one special form. Background for 5.14, 5.16 and 5.3 | Search result |
| 8.15 | [What Matters in On-Policy RL, review of the paper](https://vitalab.github.io/article/2020/07/02/What_Matters_in_RL.html) | Gamma is among the most important PPO values (6.22), starting at 0.99; learning rate 0.0003; GAE λ 0.9 | Search result, review and not the paper |
| 8.16 | [Agarwal et al. 2021, few runs](https://arxiv.org/abs/2108.13264) | Report the spread and not only the average when there are few runs (2.7). They recommend the interquartile mean | Search result |
| 8.17 | [Procgen, Cobbe et al.](https://arxiv.org/pdf/1912.01588) | Many training levels are needed to generalise, so three mazes are few (3.2, 5.4) | Search result |
| 8.18 | [Awad et al. 2021, DEHB](https://ar5iv.labs.arxiv.org/html/2105.09821) | Considered, not used: plain differential evolution was slightly worse than random search on a reinforcement learning test, and clearly worse on very noisy ones | Detailed summary of the page |
| 8.19 | [Rechenberg's 1/5 rule, described in a later paper](https://arxiv.org/pdf/1504.03212) | Larger steps when many tries succeed, smaller when few do (6.17) | Search result |
| 8.20 | [Population Based Training in practice, Syne Tune](https://syne-tune.readthedocs.io/en/latest/_apidoc/syne_tune.optimizer.schedulers.pbt.html) | Standard change: ×0.8 or ×1.2, and 25% new random values (6.2). Eimer et al. (8.4) warn that population methods can be unreliable and overfit, but those also copy the trained players | Search result |
| 8.21 | [Doerr et al. 2017, heavy-tailed mutation](https://arxiv.org/abs/1703.03334) | Mixing many small changes with a few large ones helps escape bad areas (6.2). Theory on other problems, and some doubt it helps everywhere | Search result |
| 8.22 | [Random immigrants in genetic algorithms, an overview of the idea](https://arxiv.org/pdf/1912.08397) | Fully random newcomers are often not good enough to win late; making newcomers by changing the best ones is an alternative (6.2). Mostly about changing environments | Search result |
| 8.23 | [Yang 2007, elitism-based immigrants](https://www.cs.le.ac.uk/people/syang/Papers/EvoSTOC07_1.pdf) | Newcomers made by changing the best ones instead of fully random ones (6.2). About changing environments | Search result |
| 8.24 | [Hansen 2016, The CMA Evolution Strategy: A Tutorial](https://arxiv.org/abs/1604.00772) | The CMA-ES method and its default constants (6.19); drawing again instead of clipping at the limits, appendix B.5 (6.17) | The algorithm summary, the default constants and appendix B.5 |

Sources marked *Search result* have only been read as excerpts and must be read before they are cited in the report. Rows 8.10–8.13 and 8.19–8.23 support the method of rounds 1–3 (6.2).

## 9. Dropped, possible improvements

Not needed if the results are good enough. Each can be added later if it is done for both models.

| # | Idea | Why it was dropped |
|---|---|---|
| 9.1 | Penalty for standing still | One more value to search. Standing still already costs points, and recorded games will show if it becomes a problem |
| 9.2 | Own values for pellet, fruit and clearing a maze | Same. Dots left and a high gamma already show the value of clearing |
| 9.3 | Inputs as on/off options | The search gets larger, and fixed inputs are easier to defend |
| 9.4 | Danger distance and other radii as options | Never tested, but every extra value makes the others harder to judge |
| 9.5 | Network size, rollout, minibatch as options | In the earlier game larger networks and shorter rollouts hurt, and minibatch had no clear effect |
| 9.6 | Learning rate that falls over the run (Q-learning) | No clear effect in the earlier game. It could reduce the swings late in training |
| 9.7 | More settings per round (40/8) | The simulation favoured more rounds of 20 (6.18). The loop can be continued (6.8) |
| 9.8 | Budget above 300,000 steps | More time (5.5) |
| 9.9 | Keeping the best moment of a run instead of the last | Hides that a run is unstable |
| 9.10 | Rules that block actions, such as no turning round | Gives the agent the solution and is not the default game |
| 9.11 | Exploration that rises later in a game, or starting from saved places (Go-Explore) | Complex, and the second changes the game |
| 9.12 | Nine actions (diagonals, standing still) | More training needed, and a normal player does not use them |
| 9.13 | Separate, better search methods (for example Bayesian) | Not needed for this size |
| 9.14 | Step penalty in the tuning | It conflicts with C, and such rewards can change which play is best (8.14). Gamma already makes early points worth more |

## 10. Open

| # | Task | When | Why |
|---|---|---|---|
| 10.5 | Write the rule for finding States 1–3 in the recorded games (5.9) | Before anyone studies the recordings | A rule written after seeing the games can be bent to fit them |
| 10.6 | Design and run the diagnostic experiment for Part 3 | After the Part 3 results | Required by Part 3 |
| 10.7 | AI-assistance log | Before submission | Required by the assignment |
| 10.8 | Run the reference players (`baseline.py`) | Any time after the winners are picked | A reference line for the results; no training |
| 10.9 | Decide whether to add seeds 5–9 to A–D and Part 3 | After A–D | More seeds make differences easier to see; the scripts skip finished runs, so they can be added later |
