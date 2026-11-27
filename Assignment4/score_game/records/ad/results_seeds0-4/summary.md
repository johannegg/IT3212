| Test | Trained in | Model | Policy | Agents | Score | Return | Success | Failure | Timeout | Length (success) | Length (failure) | Training steps | Training time (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | A | PPO | argmax | 5 | 21736 ± 3876 | 23242 ± 3961 | 0.94 ± 0.06 | 0.06 ± 0.06 | 0.00 ± 0.00 | 4064 ± 496 | 1314 ± 158 | 300003 | 11.6 ± 0.1 |
| A | A | PPO | sampled | 5 | 20195 ± 1885 | 21739 ± 1896 | 0.97 ± 0.01 | 0.03 ± 0.01 | 0.00 ± 0.00 | 4014 ± 268 | 1432 ± 83 | 300003 | 11.6 ± 0.1 |
| A | A | Q-learning | greedy | 5 | 5522 ± 1000 | 5931 ± 1251 | 0.26 ± 0.17 | 0.74 ± 0.17 | -0.00 ± 0.00 | 1455 ± 134 | 1169 ± 50 | 300000 | 7.1 ± 0.0 |
| B | B | PPO | argmax | 5 | 19746 ± 1318 | 21269 ± 1315 | 0.95 ± 0.01 | 0.05 ± 0.01 | 0.00 ± 0.00 | 3791 ± 217 | 1159 ± 91 | 300001 | 11.5 ± 0.0 |
| B | B | PPO | sampled | 5 | 18555 ± 1164 | 20048 ± 1205 | 0.93 ± 0.03 | 0.07 ± 0.03 | 0.00 ± 0.00 | 3778 ± 128 | 1294 ± 121 | 300001 | 11.5 ± 0.0 |
| B | B | Q-learning | greedy | 5 | 5169 ± 395 | 5427 ± 488 | 0.16 ± 0.06 | 0.84 ± 0.06 | 0.00 ± 0.00 | 1448 ± 261 | 1132 ± 93 | 300000 | 7.3 ± 0.0 |
| C | C | PPO | argmax | 5 | 15908 ± 7018 | 15541 ± 7055 | 0.82 ± 0.20 | 0.18 ± 0.20 | 0.00 ± 0.00 | 3678 ± 304 | 1389 ± 349 | 300002 | 7.8 ± 0.1 |
| C | C | PPO | sampled | 5 | 16311 ± 6366 | 15921 ± 6413 | 0.83 ± 0.18 | 0.17 ± 0.18 | 0.00 ± 0.00 | 3764 ± 260 | 1435 ± 270 | 300002 | 7.8 ± 0.1 |
| C | C | Q-learning | greedy | 5 | 5327 ± 351 | 5074 ± 460 | 0.19 ± 0.05 | 0.81 ± 0.05 | 0.00 ± 0.00 | 1247 ± 220 | 1079 ± 136 | 300005 | 5.7 ± 0.1 |
| D | B | PPO | argmax | 5 | 12796 ± 1104 | 14071 ± 1174 | 0.80 ± 0.05 | 0.20 ± 0.05 | 0.00 ± 0.00 | 2612 ± 103 | 1227 ± 83 | 300001 | 11.5 ± 0.0 |
| D | B | PPO | sampled | 5 | 12302 ± 1194 | 13582 ± 1299 | 0.80 ± 0.07 | 0.20 ± 0.07 | 0.00 ± 0.00 | 2540 ± 128 | 1268 ± 44 | 300001 | 11.5 ± 0.0 |
| D | B | Q-learning | greedy | 5 | 4536 ± 301 | 4662 ± 338 | 0.08 ± 0.02 | 0.92 ± 0.02 | 0.00 ± 0.00 | 1148 ± 105 | 980 ± 59 | 300000 | 7.3 ± 0.0 |

Mean ± standard deviation across trained agents (one value per agent, 200 held-out games each). Score and success use the game's real points. Return = points, +1,600 once when the score reaches X, and -0.5 per decision in C; it is not comparable between B and C. Success rate = share of an agent's 200 games with at least X points, then the mean over agents; success = at least 6952 points. Training returns use the guides of each model and are not comparable between models.