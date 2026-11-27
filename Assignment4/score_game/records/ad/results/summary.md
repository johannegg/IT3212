| Test | Trained in | Model | Policy | Agents | Score | Return | Success | Failure | Timeout | Length (success) | Length (failure) | Training steps | Training time (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | A | PPO | argmax | 15 | 21084 ± 2553 | 22610 ± 2588 | 0.95 ± 0.03 | 0.05 ± 0.03 | 0.00 ± 0.00 | 3852 ± 385 | 1168 ± 179 | 300003 | 11.2 ± 0.6 |
| A | A | PPO | sampled | 15 | 20360 ± 1757 | 21885 ± 1776 | 0.95 ± 0.02 | 0.05 ± 0.02 | 0.00 ± 0.00 | 3904 ± 282 | 1297 ± 143 | 300003 | 11.2 ± 0.6 |
| A | A | Q-learning | greedy | 15 | 5930 ± 985 | 6430 ± 1245 | 0.31 ± 0.17 | 0.69 ± 0.17 | 0.00 ± 0.00 | 1520 ± 210 | 1133 ± 75 | 300001 | 7.0 ± 0.4 |
| B | B | PPO | argmax | 15 | 19398 ± 1841 | 20894 ± 1883 | 0.94 ± 0.04 | 0.07 ± 0.04 | 0.00 ± 0.00 | 3742 ± 277 | 1202 ± 172 | 300001 | 11.2 ± 0.6 |
| B | B | PPO | sampled | 15 | 18467 ± 1856 | 19949 ± 1897 | 0.93 ± 0.03 | 0.07 ± 0.03 | -0.00 ± 0.00 | 3686 ± 187 | 1282 ± 86 | 300001 | 11.2 ± 0.6 |
| B | B | Q-learning | greedy | 15 | 5418 ± 519 | 5740 ± 663 | 0.20 ± 0.09 | 0.80 ± 0.09 | 0.00 ± 0.00 | 1390 ± 186 | 1126 ± 87 | 300008 | 7.2 ± 0.3 |
| C | C | PPO | argmax | 15 | 18396 ± 4373 | 18081 ± 4387 | 0.90 ± 0.12 | 0.10 ± 0.12 | 0.00 ± 0.00 | 3714 ± 278 | 1291 ± 229 | 300001 | 8.0 ± 0.3 |
| C | C | PPO | sampled | 15 | 18004 ± 3891 | 17699 ± 3917 | 0.90 ± 0.12 | 0.10 ± 0.12 | 0.00 ± 0.00 | 3705 ± 287 | 1304 ± 211 | 300001 | 8.0 ± 0.3 |
| C | C | Q-learning | greedy | 15 | 5313 ± 619 | 5059 ± 762 | 0.20 ± 0.09 | 0.80 ± 0.09 | -0.00 ± 0.00 | 1279 ± 194 | 1099 ± 121 | 300004 | 6.0 ± 0.3 |
| D | B | PPO | argmax | 15 | 12949 ± 1366 | 14217 ± 1450 | 0.79 ± 0.06 | 0.21 ± 0.06 | 0.00 ± 0.00 | 2657 ± 180 | 1241 ± 82 | 300001 | 11.2 ± 0.6 |
| D | B | PPO | sampled | 15 | 12585 ± 1704 | 13871 ± 1809 | 0.80 ± 0.07 | 0.20 ± 0.07 | 0.00 ± 0.00 | 2569 ± 164 | 1247 ± 47 | 300001 | 11.2 ± 0.6 |
| D | B | Q-learning | greedy | 15 | 4722 ± 447 | 4895 ± 531 | 0.11 ± 0.06 | 0.89 ± 0.06 | 0.00 ± 0.00 | 1131 ± 100 | 969 ± 49 | 300008 | 7.2 ± 0.3 |

Mean ± standard deviation across trained agents (one value per agent, 200 held-out games each). Score and success use the game's real points. Return = points, +1,600 once when the score reaches X, and -0.5 per decision in C; it is not comparable between B and C. Success rate = share of an agent's 200 games with at least X points, then the mean over agents; success = at least 6952 points. Training returns use the guides of each model and are not comparable between models.