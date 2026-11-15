| Eval | Trained in | Algorithm | Policy | Seeds | Success | Failure | Timeout | Return | Length (success) | Length (failure) | Train steps | Train time (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | A | PPO | argmax | 5 | 0.56 ± 0.23 | 0.44 ± 0.23 | 0.00 ± 0.00 | 118.2 ± 13.7 | 240 ± 20 | 273 ± 22 | 300000 ± 0 | 4.5 ± 0.2 |
| A | A | PPO | sampled | 5 | 0.66 ± 0.20 | 0.34 ± 0.20 | 0.00 ± 0.00 | 115.9 ± 10.1 | 259 ± 24 | 251 ± 17 | 300000 ± 0 | 4.5 ± 0.2 |
| A | A | Q-learning | greedy | 5 | 0.03 ± 0.07 | 0.97 ± 0.07 | 0.00 ± 0.00 | 22.7 ± 11.4 | 180 ± 0 | 202 ± 40 | 300000 ± 1 | 4.4 ± 0.2 |
| B | B | PPO | argmax | 5 | 0.69 ± 0.18 | 0.31 ± 0.18 | 0.00 ± 0.00 | 112.2 ± 11.9 | 237 ± 31 | 229 ± 16 | 300013 ± 29 | 4.8 ± 0.2 |
| B | B | PPO | sampled | 5 | 0.71 ± 0.16 | 0.29 ± 0.16 | 0.00 ± 0.00 | 113.6 ± 7.4 | 251 ± 19 | 240 ± 10 | 300013 ± 29 | 4.8 ± 0.2 |
| B | B | Q-learning | greedy | 5 | 0.01 ± 0.01 | 0.99 ± 0.01 | 0.00 ± 0.00 | 21.1 ± 8.7 | 225 ± 102 | 195 ± 47 | 300013 ± 19 | 4.3 ± 0.1 |
| C | C | PPO | argmax | 5 | 0.51 ± 0.27 | 0.49 ± 0.27 | 0.00 ± 0.00 | 96.4 ± 9.7 | 250 ± 29 | 242 ± 27 | 300023 ± 38 | 4.9 ± 0.0 |
| C | C | PPO | sampled | 5 | 0.59 ± 0.20 | 0.41 ± 0.20 | 0.00 ± 0.00 | 98.3 ± 11.8 | 265 ± 20 | 235 ± 22 | 300023 ± 38 | 4.9 ± 0.0 |
| C | C | Q-learning | greedy | 5 | 0.01 ± 0.01 | 0.99 ± 0.01 | 0.00 ± 0.00 | 20.4 ± 15.0 | 328 ± 54 | 199 ± 64 | 300016 ± 37 | 4.0 ± 0.5 |
| D | B | PPO | argmax | 5 | 0.41 ± 0.31 | 0.59 ± 0.31 | 0.00 ± 0.00 | 70.5 ± 40.2 | 286 ± 31 | 253 ± 72 | 300013 ± 29 | 4.8 ± 0.2 |
| D | B | PPO | sampled | 5 | 0.51 ± 0.22 | 0.49 ± 0.22 | 0.00 ± 0.00 | 88.1 ± 24.5 | 289 ± 12 | 259 ± 45 | 300013 ± 29 | 4.8 ± 0.2 |
| D | B | Q-learning | greedy | 5 | 0.00 ± 0.01 | 1.00 ± 0.01 | 0.00 ± 0.00 | 12.8 ± 8.9 | 228 ± 18 | 155 ± 48 | 300013 ± 19 | 4.3 ± 0.1 |

Values are mean ± standard deviation across trained agents (one value per training seed); each agent was evaluated for 200 held-out episodes. Returns in Setting C include the step penalty and are not comparable with A/B; compare success and failure rates instead. Hardware: arm.
