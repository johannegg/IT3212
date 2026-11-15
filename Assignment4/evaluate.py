"""Held-out evaluation of a trained agent (no learning during evaluation).

Examples:
  python evaluate.py records/part2/runs/A/q/seed0                         # 200 episodes, greedy
  python evaluate.py records/part2/runs/B/ppo/seed0 --mode sampled        # extra evaluation mode
  python evaluate.py records/part2/runs/B/q/seed0 --setting D             # transfer to the unseen maze
"""
import argparse
import json

from pacman_rl.experiment import evaluate_run

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--setting", choices=["A", "B", "C", "D"], default=None, help="default: the training setting")
p.add_argument("--episodes", type=int, default=200)
p.add_argument("--mode", choices=["greedy", "argmax", "sampled"], default=None,
               help="default: greedy for Q-learning, argmax (highest probability) for PPO")
p.add_argument("--no-traces", action="store_true")
a = p.parse_args()
meta = evaluate_run(a.run_dir, a.setting, a.episodes, a.mode, not a.no_traces)
print(json.dumps({k: meta[k] for k in ["mode", "success", "failure", "timeout", "mean_return", "mean_length"]}, indent=1))
