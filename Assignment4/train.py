"""Train one agent.

Examples:
  python train.py --algo q   --setting A --seed 0
  python train.py --algo ppo --setting B --seed 3 --budget 300000
  python train.py --algo q   --setting B --seed 0 --out records/part3/runs/my_test --eps-decay-frac 0.2
"""
import argparse

from pacman_rl.experiment import DEFAULTS, train

p = argparse.ArgumentParser()
p.add_argument("--algo", choices=["q", "ppo"], required=True)
p.add_argument("--setting", choices=["A", "B", "C"], required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--budget", type=int, default=300_000, help="environment interactions (agent steps incl. resets)")
p.add_argument("--out", default="records/part2/runs", help="runs are saved in <out>/<setting>/<algo>/seed<seed>")
p.add_argument("--eval-every", type=int, default=25_000)
p.add_argument("--curve-episodes", type=int, default=10)
p.add_argument("--step-penalty", type=float, default=None, help="override the step penalty of the setting")
for algo, hp in DEFAULTS.items():
    for k, v in hp.items():
        flag = "--" + k.replace("_", "-")
        try:
            p.add_argument(flag, type=type(v), default=None)
        except argparse.ArgumentError:
            pass   # shared names such as --gamma
args = p.parse_args()

overrides = {k: getattr(args, k) for k in DEFAULTS[args.algo] if getattr(args, k, None) is not None}
setting_overrides = {"step_penalty": args.step_penalty} if args.step_penalty is not None else None
run_dir = train(args.algo, args.setting, args.seed, args.budget, args.out, args.eval_every,
                args.curve_episodes, overrides, setting_overrides)
print("saved", run_dir)
