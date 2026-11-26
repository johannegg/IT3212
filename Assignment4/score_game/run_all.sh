#!/bin/bash
# Reproduces the main experiment: tuning, winner pick, Settings A-D, A-D summary, Part 3.
# Every step skips work that is already saved, so the script can be stopped and started again.
# Run from a clean tuning/ folder (no state.json) to redo the tuning from round 1.
set -e
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
W=${WORKERS:-15}

python tune.py --workers "$W"                # rounds until the stop rule ends the tuning (it ended after round 6)
python tune.py --final                    # tuning/winners.json: the best average of each model in the last round
python run_ad.py --workers "$W"           # records/ad/: 5 seeds x 2 models x Settings A, B, C; the B agents also in D
python summarize.py                       # records/ad/results/: tables and learning curves
python run_part3.py --workers "$W"        # records/part3/: one factor at a time from the winners, 5 seeds each
