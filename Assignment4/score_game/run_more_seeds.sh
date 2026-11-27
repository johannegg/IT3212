#!/bin/bash
# Adds training seeds 5-9, then 10-14, to Settings A-D and Part 3 (15 agents per configuration in total).
# Each block finishes A-D and Part 3 before the next starts, so a stop leaves a complete set with the same seeds everywhere.
# Finished runs are skipped, so the script can be stopped and started again. Summaries and figures are redone after each block.
set -e
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
W=${WORKERS:-15}
for block in 5,6,7,8,9 10,11,12,13,14; do
  echo "$(date '+%F %T') seeds $block: A-D"
  SEEDS=$block python run_ad.py --workers "$W"
  python summarize.py
  echo "$(date '+%F %T') seeds $block: Part 3"
  SEEDS=$block python run_part3.py --workers "$W"
  echo "$(date '+%F %T') seeds $block: done"
done
