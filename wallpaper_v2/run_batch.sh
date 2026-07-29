#!/bin/bash
# One OS process per sample: peak memory stays per-sample instead of cumulative.
cd "$(dirname "$0")"
for d in ../rgb-x_output/batch_output/*/; do
  n=$(basename "$d")
  [ -d "$d" ] || continue
  python3 -u run_all.py --one "$n"
done
python3 -u run_all.py --comparisons
python3 -u run_all.py --collect
