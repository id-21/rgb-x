"""
Capture the current pipeline's behaviour as outputs/baseline/<sample>.json.

Every later Phase-2 task reports a delta against this. Run once at the start
of Phase 2; never overwrite an existing baseline file without --force.

Usage:
    python3 harness/capture_baseline.py [--force]
"""

import json
import sys

from _common import BASELINE_DIR, all_samples, render_one


def main():
    force = "--force" in sys.argv[1:]
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    samples = all_samples()
    written, skipped = [], []
    for name in samples:
        out_path = BASELINE_DIR / f"{name}.json"
        if out_path.exists() and not force:
            print(f"  = {name}: baseline already exists, skipping (use --force to redo)")
            continue

        print(f"--- {name} ---")
        baseline, err = render_one(name)
        if err:
            print(f"  !! SKIP {err}")
            skipped.append((name, err))
            continue

        out_path.write_text(json.dumps(baseline, indent=2))
        written.append(name)
        print(f"  wrote {out_path.relative_to(BASELINE_DIR.parent.parent)}  "
              f"wall_coverage={baseline['wall_coverage']:.4f}")

    print(f"\n{'=' * 70}")
    print(f"captured {len(written)}/{len(samples)} baselines, {len(skipped)} skipped")
    for name, err in skipped:
        print(f"  SKIPPED: {err}")
    if written:
        cov = [json.loads((BASELINE_DIR / f"{n}.json").read_text())["wall_coverage"]
               for n in written]
        print(f"wall_coverage: min={min(cov):.4f} max={max(cov):.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
