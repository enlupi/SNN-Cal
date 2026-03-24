"""gen_eval_list.py

Scan the checkpoints directory and write eval_checkpoints.txt — one
absolute path per line — containing only *completed* checkpoints.

A checkpoint is considered complete when its filename does NOT contain
an _epochN suffix before the .pt extension (those are intermediate saves).

Examples
--------
  Included  : energy_spike.pt, centroid_x_membrane.pt, energy_ann.pt,
               energy_spike_test.pt
  Excluded  : energy_spike_epoch2.pt, centroid_x_ann_epoch8.pt

Usage
-----
    python gen_eval_list.py
    python gen_eval_list.py --ckpt_dir ./checkpoints --out eval_checkpoints.txt
    python gen_eval_list.py --categories spike membrane
    python gen_eval_list.py --categories ann
    python gen_eval_list.py --categories all
"""

import argparse
import os
import re

VALID_CATEGORIES = ("ann", "spike", "membrane", "refl")


def is_complete(filename: str) -> bool:
    """Return True if the checkpoint is a final (non-intermediate) save."""
    return not re.search(r'_epoch\d+\.pt$', filename)


def matches_category(filename: str, categories: list[str]) -> bool:
    """Return True if the filename contains any of the requested categories."""
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    return any(cat in parts for cat in categories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="./checkpoints",
                        help="Directory to scan (default: ./checkpoints)")
    parser.add_argument("--out", default="eval_checkpoints.txt",
                        help="Output list file (default: eval_checkpoints.txt)")
    parser.add_argument(
        "--categories", nargs="+", default=["all"],
        metavar="CAT",
        help=(
            "Filter by category. Choose one or more of: "
            f"{', '.join(VALID_CATEGORIES)}, all. "
            "Default: all."
        ),
    )
    args = parser.parse_args()

    # Resolve categories
    requested = [c.lower() for c in args.categories]
    if "all" in requested:
        categories = list(VALID_CATEGORIES)
    else:
        unknown = [c for c in requested if c not in VALID_CATEGORIES]
        if unknown:
            raise SystemExit(
                f"ERROR: unknown category/categories: {', '.join(unknown)}\n"
                f"Valid choices: {', '.join(VALID_CATEGORIES)}, all"
            )
        categories = requested

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"ERROR: checkpoint directory not found: {ckpt_dir}")

    checkpoints = sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".pt") and is_complete(f) and matches_category(f, categories)
    )

    if not checkpoints:
        raise SystemExit(
            f"No completed checkpoints found in {ckpt_dir} "
            f"for categories: {', '.join(categories)}"
        )

    with open(args.out, "w") as fh:
        fh.write("\n".join(checkpoints) + "\n")

    print(f"Categories: {', '.join(categories)}")
    print(f"Found {len(checkpoints)} completed checkpoint(s):")
    for p in checkpoints:
        print(f"  {p}")
    print(f"\nList written to: {args.out}")
    print(f"\nNow submit with:  condor_submit eval.sub")


if __name__ == "__main__":
    main()
