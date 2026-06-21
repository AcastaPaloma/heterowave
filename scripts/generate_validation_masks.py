"""Generate deterministic Phase 5 validation-sector masks."""

from __future__ import annotations

import argparse

from heterowave.data.masks import save_validation_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-sectors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    print(save_validation_masks(args.output, num_sectors=args.num_sectors, seed=args.seed))


if __name__ == "__main__":
    main()
