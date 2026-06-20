"""Build a restart-safe HeteroWave cache from explicit MATLAB paths."""

from __future__ import annotations

import argparse

from heterowave.data.cache import prepare_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-mat", required=True)
    parser.add_argument("--test-mat", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-key")
    parser.add_argument("--test-key")
    parser.add_argument("--train-sample-axis", type=int)
    parser.add_argument("--test-sample-axis", type=int)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-angles", type=int, default=64)
    parser.add_argument("--detector-bins", type=int)
    parser.add_argument("--num-sectors", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--water-speed", type=float, default=1500.0)
    parser.add_argument("--align-corners", action="store_true")
    args = parser.parse_args()
    prepare_cache(**vars(args))

if __name__ == "__main__":
    main()
