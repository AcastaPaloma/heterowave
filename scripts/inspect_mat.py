"""Print all numeric array candidates in one or more MATLAB files."""

from __future__ import annotations

import argparse

from heterowave.data import inspect_mat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="MATLAB files to inspect")
    args = parser.parse_args()
    for path in args.paths:
        inspect_mat(path)


if __name__ == "__main__":
    main()
