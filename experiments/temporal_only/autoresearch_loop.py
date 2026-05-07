#!/usr/bin/env python3
"""Sequential hyperparameter sweep: runs ``train.py`` multiple times with preset configs.

Usage (from repo root, with venv activated or explicit interpreter):

    ./experiments/temporal_only/autoresearch_loop.py --epochs 3

    python experiments/temporal_only/autoresearch_loop.py --epochs 5 --seed 1

This is the project's \"autoresearch loop\": real training jobs chained in order,
each writing ``logs/<run-name>.txt`` via ``train.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRAIN = Path(__file__).resolve().parent / "train.py"


# Default grid matches the Week 3 dry-run variants (names stay stable for logs).
DEFAULT_EXPERIMENTS: list[dict[str, str | int | float]] = [
    {"run-name": "iteration_2", "hidden": "64", "layers": "2", "dropout": "0.0"},
    {"run-name": "iteration_3", "hidden": "128", "layers": "2", "dropout": "0.0"},
    {"run-name": "iteration_4", "hidden": "64", "layers": "1", "dropout": "0.0"},
    {"run-name": "iteration_5", "hidden": "64", "layers": "2", "dropout": "0.2"},
]


def build_argv(
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    exp: dict[str, str | int | float],
) -> list[str]:
    argv = [
        sys.executable,
        str(_TRAIN),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--run-name",
        str(exp["run-name"]),
        "--hidden",
        str(exp["hidden"]),
        "--layers",
        str(exp["layers"]),
        "--dropout",
        str(exp["dropout"]),
    ]
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not _TRAIN.is_file():
        print(f"error: missing train script {_TRAIN}", file=sys.stderr)
        sys.exit(1)

    experiments = DEFAULT_EXPERIMENTS
    print(f"Autoresearch loop: {len(experiments)} runs, cwd={_PROJECT_ROOT}")
    for i, exp in enumerate(experiments, start=1):
        name = exp["run-name"]
        print(f"\n--- [{i}/{len(experiments)}] {name} ---")
        argv = build_argv(
            args.epochs,
            args.batch_size,
            args.lr,
            args.seed,
            exp,
        )
        subprocess.run(argv, cwd=_PROJECT_ROOT, check=True)

    print("\nDone. Logs under experiments/temporal_only/logs/")


if __name__ == "__main__":
    main()
