"""Select the SPEED-Bench easy and medium prompts that are runnable as-is.

  prepare_easy_medium.py --out <dir>/test.parquet

Loads the public nvidia/SPEED-Bench qualitative split and keeps difficulty easy and medium.
Rows whose first turn is a masked placeholder (coding prompts that need an external source
dataset) are dropped, so nothing outside nvidia/SPEED-Bench is downloaded.
Multi-turn rows keep all their turns.
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from models import SpeedRow, Tier

PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
TIERS: tuple[Tier, ...] = ("easy", "medium")


def load_qualitative() -> list[SpeedRow]:
    dataset = load_dataset("nvidia/SPEED-Bench", "qualitative", split="test")
    return [SpeedRow.model_validate(dict(row)) for row in dataset]


def keep_runnable(rows: list[SpeedRow]) -> list[SpeedRow]:
    return [
        row
        for row in rows
        if row.difficulty in TIERS and not any(turn.startswith(PLACEHOLDER) for turn in row.turns)
    ]


def write_parquet(rows: list[SpeedRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row.model_dump() for row in rows]).to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kept = keep_runnable(load_qualitative())
    write_parquet(kept, args.out)
    multi = sum(1 for row in kept if len(row.turns) > 1)
    print(f"wrote {len(kept)} prompts ({multi} multi-turn) to {args.out}")


if __name__ == "__main__":
    main()
