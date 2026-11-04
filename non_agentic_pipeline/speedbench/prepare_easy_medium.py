"""Resolve the SPEED-Bench easy and medium rows without downloading cais/hle.

  prepare_easy_medium.py --out <dir>/test.parquet

Loads the public nvidia/SPEED-Bench qualitative split, keeps difficulty easy and medium
(22 and 29), and fills the masked coding prompts from LiveCodeBench and Code Contests.
Hard rows, including every cais/hle row, are dropped before any external download.
The turn text uses the same placeholder removal and format keys as NVIDIA's specdec_bench.
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from pandas import DataFrame

from models import SpeedRow, Tier

PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
HLE_SOURCE = "cais/hle"
LIVECODEBENCH_SOURCE = "livecodebench/code_generation_lite"
CODE_CONTESTS_SOURCE = "deepmind/code_contests"
LIVECODEBENCH_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
CODE_CONTESTS_REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"
LIVECODEBENCH_FILES: tuple[str, ...] = ("test", "test2", "test3", "test4", "test5", "test6")
EXPECTED_COUNTS: dict[Tier, int] = {"easy": 22, "medium": 29}


def load_qualitative() -> list[SpeedRow]:
    dataset = load_dataset("nvidia/SPEED-Bench", "qualitative", split="test")
    return [SpeedRow.model_validate(dict(row)) for row in dataset]


def keep_easy_medium(rows: list[SpeedRow]) -> list[SpeedRow]:
    hle = [row.question_id for row in rows if HLE_SOURCE in row.source and row.difficulty in EXPECTED_COUNTS]
    if hle:
        raise ValueError(f"easy/medium rows point at {HLE_SOURCE}: {hle}")
    kept = [row for row in rows if row.difficulty in EXPECTED_COUNTS]
    counts = {tier: sum(1 for row in kept if row.difficulty == tier) for tier in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"tier counts {counts} differ from expected {EXPECTED_COUNTS}")
    return kept


def load_livecodebench() -> DataFrame:
    files = [
        f"https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{LIVECODEBENCH_REVISION}/{name}.jsonl"
        for name in LIVECODEBENCH_FILES
    ]
    return load_dataset("json", data_files={"test": files}, split="test").to_pandas()


def load_code_contests() -> DataFrame:
    return load_dataset(CODE_CONTESTS_SOURCE, split="test", revision=CODE_CONTESTS_REVISION).to_pandas()


def one_row(frame: DataFrame, column: str, src_id: str) -> pd.Series:
    hit = frame[frame[column] == src_id]
    if hit.empty:
        raise ValueError(f"no {column}={src_id} in the source dataset")
    return hit.iloc[0]


def fill_placeholder(turn: str, values: dict[str, str]) -> str:
    return turn.removeprefix(f"{PLACEHOLDER}\n\n").format(**values)


def resolve_row(row: SpeedRow, livecodebench: DataFrame, code_contests: DataFrame) -> SpeedRow:
    turn = row.turns[0]
    if not turn.startswith(PLACEHOLDER):
        return row
    if LIVECODEBENCH_SOURCE in row.source:
        source_row = one_row(livecodebench, "question_id", row.src_id)
        filled = fill_placeholder(
            turn,
            {"question": str(source_row["question_content"]), "starter_code": str(source_row["starter_code"])},
        )
    elif CODE_CONTESTS_SOURCE in row.source:
        source_row = one_row(code_contests, "name", row.src_id)
        filled = fill_placeholder(turn, {"question": str(source_row["description"])})
    else:
        raise ValueError(f"masked row {row.question_id} has a source this script does not fetch: {row.source}")
    return row.model_copy(update={"turns": [filled, *row.turns[1:]]})


def write_parquet(rows: list[SpeedRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row.model_dump() for row in rows])
    frame.to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kept = keep_easy_medium(load_qualitative())
    masked = [row for row in kept if row.turns[0].startswith(PLACEHOLDER)]
    livecodebench = load_livecodebench() if any(LIVECODEBENCH_SOURCE in row.source for row in masked) else pd.DataFrame()
    code_contests = load_code_contests() if any(CODE_CONTESTS_SOURCE in row.source for row in masked) else pd.DataFrame()
    resolved = [resolve_row(row, livecodebench, code_contests) for row in kept]
    still_masked = [row.question_id for row in resolved if row.turns[0].startswith(PLACEHOLDER)]
    if still_masked:
        raise ValueError(f"placeholders left after resolve: {still_masked}")
    write_parquet(resolved, args.out)
    print(f"wrote {len(resolved)} rows ({len(masked)} resolved from source) to {args.out}")


if __name__ == "__main__":
    main()
