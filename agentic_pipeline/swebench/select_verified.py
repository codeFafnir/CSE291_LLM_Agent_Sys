"""Build the SWE-bench Verified slice: every "<15 min fix" task plus the smallest single-file "15 min - 1 hour" fixes.

Writes into --out-dir:
  instances_verified.jsonl       slim rows for swesmith/run.py (InstanceRow fields + base_commit)
  instances_full_verified.jsonl  full rows (patch, F2P, P2P, difficulty) for a later evaluation; not for run.py
  instances_smoke<N>.jsonl       N slim rows sampled from the easy band with a fixed seed
(Named select_verified.py so it does not collide with swesmith/select_instances.py on PYTHONPATH.)
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from pydantic import BaseModel, ConfigDict, field_validator

DATASET = "SWE-bench/SWE-bench_Verified"
EASY_DIFFICULTY = "<15 min fix"
FILL_DIFFICULTY = "15 min - 1 hour"
IMAGE_PREFIX = "ghcr.io/epoch-research/swe-bench.eval.x86_64."
IMAGE_KEY = "verified"
DIFF_HEADER = re.compile(r"^diff --git a/\S+ b/(\S+)$", re.MULTILINE)


class VerifiedRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    repo: str
    base_commit: str
    patch: str
    problem_statement: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    difficulty: str

    @field_validator("FAIL_TO_PASS", "PASS_TO_PASS", mode="before")
    @classmethod
    def decode_json_list(cls, value: list[str] | str) -> list[str]:
        """Older dataset revisions store these lists as JSON strings."""
        return json.loads(value) if isinstance(value, str) else value


class Instance(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    image_key: str
    image_name: str
    problem_statement: str
    base_commit: str


class FullRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    repo: str
    base_commit: str
    patch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    difficulty: str
    gold_patch_lines: int


class Selection(BaseModel):
    model_config = ConfigDict(frozen=True)

    picked: list[tuple[Instance, FullRow]]
    smoke: list[Instance]


def image_name_of(instance_id: str) -> str:
    return f"{IMAGE_PREFIX}{instance_id}:latest"


def patch_files(patch: str) -> list[str]:
    return DIFF_HEADER.findall(patch)


def patch_lines(patch: str) -> int:
    return sum(
        1
        for line in patch.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )


def is_test_path(path: str) -> bool:
    return path.startswith("test") or "/test" in path or path.rsplit("/", 1)[-1].startswith("test_")


def is_single_source_file(patch: str) -> bool:
    files = patch_files(patch)
    return len(files) == 1 and not is_test_path(files[0])


def to_instance(row: VerifiedRow) -> Instance:
    return Instance(
        instance_id=row.instance_id,
        image_key=IMAGE_KEY,
        image_name=image_name_of(row.instance_id),
        problem_statement=row.problem_statement,
        base_commit=row.base_commit,
    )


def to_full(row: VerifiedRow) -> FullRow:
    return FullRow(
        instance_id=row.instance_id,
        repo=row.repo,
        base_commit=row.base_commit,
        patch=row.patch,
        FAIL_TO_PASS=row.FAIL_TO_PASS,
        PASS_TO_PASS=row.PASS_TO_PASS,
        difficulty=row.difficulty,
        gold_patch_lines=patch_lines(row.patch),
    )


def select(rows: list[VerifiedRow], fill_count: int, smoke_count: int, seed: int) -> Selection:
    easy = sorted((r for r in rows if r.difficulty == EASY_DIFFICULTY), key=lambda r: r.instance_id)
    fill = sorted(
        (r for r in rows if r.difficulty == FILL_DIFFICULTY and is_single_source_file(r.patch)),
        key=lambda r: (patch_lines(r.patch), r.instance_id),
    )
    if len(fill) < fill_count:
        raise ValueError(
            f"need {fill_count} single-file {FILL_DIFFICULTY!r} rows, found {len(fill)}"
        )
    if len(easy) < smoke_count:
        raise ValueError(f"need {smoke_count} {EASY_DIFFICULTY!r} rows for the smoke set, found {len(easy)}")
    shuffled = list(easy)
    random.Random(seed).shuffle(shuffled)
    return Selection(
        picked=[(to_instance(r), to_full(r)) for r in [*easy, *fill[:fill_count]]],
        smoke=[to_instance(r) for r in shuffled[:smoke_count]],
    )


def write_jsonl(path: Path, items: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i.model_dump_json() + "\n" for i in items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fill-count", type=int, required=True)
    parser.add_argument("--smoke-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [VerifiedRow.model_validate(row) for row in load_dataset(DATASET, split="test")]
    selection = select(rows, args.fill_count, args.smoke_count, args.seed)
    slim = args.out_dir / "instances_verified.jsonl"
    write_jsonl(slim, [instance for instance, _ in selection.picked])
    write_jsonl(args.out_dir / "instances_full_verified.jsonl", [full for _, full in selection.picked])
    smoke = args.out_dir / f"instances_smoke{args.smoke_count}.jsonl"
    write_jsonl(smoke, selection.smoke)
    print(f"picked={len(selection.picked)} smoke={len(selection.smoke)} -> {slim} {smoke}")


if __name__ == "__main__":
    main()
