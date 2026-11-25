"""Select django SWE-bench Verified rows and write the SWE-agent --instances file plus an images-ready list.

Docker pulls and the python probe live in experiments/prep_images.sh, not here.
"""

import argparse
import json
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict

DJANGO_REPO = "django/django"
AGENT_REPO_NAME = "testbed"


class DatasetRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    image: str
    repo: str
    problem_statement: str
    base_commit: str


class ReadyRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    image: str


def select_django(rows: list[DatasetRow], count: int, seed: int) -> list[DatasetRow]:
    django = sorted((r for r in rows if r.repo == DJANGO_REPO), key=lambda r: r.instance_id)
    if len(django) < count:
        raise ValueError(f"need {count} {DJANGO_REPO} rows, found {len(django)}")
    return random.Random(seed).sample(django, count)


def to_agent_record(row: DatasetRow) -> dict[str, str]:
    return {
        "image_name": row.image,
        "problem_statement": row.problem_statement,
        "instance_id": row.instance_id,
        "repo_name": AGENT_REPO_NAME,
        "base_commit": row.base_commit,
    }


def to_ready_record(row: DatasetRow) -> ReadyRecord:
    return ReadyRecord(instance_id=row.instance_id, image=row.image)


def read_rows(path: Path) -> list[DatasetRow]:
    return [DatasetRow.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-instances", type=Path, required=True)
    parser.add_argument("--out-ready", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    picked = select_django(read_rows(args.dataset_jsonl), args.count, args.seed)
    write_jsonl(args.out_instances, [json.dumps(to_agent_record(r)) for r in picked])
    write_jsonl(args.out_ready, [to_ready_record(r).model_dump_json() for r in picked])


if __name__ == "__main__":
    main()
