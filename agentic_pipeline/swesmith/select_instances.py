"""Pick SWE-smith-cpp instances per docker image and write instances_<image>.jsonl.

Also writes instances_full_<image>.jsonl (full dataset rows, same order) as the local dataset
file for swesmith.harness.eval.

Prompt tokens are the problem statement only; the agent templates and the observed trajectory are
bounded at run time by MAX_MODEL_LEN. Instances are shuffled with a fixed seed, then capped.
(Named select_instances.py because select.py would shadow the stdlib module.)
"""

import argparse
import random
import re
from collections.abc import Callable
from pathlib import Path

from datasets import load_dataset
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer

DATASET = "SWE-bench/SWE-smith-cpp"
TOKENIZER = "Qwen/Qwen3-14B-AWQ"
PROMPT_BUDGET_FRACTION = 4


class DatasetRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    repo: str
    patch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    image_name: str
    problem_statement: str


class Instance(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    image_key: str
    image_name: str
    repo: str
    problem_statement: str
    prompt_tokens: int
    gold_patch_lines: int


def image_key_of(image_name: str, keys: list[str]) -> str | None:
    for key in keys:
        if re.search(rf"_1776_{re.escape(key)}\.", image_name):
            return key
    return None


def patch_lines(patch: str) -> int:
    return sum(
        1
        for line in patch.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )


def to_instance(
    row: DatasetRow, key: str, count_tokens: Callable[[str], int]
) -> Instance:
    return Instance(
        instance_id=row.instance_id,
        image_key=key,
        image_name=row.image_name,
        repo=row.repo,
        problem_statement=row.problem_statement,
        prompt_tokens=count_tokens(row.problem_statement),
        gold_patch_lines=patch_lines(row.patch),
    )


def choose(
    instances: list[Instance], max_prompt_tokens: int, count: int, seed: int
) -> list[Instance]:
    fits = sorted(
        (i for i in instances if i.prompt_tokens <= max_prompt_tokens),
        key=lambda i: i.instance_id,
    )
    random.Random(seed).shuffle(fits)
    return fits[:count]


def select(
    rows: list[DatasetRow],
    keys: list[str],
    max_prompt_tokens: int,
    count: int,
    seed: int,
    count_tokens: Callable[[str], int],
) -> dict[str, list[tuple[Instance, DatasetRow]]]:
    matched: dict[str, list[tuple[Instance, DatasetRow]]] = {key: [] for key in keys}
    for row in rows:
        key = image_key_of(row.image_name, keys)
        if key is not None:
            matched[key].append((to_instance(row, key, count_tokens), row))
    result: dict[str, list[tuple[Instance, DatasetRow]]] = {}
    for key, pairs in matched.items():
        by_id = {instance.instance_id: row for instance, row in pairs}
        picked = choose([instance for instance, _ in pairs], max_prompt_tokens, count, seed)
        result[key] = [(instance, by_id[instance.instance_id]) for instance in picked]
    return result


def write_jsonl(path: Path, items: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i.model_dump_json() + "\n" for i in items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True, help="image keys, e.g. openh264 srt thrift")
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    rows = [DatasetRow.model_validate(row) for row in load_dataset(DATASET, split="train")]
    selected = select(
        rows,
        args.images,
        args.max_model_len // PROMPT_BUDGET_FRACTION,
        args.count,
        args.seed,
        lambda text: len(tokenizer(text)["input_ids"]),
    )
    for key, pairs in selected.items():
        out = args.out_dir / f"instances_{key}.jsonl"
        write_jsonl(out, [instance for instance, _ in pairs])
        write_jsonl(args.out_dir / f"instances_full_{key}.jsonl", [row for _, row in pairs])
        print(f"{key}: picked={len(pairs)} image={pairs[0][0].image_name} -> {out}")


if __name__ == "__main__":
    main()
