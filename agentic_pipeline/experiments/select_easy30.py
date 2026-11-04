"""Pick the easiest SWE-bench Pro V2 tasks of one repo, ranked by gold-patch size and prompt length.

The dataset has no easy label. A small gold patch and a short first prompt are the proxy.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DATASET = "ScaleAI/SWE-bench_Pro"
TOKENIZER = "Qwen/Qwen3-14B-AWQ"
REPO = "future-architect/vuls"
PROMPT_BUDGET_FRACTION = 4


@dataclass(frozen=True)
class Candidate:
    instance_id: str
    files_touched: int
    lines_changed: int
    prompt_tokens: int


def patch_size(patch: str) -> tuple[int, int]:
    lines = patch.splitlines()
    files = sum(1 for line in lines if line.startswith("diff --git "))
    changed = sum(
        1
        for line in lines
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )
    return files, changed


def candidate(row: dict[str, object], tokenizer: PreTrainedTokenizerBase) -> Candidate:
    files, changed = patch_size(str(row["patch"]))
    prompt = "\n\n".join(str(row[key]) for key in ("problem_statement", "requirements", "interface"))
    return Candidate(
        instance_id=str(row["instance_id"]),
        files_touched=files,
        lines_changed=changed,
        prompt_tokens=len(tokenizer(prompt)["input_ids"]),
    )


def rank(candidates: list[Candidate], max_prompt_tokens: int, count: int) -> list[Candidate]:
    fits = [c for c in candidates if c.prompt_tokens <= max_prompt_tokens]
    ordered = sorted(fits, key=lambda c: (c.files_touched, c.lines_changed, c.prompt_tokens))
    return ordered[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True, help="SWE-bench_Pro-os/v2/tasks")
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--copy-to", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dataset = load_dataset(DATASET, split="test")
    available = {path.name for path in args.tasks_dir.iterdir() if path.is_dir()}
    rows = [
        row
        for row in dataset
        if row["repo"] == REPO and not row["hard"] and row["instance_id"] in available
    ]
    candidates = [candidate(row, tokenizer) for row in rows]
    picked = rank(candidates, args.max_model_len // PROMPT_BUDGET_FRACTION, args.count)
    ids = "\n".join(c.instance_id for c in picked) + "\n"
    for path in (args.out, args.copy_to):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ids)
    for c in picked:
        print(f"{c.instance_id} files={c.files_touched} lines={c.lines_changed} prompt_tokens={c.prompt_tokens}")
    print(f"candidates={len(candidates)} picked={len(picked)} wrote {args.out} and {args.copy_to}")


if __name__ == "__main__":
    main()
