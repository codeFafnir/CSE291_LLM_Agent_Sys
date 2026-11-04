"""Build the SPEED-Bench easy+medium subset from the prepared qualitative parquet.

  build_subset.py --prepared-parquet <test.parquet> --tokenizer Qwen/Qwen3-14B-AWQ \
    --max-model-len 20000 --max-tokens 2048 --seed 291 --out <dir>/speed_subset.jsonl

Fails if any selected prompt is still a masked placeholder or the tier counts differ from 22 easy + 29 medium.
Prompts whose worst-case context exceeds --max-model-len are kept and flagged over_context.
Also writes speed_subset_ids.txt next to --out.
"""

import argparse
import random
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from models import SpeedRow, SubsetRow, Tier

PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED"
EXPECTED_COUNTS: dict[Tier, int] = {"easy": 22, "medium": 29}
IDS_FILENAME = "speed_subset_ids.txt"

TokenCounter = Callable[[list[str]], int]


def load_rows(path: Path) -> list[SpeedRow]:
    return [SpeedRow.model_validate(record) for record in pq.read_table(path).to_pylist()]


def as_tier(difficulty: str | None) -> Tier | None:
    if difficulty == "easy":
        return "easy"
    if difficulty == "medium":
        return "medium"
    return None


def select_tiered(rows: list[SpeedRow]) -> list[tuple[Tier, SpeedRow]]:
    selected = [(tier, row) for row in rows if (tier := as_tier(row.difficulty)) is not None]
    masked = [row.question_id for _, row in selected if any(turn.startswith(PLACEHOLDER) for turn in row.turns)]
    if masked:
        raise ValueError(f"{len(masked)} selected rows are still masked placeholders: {masked}")
    counts = {tier: sum(1 for t, _ in selected if t == tier) for tier in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"tier counts {counts} differ from expected {EXPECTED_COUNTS}")
    return selected


def make_token_counter(tokenizer: PreTrainedTokenizerBase) -> TokenCounter:
    def count(turns: list[str]) -> int:
        messages = [{"role": "user", "content": turn} for turn in turns]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


def to_subset_row(tier: Tier, row: SpeedRow, count: TokenCounter, max_model_len: int, max_tokens: int) -> SubsetRow:
    n_turns = len(row.turns)
    # Every turn may emit max_tokens and each reply is fed back as history, so this bounds the last turn's context.
    worst_case = count(row.turns) + max_tokens * n_turns
    return SubsetRow(
        question_id=row.question_id,
        category=row.category,
        sub_category=row.sub_category,
        tier=tier,
        turns=row.turns,
        n_turns=n_turns,
        first_turn_tokens=count(row.turns[:1]),
        worst_case_context=worst_case,
        over_context=worst_case > max_model_len,
        source=row.source,
        src_id=row.src_id,
    )


def shuffled(items: list[SubsetRow], seed: int) -> list[SubsetRow]:
    ordered = list(items)
    random.Random(seed).shuffle(ordered)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = make_token_counter(AutoTokenizer.from_pretrained(args.tokenizer))
    selected = select_tiered(load_rows(args.prepared_parquet))
    subset = shuffled(
        [to_subset_row(tier, row, count, args.max_model_len, args.max_tokens) for tier, row in selected],
        args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(row.model_dump_json() + "\n" for row in subset))
    (args.out.parent / IDS_FILENAME).write_text("".join(row.question_id + "\n" for row in subset))
    over = [row.question_id for row in subset if row.over_context]
    turns = sum(row.n_turns for row in subset)
    print(f"wrote {len(subset)} prompts ({turns} requests) to {args.out}; over_context={len(over)} {over}")


if __name__ == "__main__":
    main()
