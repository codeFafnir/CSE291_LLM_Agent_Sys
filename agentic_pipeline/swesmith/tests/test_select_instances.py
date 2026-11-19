import json
from pathlib import Path

import pytest

from select_instances import DatasetRow, image_key_of, select, write_jsonl

KEYS = ["openh264", "srt", "thrift"]
IMAGES = {
    "openh264": "swebench/swesmith.x86_64.cisco_1776_openh264.cf568c83",
    "srt": "swebench/swesmith.x86_64.haivision_1776_srt.ce54b5ea",
    "thrift": "swebench/swesmith.x86_64.apache_1776_thrift.32776c0f",
}
OTHER_IMAGE = "swebench/swesmith.x86_64.nlohmann_1776_json.abcdef12"
PATCH = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n"


def count_by_words(text: str) -> int:
    return len(text.split())


def make_row(index: int, image_name: str, words: int) -> DatasetRow:
    return DatasetRow(
        instance_id=f"inst.{index:03d}",
        repo="owner/repo",
        patch=PATCH,
        FAIL_TO_PASS=[f"test_f2p_{index}"],
        PASS_TO_PASS=[f"test_p2p_{index}", "test_common"],
        image_name=image_name,
        problem_statement=" ".join(["w"] * words),
    )


def synthetic_rows() -> list[DatasetRow]:
    names = [IMAGES["srt"], IMAGES["thrift"], OTHER_IMAGE, IMAGES["openh264"]]
    return [make_row(i, names[i % len(names)], 5 + i % 7) for i in range(60)]


@pytest.mark.parametrize("key", KEYS)
def test_image_key_of_matches_real_images(key: str) -> None:
    assert image_key_of(IMAGES[key], KEYS) == key


def test_image_key_of_rejects_other_image() -> None:
    assert image_key_of(OTHER_IMAGE, KEYS) is None
    assert image_key_of("swebench/swesmith.x86_64.foo_1776_srtx.abc", KEYS) is None
    assert image_key_of("swebench/swesmith.x86_64.srt.abc", KEYS) is None


def test_select_only_keeps_matching_images() -> None:
    selected = select(synthetic_rows(), KEYS, 1000, 1000, 0, count_by_words)
    for key, pairs in selected.items():
        assert pairs
        assert all(image_key_of(row.image_name, KEYS) == key for _, row in pairs)
        assert all(instance.image_key == key for instance, _ in pairs)


def test_select_deterministic_for_seed() -> None:
    rows = synthetic_rows()
    first = select(rows, KEYS, 1000, 5, 7, count_by_words)
    again = select(list(reversed(rows)), KEYS, 1000, 5, 7, count_by_words)
    other = select(rows, KEYS, 1000, 5, 8, count_by_words)
    ids = lambda sel: {k: [i.instance_id for i, _ in v] for k, v in sel.items()}  # noqa: E731
    assert ids(first) == ids(again)
    assert ids(first) != ids(other)
    assert all(len(v) == 5 for v in first.values())


def test_select_budget_filter() -> None:
    selected = select(synthetic_rows(), KEYS, 8, 1000, 0, count_by_words)
    all_pairs = [pair for pairs in selected.values() for pair in pairs]
    assert all_pairs
    assert all(instance.prompt_tokens <= 8 for instance, _ in all_pairs)
    unfiltered = select(synthetic_rows(), KEYS, 1000, 1000, 0, count_by_words)
    assert sum(len(v) for v in unfiltered.values()) > len(all_pairs)


def test_full_file_matches_slim_file(tmp_path: Path) -> None:
    selected = select(synthetic_rows(), KEYS, 1000, 6, 3, count_by_words)
    pairs = selected["srt"]
    slim, full = tmp_path / "instances_srt.jsonl", tmp_path / "instances_full_srt.jsonl"
    write_jsonl(slim, [instance for instance, _ in pairs])
    write_jsonl(full, [row for _, row in pairs])
    slim_rows = [json.loads(line) for line in slim.read_text().splitlines()]
    full_rows = [json.loads(line) for line in full.read_text().splitlines()]
    assert [r["instance_id"] for r in slim_rows] == [r["instance_id"] for r in full_rows]
    assert set(slim_rows[0]) == {
        "instance_id", "image_key", "image_name", "repo",
        "problem_statement", "prompt_tokens", "gold_patch_lines",
    }
    assert set(full_rows[0]) == {
        "instance_id", "repo", "patch", "FAIL_TO_PASS", "PASS_TO_PASS",
        "image_name", "problem_statement",
    }
    for row in full_rows:
        index = int(row["instance_id"].split(".")[1])
        assert row["patch"] == PATCH
        assert row["FAIL_TO_PASS"] == [f"test_f2p_{index}"]
        assert row["PASS_TO_PASS"] == [f"test_p2p_{index}", "test_common"]
    assert slim_rows[0]["gold_patch_lines"] == 2
