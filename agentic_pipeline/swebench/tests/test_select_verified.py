import json
from pathlib import Path

import pytest

from select_verified import (
    EASY_DIFFICULTY,
    FILL_DIFFICULTY,
    VerifiedRow,
    image_name_of,
    is_single_source_file,
    patch_files,
    patch_lines,
    select,
    write_jsonl,
)


def make_patch(paths: list[str], lines: int) -> str:
    body = "".join(f"+x{i}\n" for i in range(lines))
    return "".join(
        f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -0,0 +1,{lines} @@\n{body}" for p in paths
    )


def make_row(name: str, difficulty: str, paths: list[str], lines: int) -> VerifiedRow:
    return VerifiedRow(
        instance_id=name,
        repo="o/r",
        base_commit="c" * 40,
        patch=make_patch(paths, lines),
        problem_statement=f"issue {name}",
        FAIL_TO_PASS=[f"t_{name}"],
        PASS_TO_PASS=["t_common"],
        difficulty=difficulty,
    )


def synthetic_rows() -> list[VerifiedRow]:
    easy = [make_row(f"e__r-{i:03d}", EASY_DIFFICULTY, ["src/a.py"], 1 + i % 3) for i in range(30)]
    fill = [make_row(f"m__r-{i:03d}", FILL_DIFFICULTY, ["src/b.py"], 10 - i) for i in range(8)]
    skipped = [
        make_row("m__r-multi", FILL_DIFFICULTY, ["src/a.py", "src/b.py"], 1),
        make_row("m__r-test", FILL_DIFFICULTY, ["pkg/tests/test_a.py"], 1),
        make_row("m__r-hard", "1-4 hours", ["src/a.py"], 1),
    ]
    return [*easy, *fill, *skipped]


def test_patch_files_and_lines() -> None:
    patch = make_patch(["a/b.py", "c.py"], 2)
    assert patch_files(patch) == ["a/b.py", "c.py"]
    assert patch_lines(patch) == 4


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["src/a.py"], True),
        (["src/a.py", "src/b.py"], False),
        (["pkg/tests/test_a.py"], False),
        (["tests/a.py"], False),
        (["pkg/test_a.py"], False),
        (["pkg/testing/a.py"], False),
        ([], False),
    ],
)
def test_is_single_source_file(paths: list[str], expected: bool) -> None:
    assert is_single_source_file(make_patch(paths, 1)) is expected


def test_select_takes_all_easy_and_smallest_single_file_fill() -> None:
    selection = select(synthetic_rows(), 6, 5, 291)
    ids = [instance.instance_id for instance, _ in selection.picked]
    assert len(ids) == 36
    assert ids[:30] == sorted(ids[:30]) and all(i.startswith("e__") for i in ids[:30])
    assert ids[30:] == [f"m__r-{i:03d}" for i in (7, 6, 5, 4, 3, 2)]
    assert "m__r-multi" not in ids and "m__r-test" not in ids and "m__r-hard" not in ids


def test_smoke_is_seeded_easy_only() -> None:
    rows = synthetic_rows()
    first = select(rows, 6, 5, 291)
    again = select(list(reversed(rows)), 6, 5, 291)
    other = select(rows, 6, 5, 292)
    ids = lambda s: [i.instance_id for i in s.smoke]  # noqa: E731
    assert ids(first) == ids(again) and ids(first) != ids(other)
    assert len(first.smoke) == 5 and all(i.startswith("e__") for i in ids(first))


def test_select_raises_when_too_few_fill_rows() -> None:
    with pytest.raises(ValueError, match="single-file"):
        select(synthetic_rows(), 9, 5, 0)


def test_select_raises_when_too_few_easy_rows() -> None:
    with pytest.raises(ValueError, match="smoke"):
        select(synthetic_rows(), 6, 31, 0)


def test_instance_fields_and_image_name() -> None:
    instance, full = select(synthetic_rows(), 6, 5, 0).picked[0]
    assert instance.image_name == image_name_of("e__r-000") == (
        "ghcr.io/epoch-research/swe-bench.eval.x86_64.e__r-000:latest"
    )
    assert instance.image_key == "verified" and instance.base_commit == "c" * 40
    assert full.gold_patch_lines == 1 and full.difficulty == EASY_DIFFICULTY


def test_f2p_decodes_json_string() -> None:
    row = make_row("e__r-x", EASY_DIFFICULTY, ["a.py"], 1).model_dump()
    row["FAIL_TO_PASS"] = json.dumps(["t1", "t2"])
    assert VerifiedRow.model_validate(row).FAIL_TO_PASS == ["t1", "t2"]


def test_written_files_have_expected_columns(tmp_path: Path) -> None:
    selection = select(synthetic_rows(), 6, 5, 0)
    slim, full = tmp_path / "slim.jsonl", tmp_path / "full.jsonl"
    write_jsonl(slim, [i for i, _ in selection.picked])
    write_jsonl(full, [f for _, f in selection.picked])
    slim_rows = [json.loads(line) for line in slim.read_text().splitlines()]
    full_rows = [json.loads(line) for line in full.read_text().splitlines()]
    assert set(slim_rows[0]) == {"instance_id", "image_key", "image_name", "problem_statement", "base_commit"}
    assert set(full_rows[0]) == {
        "instance_id", "repo", "base_commit", "patch", "FAIL_TO_PASS", "PASS_TO_PASS",
        "difficulty", "gold_patch_lines",
    }
    assert [r["instance_id"] for r in slim_rows] == [r["instance_id"] for r in full_rows]
