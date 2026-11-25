import json
from pathlib import Path

import pytest

from prep_images import (
    DatasetRow,
    main,
    read_rows,
    select_django,
    to_agent_record,
)


def make_row(index: int, repo: str) -> DatasetRow:
    return DatasetRow(
        instance_id=f"{repo.split('/')[0]}__{repo.split('/')[1]}-{index:04d}",
        image=f"swebench/sweb.eval.x86_64.{repo.replace('/', '_1776_')}-{index}:latest",
        repo=repo,
        problem_statement=f"problem {index}",
        base_commit=f"commit{index}",
    )


def synthetic_rows() -> list[DatasetRow]:
    django = [make_row(i, "django/django") for i in range(20)]
    other = [make_row(i, "sympy/sympy") for i in range(20)]
    return other + django


def test_select_django_only_django_and_count() -> None:
    picked = select_django(synthetic_rows(), 5, 291)
    assert len(picked) == 5
    assert {r.repo for r in picked} == {"django/django"}
    assert len({r.instance_id for r in picked}) == 5


def test_select_django_stable_for_seed_and_input_order() -> None:
    rows = synthetic_rows()
    first = select_django(rows, 7, 291)
    assert first == select_django(rows, 7, 291)
    assert first == select_django(list(reversed(rows)), 7, 291)


def test_select_django_differs_across_seeds() -> None:
    rows = synthetic_rows()
    assert select_django(rows, 7, 1) != select_django(rows, 7, 2)


def test_select_django_too_few_raises() -> None:
    with pytest.raises(ValueError):
        select_django(synthetic_rows(), 21, 291)


def test_select_django_exact_count_returns_all() -> None:
    picked = select_django(synthetic_rows(), 20, 291)
    assert {r.instance_id for r in picked} == {r.instance_id for r in synthetic_rows() if r.repo == "django/django"}


def test_to_agent_record_keys_and_values() -> None:
    row = make_row(3, "django/django")
    assert to_agent_record(row) == {
        "image_name": row.image,
        "problem_statement": "problem 3",
        "instance_id": row.instance_id,
        "repo_name": "testbed",
        "base_commit": "commit3",
    }


def test_cli_writes_instances_and_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "verified.jsonl"
    rows = synthetic_rows()
    dataset.write_text("".join(json.dumps({**r.model_dump(), "extra": "ignored"}) + "\n" for r in rows))
    out_instances = tmp_path / "out" / "instances_verified.jsonl"
    out_ready = tmp_path / "out" / "images_ready.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prep_images.py",
            "--dataset-jsonl", str(dataset),
            "--count", "4",
            "--seed", "291",
            "--out-instances", str(out_instances),
            "--out-ready", str(out_ready),
        ],
    )
    main()
    expected = select_django(read_rows(dataset), 4, 291)
    instances = [json.loads(line) for line in out_instances.read_text().splitlines()]
    ready = [json.loads(line) for line in out_ready.read_text().splitlines()]
    assert instances == [to_agent_record(r) for r in expected]
    assert ready == [{"instance_id": r.instance_id, "image": r.image} for r in expected]
