"""Score SWE-agent predictions with swesmith eval, fixing two defects of the stock eval.

The stock eval applies the prediction to clean main (not the bug commit) and never rebuilds
before running tests, so the score is vacuous. We prepend the bug patch to each prediction and
make every profile's test command build first.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator

IMAGE_KEYS = ["openh264", "srt", "thrift"]
BUILD_DIRS = {"srt": "build", "thrift": "cmake-build"}
OPENH264_BUILD = "make ENABLE64BIT=Yes BUILDTYPE=Release all"
OPENH264_TEST = "./codec_unittest"


class Prediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    model_name_or_path: str
    model_patch: str

    @field_validator("model_patch", mode="before")
    @classmethod
    def none_to_empty(cls, value: str | None) -> str:
        return "" if value is None else value


class FullRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    repo: str
    patch: str
    image_name: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    problem_statement: str


class Profile(Protocol):
    test_cmd: str
    timeout: int


class Registry(Protocol):
    def get(self, key: str) -> Profile: ...


class EvalModule(Protocol):
    def main(
        self,
        run_id: str,
        workers: int,
        predictions_path: str,
        dataset_path: str,
        f2p_only: bool,
    ) -> None: ...


def convert_predictions(
    preds: dict[str, Prediction], rows: dict[str, FullRow]
) -> dict[str, Prediction]:
    converted: dict[str, Prediction] = {}
    for instance_id, pred in preds.items():
        if instance_id not in rows:
            raise KeyError(f"instance_id {instance_id!r} has a prediction but no full row")
        converted[instance_id] = Prediction(
            instance_id=pred.instance_id,
            model_name_or_path=pred.model_name_or_path,
            model_patch=rows[instance_id].patch + pred.model_patch,
        )
    return converted


def build_test_command(image_key: str, original_test_cmd: str) -> str:
    if image_key in BUILD_DIRS:
        prefix = f"cd {BUILD_DIRS[image_key]} && "
        if not original_test_cmd.startswith(prefix):
            raise ValueError(
                f"{image_key}: test command {original_test_cmd!r} does not start with {prefix!r}"
            )
        return f"{prefix}make -j$(nproc) && {original_test_cmd[len(prefix):]}"
    if image_key == "openh264":
        if original_test_cmd != OPENH264_TEST:
            raise ValueError(
                f"openh264: test command {original_test_cmd!r} is not {OPENH264_TEST!r}"
            )
        return f"{OPENH264_BUILD} && {original_test_cmd}"
    raise ValueError(f"unknown image key {image_key!r}, expected one of {IMAGE_KEYS}")


def image_key_of(image_name: str) -> str:
    for key in IMAGE_KEYS:
        if re.search(rf"_1776_{key}\.", image_name):
            return key
    raise ValueError(f"image {image_name!r} matches none of {IMAGE_KEYS}")


def apply_profile_overrides(
    registry: Registry, rows: list[FullRow], timeout_s: int
) -> None:
    for repo, row in {r.repo: r for r in rows}.items():
        profile = registry.get(repo)
        profile.test_cmd = build_test_command(image_key_of(row.image_name), profile.test_cmd)
        profile.timeout = timeout_s


def run_eval(
    ev: EvalModule,
    registry: Registry,
    rows: list[FullRow],
    run_id: str,
    workers: int,
    timeout_s: int,
    predictions_path: Path,
    dataset_path: Path,
) -> None:
    apply_profile_overrides(registry, rows, timeout_s)
    ev.main(
        run_id=run_id,
        workers=workers,
        predictions_path=str(predictions_path),
        dataset_path=str(dataset_path),
        f2p_only=False,
    )


def load_rows(paths: list[Path]) -> list[FullRow]:
    return [
        FullRow.model_validate_json(line)
        for path in paths
        for line in path.read_text().splitlines()
        if line.strip()
    ]


class ScoreArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    preds: Path
    full_rows: list[Path]
    run_id: str
    workers: int
    timeout_s: int
    out_dir: Path


def parse_args() -> ScoreArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--full-rows", type=Path, nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--timeout-s", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return ScoreArgs.model_validate(vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.full_rows)
    preds = {
        k: Prediction.model_validate(v) for k, v in json.loads(args.preds.read_text()).items()
    }
    converted = convert_predictions(preds, {r.instance_id: r for r in rows})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.out_dir / "preds_for_eval.json"
    dataset_path = args.out_dir / "dataset_for_eval.jsonl"
    preds_path.write_text(json.dumps({k: v.model_dump() for k, v in converted.items()}))
    dataset_path.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    # Lazy: swesmith is installed only in the box's swesmith env.
    import swesmith.harness.eval as ev
    from swesmith.profiles import registry

    run_eval(
        ev, registry, rows, args.run_id, args.workers, args.timeout_s, preds_path, dataset_path
    )
    report_path = Path("logs/run_evaluation") / args.run_id / "report.json"
    shutil.copy(report_path, args.out_dir / "report.json")
    report = json.loads(report_path.read_text())
    print(f"resolved {report['resolved']}/{report['total']}")


if __name__ == "__main__":
    main()
