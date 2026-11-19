from pathlib import Path

import pytest

from score import (
    FullRow,
    Prediction,
    apply_profile_overrides,
    build_test_command,
    convert_predictions,
    image_key_of,
    run_eval,
)

BUG = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+bug\n"
FIX = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-bug\n+old\n"
CTEST = "ctest --verbose --output-on-failure --rerun-failed --repeat until-pass:1"
ORIGINAL = {
    "srt": f"cd build && {CTEST}",
    "thrift": f"cd cmake-build && {CTEST}",
    "openh264": "./codec_unittest",
}
IMAGES = {
    "openh264": "swebench/swesmith.x86_64.cisco_1776_openh264.cf568c83",
    "srt": "swebench/swesmith.x86_64.haivision_1776_srt.ce54b5ea",
    "thrift": "swebench/swesmith.x86_64.apache_1776_thrift.32776c0f",
}
REPOS = {
    "openh264": "swesmith/cisco__openh264.cf568c83",
    "srt": "swesmith/Haivision__srt.ce54b5ea",
    "thrift": "swesmith/apache__thrift.32776c0f",
}


def row(instance_id: str, key: str) -> FullRow:
    return FullRow(
        instance_id=instance_id,
        repo=REPOS[key],
        patch=BUG,
        image_name=IMAGES[key],
        FAIL_TO_PASS=["a"],
        PASS_TO_PASS=["b"],
        problem_statement="s",
    )


def pred(instance_id: str, patch: str | None) -> Prediction:
    return Prediction.model_validate(
        {"instance_id": instance_id, "model_name_or_path": "m", "model_patch": patch}
    )


def test_convert_concatenates_and_preserves_fields() -> None:
    out = convert_predictions({"i1": pred("i1", FIX)}, {"i1": row("i1", "srt")})
    assert out["i1"].model_patch == BUG + FIX
    assert out["i1"].model_name_or_path == "m"
    assert out["i1"].instance_id == "i1"


@pytest.mark.parametrize("patch", [None, ""])
def test_convert_empty_patch_is_bug_patch_only(patch: str | None) -> None:
    out = convert_predictions({"i1": pred("i1", patch)}, {"i1": row("i1", "srt")})
    assert out["i1"].model_patch == BUG


def test_convert_unknown_id_raises_with_context() -> None:
    with pytest.raises(KeyError, match="nope"):
        convert_predictions({"nope": pred("nope", FIX)}, {"i1": row("i1", "srt")})


def test_build_test_command_srt() -> None:
    assert build_test_command("srt", ORIGINAL["srt"]) == f"cd build && make -j$(nproc) && {CTEST}"


def test_build_test_command_thrift() -> None:
    assert (
        build_test_command("thrift", ORIGINAL["thrift"])
        == f"cd cmake-build && make -j$(nproc) && {CTEST}"
    )


def test_build_test_command_openh264() -> None:
    assert (
        build_test_command("openh264", ORIGINAL["openh264"])
        == "make ENABLE64BIT=Yes BUILDTYPE=Release all && ./codec_unittest"
    )


@pytest.mark.parametrize(
    ("key", "cmd"),
    [("srt", CTEST), ("thrift", f"cd build && {CTEST}"), ("openh264", "./other")],
)
def test_build_test_command_rejects_unexpected_prefix(key: str, cmd: str) -> None:
    with pytest.raises(ValueError, match=key):
        build_test_command(key, cmd)


def test_build_test_command_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_test_command("zlib", "make test")


def test_image_key_of() -> None:
    assert {image_key_of(name) for name in IMAGES.values()} == set(IMAGES)
    with pytest.raises(ValueError):
        image_key_of("swebench/swesmith.x86_64.nlohmann_1776_json.abcdef12")


class FakeProfile:
    def __init__(self, test_cmd: str) -> None:
        self.test_cmd = test_cmd
        self.timeout = 90


class FakeRegistry:
    def __init__(self) -> None:
        self.profiles = {
            REPOS[key]: FakeProfile(ORIGINAL[key]) for key in IMAGES
        }
        self.requested: list[str] = []

    def get(self, key: str) -> FakeProfile:
        self.requested.append(key)
        return self.profiles[key]


class FakeEval:
    def __init__(self, registry: FakeRegistry) -> None:
        self.registry = registry
        self.calls: list[dict[str, str | int | bool]] = []
        self.cmds_at_call: dict[str, str] = {}

    def main(
        self,
        run_id: str,
        workers: int,
        predictions_path: str,
        dataset_path: str,
        f2p_only: bool,
    ) -> None:
        self.cmds_at_call = {k: p.test_cmd for k, p in self.registry.profiles.items()}
        self.calls.append(
            {
                "run_id": run_id,
                "workers": workers,
                "predictions_path": predictions_path,
                "dataset_path": dataset_path,
                "f2p_only": f2p_only,
            }
        )


def test_apply_profile_overrides_once_per_repo() -> None:
    registry = FakeRegistry()
    rows = [row("a", "srt"), row("b", "srt"), row("c", "thrift"), row("d", "openh264")]
    apply_profile_overrides(registry, rows, 900)
    assert sorted(registry.requested) == sorted(REPOS.values())
    assert all(p.timeout == 900 for p in registry.profiles.values())
    for key in IMAGES:
        assert registry.profiles[REPOS[key]].test_cmd == build_test_command(key, ORIGINAL[key])


def test_run_eval_overrides_before_calling_eval(tmp_path: Path) -> None:
    registry = FakeRegistry()
    fake_eval = FakeEval(registry)
    run_eval(
        fake_eval,
        registry,
        [row("a", "srt")],
        "r1",
        2,
        900,
        tmp_path / "p.json",
        tmp_path / "d.jsonl",
    )
    assert fake_eval.calls == [
        {
            "run_id": "r1",
            "workers": 2,
            "predictions_path": str(tmp_path / "p.json"),
            "dataset_path": str(tmp_path / "d.jsonl"),
            "f2p_only": False,
        }
    ]
    assert fake_eval.cmds_at_call[REPOS["srt"]].startswith("cd build && make -j$(nproc) && ")
    assert fake_eval.cmds_at_call[REPOS["thrift"]] == ORIGINAL["thrift"]
