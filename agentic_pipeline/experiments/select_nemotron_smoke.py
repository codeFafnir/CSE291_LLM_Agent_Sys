"""Sample real easy and medium Nemotron-Terminal tasks into a flat Harbor tasks dir.

Harbor's -p expects task dirs as direct children, so the picked dirs are copied untouched.
"""

import argparse
import random
import shutil
from pathlib import Path

EASY_SET = "easy_5000"
MEDIUM_SET = "medium_20000"


def task_dirs(extracted_dir: Path, set_name: str) -> list[Path]:
    return sorted(path.parent for path in (extracted_dir / set_name).glob("*/*/task.toml"))


def sample(tasks: list[Path], count: int, seed: int) -> list[Path]:
    return random.Random(seed).sample(tasks, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ids-out", type=Path, required=True)
    parser.add_argument("--easy-count", type=int, required=True)
    parser.add_argument("--medium-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    picked = [
        *sample(task_dirs(args.extracted_dir, EASY_SET), args.easy_count, args.seed),
        *sample(task_dirs(args.extracted_dir, MEDIUM_SET), args.medium_count, args.seed),
    ]
    names = [path.name for path in picked]
    if len(set(names)) != len(names):
        raise ValueError(f"task name collision between easy and medium picks: {names}")
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    for path in picked:
        shutil.copytree(path, args.out_dir / path.name)
    args.ids_out.write_text("\n".join(names) + "\n")
    for path in picked:
        print(f"{path.parent.parent.name}/{path.name}")
    print(f"picked={len(picked)} wrote {args.out_dir} and {args.ids_out}")


if __name__ == "__main__":
    main()
