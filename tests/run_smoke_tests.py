#!/usr/bin/env python3

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from gen_smoke_dataset import generate_dataset

CASES = [
    {
        "name": "root-depth0",
        "scenario": "root",
        "max_depth": 0,
        "folding_depth": 0,
        "prefix": "smoke_root",
        "comment": "Root-only smoke test",
    },
    {
        "name": "depth1-balanced",
        "scenario": "depth1",
        "max_depth": 1,
        "folding_depth": 1,
        "prefix": "smoke_depth1",
        "comment": "Depth-1 smoke test",
    },
]


def read_u16_file(path: Path):
    data = path.read_bytes()
    if len(data) % 2 != 0:
        raise ValueError(f"{path} does not contain an even number of bytes")
    return list(struct.unpack("<{}H".format(len(data) // 2), data))


def require_files(paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expected output files:\n" + "\n".join(missing))
    empty = [str(path) for path in paths if path.stat().st_size == 0]
    if empty:
        raise ValueError("Expected output files are empty:\n" + "\n".join(empty))


def run_case(tea_bin: Path, work_dir: Path, case):
    case_dir = work_dir / case["name"]
    dataset = generate_dataset(str(case_dir), case["scenario"])

    cmd = [
        str(tea_bin),
        str(len(dataset["features"])),
        str(len(dataset["features"])),
        str(case["max_depth"]),
        str(case["folding_depth"]),
        "F_FLOAT",
        str(len(dataset["labels"])),
        "4",
        "0",
        "0",
        "W_ONES",
        str(case_dir) + "/",
        case["prefix"],
        "0",
        "LOG1",
        case["comment"],
    ]

    print(f"\n== Running {case['name']} ==")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{case['name']} failed with exit code {result.returncode}")

    log_file = case_dir / f"LogFile_{case['prefix']}_0000.log"
    tree_file = case_dir / "Tree_0000.tree"
    labels_file = case_dir / "Tree_0000.tree.labels"
    require_files([log_file, tree_file, labels_file])

    predicted = read_u16_file(labels_file)
    if len(predicted) != len(dataset["labels"]):
        raise AssertionError(
            f"{case['name']} wrote {len(predicted)} predicted labels, "
            f"expected {len(dataset['labels'])}"
        )
    if predicted != dataset["labels"]:
        raise AssertionError(
            f"{case['name']} predicted labels do not match the expected labels.\n"
            f"Expected: {dataset['labels']}\n"
            f"Actual:   {predicted}"
        )

    log_text = log_file.read_text()
    expected_line = f"Depth {case['max_depth']}: 100.0000%"
    if expected_line not in log_text:
        raise AssertionError(f"{case['name']} log does not contain expected line: {expected_line}")

    return {
        "name": case["name"],
        "case_dir": case_dir,
        "log_file": log_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Tea smoke tests.")
    parser.add_argument(
        "--tea",
        default="./Tea",
        help="Path to the Tea executable to test. Default: ./Tea",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/tea-smoke-tests",
        help="Directory to write generated datasets and outputs into. Default: /tmp/tea-smoke-tests",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep any existing work directory contents instead of deleting them first.",
    )
    parser.add_argument(
        "--case",
        choices=[case["name"] for case in CASES],
        action="append",
        help="Run only the named smoke-test case. Repeat to run more than one case.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available smoke-test cases and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for case in CASES:
            print(case["name"])
        return

    tea_bin = Path(args.tea).resolve()
    if not tea_bin.exists():
        raise FileNotFoundError(f"Tea binary not found: {tea_bin}")

    work_dir = Path(args.work_dir).resolve()
    if work_dir.exists() and not args.keep:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.case or [])
    cases = [case for case in CASES if not selected or case["name"] in selected]

    results = [run_case(tea_bin, work_dir, case) for case in cases]
    print("\nSmoke tests passed:")
    for result in results:
        print(f"- {result['name']} -> {result['case_dir']}")
        print(f"  log: {result['log_file']}")


if __name__ == "__main__":
    main()
