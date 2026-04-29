#!/usr/bin/env python3

import argparse
import array
import csv
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


BLOCKSIZE_1 = 1 * 1024 * 1024
HISTOGRAM64_BIN_COUNT = 64
MAX_PARTIAL_HISTOGRAM64_COUNT = 2 * 32768
PARTIAL_HISTOGRAM_BYTES = MAX_PARTIAL_HISTOGRAM64_COUNT * HISTOGRAM64_BIN_COUNT * 4
RFTRAINNODE_BYTES_ESTIMATE = 48
SAMPLE_BYTES_ESTIMATE = 2
SAFETY_HEADROOM_BYTES = 32 * 1024 * 1024

FEATURE_TYPES = {
    "F_CHAR": {"dtype": "b", "bytes": 1, "max_proxy": 127.0},
    "F_SHORT": {"dtype": "h", "bytes": 2, "max_proxy": 32767.0},
    "F_INT": {"dtype": "i", "bytes": 4, "max_proxy": 2147483500.0},
    "F_FLOAT": {"dtype": "f", "bytes": 4, "max_proxy": 1024.0},
}

TREE_TIME_RE = re.compile(r"Processed Tree \d+ in ([0-9]*\.?[0-9]+)s")
DEPTH_RE = re.compile(r"Depth (\d+): ([0-9]*\.?[0-9]+)%")


@dataclass(frozen=True)
class BenchmarkCase:
    suite: str
    name: str
    num_samples: int
    num_features: int
    max_depth: int
    folding_depth: int
    num_thresh: int
    pool_size: int


def parse_int_list(raw: str):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def feature_type_info(feature_type: str):
    try:
        return FEATURE_TYPES[feature_type]
    except KeyError as exc:
        supported = ", ".join(sorted(FEATURE_TYPES))
        raise ValueError(f"Unsupported feature type '{feature_type}'. Supported: {supported}") from exc


def estimate_vram_bytes(num_samples: int, max_depth: int, folding_depth: int, num_classes: int, feature_type: str, num_streams: int = 1):
    feature_bytes = feature_type_info(feature_type)["bytes"]
    max_nodes = int(pow(2.0, max_depth + 1) - 1)
    hist_mem = 2 * num_classes * int(pow(2.0, folding_depth)) * 4
    node_mem = 2 * max_nodes * RFTRAINNODE_BYTES_ESTIMATE
    sample_mem = num_samples * SAMPLE_BYTES_ESTIMATE
    classes_mem = num_classes * 4
    worker_feature_mem = num_samples * feature_bytes
    worker_index_buf_mem = BLOCKSIZE_1 * 4
    worker_data_buf_mem = BLOCKSIZE_1 * feature_bytes
    per_worker_mem = worker_feature_mem + worker_index_buf_mem + worker_data_buf_mem

    total = (
        PARTIAL_HISTOGRAM_BYTES
        + hist_mem
        + node_mem
        + sample_mem
        + classes_mem
        + per_worker_mem * num_streams
        + SAFETY_HEADROOM_BYTES
    )
    return total


def choose_num_classes(depth: int):
    return min(1 << max(depth, 1), 256)


def value_cycle(num_classes: int, feature_index: int, informative_depth: int, feature_type: str):
    info = feature_type_info(feature_type)
    dtype = info["dtype"]

    if feature_index < informative_depth:
        bit_index = informative_depth - 1 - feature_index
        values = [-1 if ((leaf >> bit_index) & 1) == 0 else 1 for leaf in range(num_classes)]
    else:
        values = [((leaf * (feature_index + 3) + feature_index) % 5) - 2 for leaf in range(num_classes)]

    if feature_type == "F_FLOAT":
        return array.array(dtype, [float(value) for value in values])
    return array.array(dtype, values)


def repeated_array(cycle: array.array, total_count: int):
    repeats, remainder = divmod(total_count, len(cycle))
    return cycle * repeats + cycle[:remainder]


def write_array(path: Path, values: array.array):
    with path.open("wb") as handle:
        values.tofile(handle)


def generate_dataset(dataset_dir: Path, num_samples: int, num_features: int, separable_depth: int, feature_type: str):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    info = feature_type_info(feature_type)
    num_classes = choose_num_classes(separable_depth)

    label_cycle = array.array("H", range(num_classes))
    labels = repeated_array(label_cycle, num_samples)
    write_array(dataset_dir / "Labels.lbl", labels)

    thresholds = array.array("f")
    for feature_index in range(num_features):
        cycle = value_cycle(num_classes, feature_index, separable_depth, feature_type)
        values = repeated_array(cycle, num_samples)
        write_array(dataset_dir / f"F_{feature_index:04d}.feat", values)

        feature_min = float(min(cycle))
        feature_max = float(max(cycle))
        thresholds.extend([float(info["max_proxy"]), feature_min, feature_max])

    write_array(dataset_dir / "Threshholds.thr", thresholds)
    return num_classes


def build_cases(samples, features, depths, num_thresh):
    samples_sorted = sorted(samples)
    features_sorted = sorted(features)
    depths_sorted = sorted(depths)

    mid_sample = samples_sorted[len(samples_sorted) // 2]
    mid_feature = features_sorted[len(features_sorted) // 2]
    max_depth = max(depths_sorted)

    cases = []
    for sample in samples_sorted:
        pool_size = min(mid_feature, 32)
        cases.append(
            BenchmarkCase(
                suite="samples",
                name=f"samples-s{sample}-f{mid_feature}-d{max_depth}",
                num_samples=sample,
                num_features=mid_feature,
                max_depth=max_depth,
                folding_depth=max_depth,
                num_thresh=num_thresh,
                pool_size=pool_size,
            )
        )

    for feature in features_sorted:
        pool_size = min(feature, 32)
        cases.append(
            BenchmarkCase(
                suite="features",
                name=f"features-s{mid_sample}-f{feature}-d{max_depth}",
                num_samples=mid_sample,
                num_features=feature,
                max_depth=max_depth,
                folding_depth=max_depth,
                num_thresh=num_thresh,
                pool_size=pool_size,
            )
        )

    for depth in depths_sorted:
        pool_size = min(mid_feature, 32)
        cases.append(
            BenchmarkCase(
                suite="depths",
                name=f"depths-s{mid_sample}-f{mid_feature}-d{depth}",
                num_samples=mid_sample,
                num_features=mid_feature,
                max_depth=depth,
                folding_depth=depth,
                num_thresh=num_thresh,
                pool_size=pool_size,
            )
        )

    return cases


def parse_log(log_path: Path):
    log_text = log_path.read_text()
    tree_match = TREE_TIME_RE.findall(log_text)
    tree_time_sec = float(tree_match[-1]) if tree_match else None

    final_depth = None
    final_success_pct = None
    for depth_str, success_str in DEPTH_RE.findall(log_text):
        final_depth = int(depth_str)
        final_success_pct = float(success_str)

    return {
        "tree_time_sec": tree_time_sec,
        "final_depth": final_depth,
        "final_success_pct": final_success_pct,
    }


def run_case(tea_bin: Path, case: BenchmarkCase, dataset_dir: Path, feature_type: str, work_dir: Path):
    prefix = case.name.replace("_", "-")
    cmd = [
        str(tea_bin),
        str(case.num_features),
        str(case.pool_size),
        str(case.max_depth),
        str(case.folding_depth),
        feature_type,
        str(case.num_samples),
        str(case.num_thresh),
        "0",
        "0",
        "W_ONES",
        str(dataset_dir) + "/",
        prefix,
        "0",
        "LOG0",
        f"benchmark {case.name}",
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"{case.name} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    log_file = dataset_dir / f"LogFile_{prefix}_0000.log"
    if not log_file.exists():
        raise FileNotFoundError(f"Expected benchmark log file was not written: {log_file}")

    parsed = parse_log(log_file)
    return {
        "suite": case.suite,
        "name": case.name,
        "num_samples": case.num_samples,
        "num_features": case.num_features,
        "max_depth": case.max_depth,
        "folding_depth": case.folding_depth,
        "num_thresh": case.num_thresh,
        "pool_size": case.pool_size,
        "dataset_dir": str(dataset_dir),
        "elapsed_sec": elapsed,
        "tree_time_sec": parsed["tree_time_sec"],
        "final_depth": parsed["final_depth"],
        "final_success_pct": parsed["final_success_pct"],
        "log_file": str(log_file),
    }


def write_results(csv_path: Path, rows):
    fieldnames = [
        "suite",
        "name",
        "num_samples",
        "num_features",
        "max_depth",
        "folding_depth",
        "num_thresh",
        "pool_size",
        "estimated_vram_mb",
        "num_classes",
        "dataset_dir",
        "elapsed_sec",
        "tree_time_sec",
        "final_depth",
        "final_success_pct",
        "log_file",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nBenchmark results:")
    for row in rows:
        tree_time = "n/a" if row["tree_time_sec"] is None else f"{row['tree_time_sec']:.4f}s"
        success = "n/a" if row["final_success_pct"] is None else f"{row['final_success_pct']:.2f}%"
        print(
            f"- {row['name']}: "
            f"samples={row['num_samples']}, "
            f"features={row['num_features']}, "
            f"depth={row['max_depth']}, "
            f"est_vram={row['estimated_vram_mb']:.1f}MB, "
            f"tree_time={tree_time}, "
            f"success={success}"
        )


def main():
    parser = argparse.ArgumentParser(description="Generate bounded Tea benchmark datasets and run speed sweeps.")
    parser.add_argument("--tea", default="./Tea", help="Path to the Tea executable. Default: ./Tea")
    parser.add_argument("--work-dir", default=f"/tmp/tea-bench-{int(time.time())}", help="Benchmark workspace directory.")
    parser.add_argument("--samples", default="250000,1000000,4000000", help="Comma-separated sample counts.")
    parser.add_argument("--features", default="32,128,512", help="Comma-separated feature counts.")
    parser.add_argument("--depths", default="1,2,4,6", help="Comma-separated tree depths.")
    parser.add_argument("--num-thresh", type=int, default=8, help="Threshold count per feature. Default: 8")
    parser.add_argument("--feature-type", choices=sorted(FEATURE_TYPES), default="F_CHAR", help="Tea feature type to benchmark.")
    parser.add_argument("--vram-cap-mb", type=float, default=500.0, help="Per-device VRAM cap in MB. Default: 500")
    args = parser.parse_args()

    tea_bin = Path(args.tea).resolve()
    if not tea_bin.exists():
        raise FileNotFoundError(f"Tea binary not found: {tea_bin}")

    work_dir = Path(args.work_dir).resolve()
    datasets_dir = work_dir / "datasets"
    results_dir = work_dir / "results"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    samples = parse_int_list(args.samples)
    features = parse_int_list(args.features)
    depths = parse_int_list(args.depths)
    cases = build_cases(samples, features, depths, args.num_thresh)

    rows = []
    for case in cases:
        num_classes = choose_num_classes(case.max_depth)
        est_vram_bytes = estimate_vram_bytes(
            num_samples=case.num_samples,
            max_depth=case.max_depth,
            folding_depth=case.folding_depth,
            num_classes=num_classes,
            feature_type=args.feature_type,
        )
        est_vram_mb = est_vram_bytes / (1024.0 * 1024.0)
        if est_vram_mb > args.vram_cap_mb:
            raise RuntimeError(
                f"{case.name} is estimated to require {est_vram_mb:.1f} MB of VRAM, "
                f"which exceeds the configured cap of {args.vram_cap_mb:.1f} MB"
            )

        dataset_name = f"data-s{case.num_samples}-f{case.num_features}-d{case.max_depth}-{args.feature_type}"
        dataset_dir = datasets_dir / dataset_name
        if not (dataset_dir / "Labels.lbl").exists():
            generate_dataset(
                dataset_dir=dataset_dir,
                num_samples=case.num_samples,
                num_features=case.num_features,
                separable_depth=case.max_depth,
                feature_type=args.feature_type,
            )

        row = run_case(tea_bin, case, dataset_dir, args.feature_type, work_dir)
        row["estimated_vram_mb"] = round(est_vram_mb, 2)
        row["num_classes"] = num_classes
        rows.append(row)

    csv_path = results_dir / "benchmark_results.csv"
    write_results(csv_path, rows)
    print_summary(rows)
    print(f"\nWrote results to {csv_path}")


if __name__ == "__main__":
    main()
