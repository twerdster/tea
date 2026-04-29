#!/usr/bin/env python3

import argparse
import array
import subprocess
from pathlib import Path

from benchmark import estimate_vram_bytes


def read_label_stats(labels_path: Path):
    count = labels_path.stat().st_size // 2
    max_label = 0
    chunk_size = 1024 * 1024
    with labels_path.open("rb") as handle:
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            labels = array.array("H")
            labels.frombytes(data[: len(data) - (len(data) % 2)])
            if labels:
                max_label = max(max_label, max(labels))
    return count, max_label + 1


def query_gpus():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, name, free_mb, total_mb = parts
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "free_mb": float(free_mb),
                "total_mb": float(total_mb),
            }
        )
    return gpus


def main():
    parser = argparse.ArgumentParser(description="Estimate Tea per-device training memory.")
    parser.add_argument("--dataset", help="Tea-format dataset directory containing Labels.lbl.")
    parser.add_argument("--num-samples", type=int, help="Number of samples.")
    parser.add_argument("--num-classes", type=int, help="Number of classes.")
    parser.add_argument("--max-depth", type=int, required=True, help="Tree max depth.")
    parser.add_argument("--folding-depth", type=int, required=True, help="Tree folding depth.")
    parser.add_argument(
        "--feature-type",
        choices=("F_CHAR", "F_SHORT", "F_INT", "F_FLOAT"),
        required=True,
        help="Tea feature type.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index to compare against. Default: 0")
    parser.add_argument("--vram-cap-mb", type=float, help="Optional conservative per-device VRAM cap in MB.")
    parser.add_argument(
        "--require-fit",
        action="store_true",
        help="Exit nonzero if the estimate exceeds free VRAM or --vram-cap-mb.",
    )
    args = parser.parse_args()

    num_samples = args.num_samples
    num_classes = args.num_classes

    if args.dataset:
        dataset = Path(args.dataset)
        detected_samples, detected_classes = read_label_stats(dataset / "Labels.lbl")
        num_samples = num_samples or detected_samples
        num_classes = num_classes or detected_classes

    if num_samples is None or num_classes is None:
        raise SystemExit("Provide --dataset or both --num-samples and --num-classes")

    estimated_bytes = estimate_vram_bytes(
        num_samples=num_samples,
        max_depth=args.max_depth,
        folding_depth=args.folding_depth,
        num_classes=num_classes,
        feature_type=args.feature_type,
    )
    estimated_mb = estimated_bytes / (1024.0 * 1024.0)
    print(f"Estimated Tea per-device VRAM: {estimated_mb:.2f} MB")
    print(f"Inputs: samples={num_samples}, classes={num_classes}, max_depth={args.max_depth}, folding_depth={args.folding_depth}, feature_type={args.feature_type}")

    fits_cap = True
    if args.vram_cap_mb is not None:
        fits_cap = estimated_mb <= args.vram_cap_mb
        print(f"Configured cap: {args.vram_cap_mb:.2f} MB")
        print("Cap result: estimate fits configured cap" if fits_cap else "Cap result: estimate exceeds configured cap")

    gpus = query_gpus()
    selected = next((gpu for gpu in gpus if gpu["index"] == args.device), None)
    if selected:
        print(
            f"GPU {selected['index']}: {selected['name']}, "
            f"free={selected['free_mb']:.0f} MB, total={selected['total_mb']:.0f} MB"
        )
        fits_free = estimated_mb <= selected["free_mb"]
        if not fits_free:
            print("Result: estimate exceeds currently free VRAM")
        else:
            print("Result: estimate fits in currently free VRAM")
    else:
        fits_free = True
        print("GPU memory availability could not be read with nvidia-smi")

    if args.require_fit and (not fits_cap or not fits_free):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
