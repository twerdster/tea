#!/usr/bin/env python3

import argparse
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    import numpy as np
    import scipy.io as sio
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency {exc.name!r}. Install HandNet tooling with: "
        "python3 -m pip install -r requirements-tools.txt"
    ) from exc


def run(cmd, cwd):
    print(" ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def make_frame(frame_index):
    rows, cols = np.indices((32, 32), dtype=np.float32)
    depth = 450.0 + rows * 4.0 + cols * 2.0 + frame_index * 7.0
    label = np.zeros((32, 32), dtype=np.uint16)

    hand = (rows >= 4) & (rows < 28) & (cols >= 4) & (cols < 28)
    depth[~hand] = 0.0
    label_ids = rows[hand].astype(np.int32) // 4 + cols[hand].astype(np.int32) // 5 + frame_index
    label[hand] = (label_ids % 7 + 1).astype(np.uint16)
    return {"depth": depth.astype(np.float32), "lbl": label}


def write_synthetic_archive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for frame_index in range(4):
            buffer = io.BytesIO()
            sio.savemat(buffer, make_frame(frame_index), do_compression=False)
            archive.writestr(f"ValidationData/Data_{frame_index + 1:06d}.mat", buffer.getvalue())


def read_metadata(dataset_dir):
    with (dataset_dir / "handnet_metadata.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_dataset(dataset_dir, feature_count, feature_type_bytes):
    metadata = read_metadata(dataset_dir)
    samples = int(metadata["samples"])
    if samples != 1024:
        raise AssertionError(f"expected 1024 converted samples, got {samples}")

    expected_feature_size = samples * feature_type_bytes
    for feature_id in range(feature_count):
        path = dataset_dir / f"F_{feature_id:04d}.feat"
        if path.stat().st_size != expected_feature_size:
            raise AssertionError(f"{path} has unexpected size {path.stat().st_size}")

    expected_sizes = {
        "Labels.lbl": samples * 2,
        "Poses.pose": samples * 4,
        "Threshholds.thr": feature_count * 3 * 4,
        "Coords.co": feature_count * 4 * 4,
    }
    for name, expected_size in expected_sizes.items():
        actual_size = (dataset_dir / name).stat().st_size
        if actual_size != expected_size:
            raise AssertionError(f"{name} has size {actual_size}, expected {expected_size}")

    labels = np.fromfile(dataset_dir / "Labels.lbl", dtype=np.uint16)
    if labels.min() != 0 or labels.max() != 6:
        raise AssertionError(f"expected zero-based HandNet labels 0..6, got {labels.min()}..{labels.max()}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the Python HandNet-to-Tea converter.")
    parser.add_argument("--tea", default="./Tea", help="Path to the Tea binary. Default: ./Tea")
    parser.add_argument(
        "--work-dir",
        default="/tmp/tea-handnet-converter-smoke",
        help="Directory for synthetic archive and converted data. Default: /tmp/tea-handnet-converter-smoke",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    tea = Path(args.tea)
    if not tea.is_absolute():
        tea = repo / tea

    work_dir = Path(args.work_dir)
    archive = work_dir / "synthetic_handnet.zip"
    dataset_dir = work_dir / "tea-data"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    dataset_dir.mkdir(parents=True)
    write_synthetic_archive(archive)

    run(
        [
            sys.executable,
            "scripts/handnet_to_tea.py",
            "--archive",
            archive,
            "--output",
            dataset_dir,
            "--features",
            "40",
            "--feature-type",
            "F_CHAR",
            "--samples-per-frame",
            "256",
            "--force",
        ],
        cwd=repo,
    )
    samples = validate_dataset(dataset_dir, feature_count=40, feature_type_bytes=1)

    run(
        [
            sys.executable,
            "scripts/estimate_training_memory.py",
            "--dataset",
            dataset_dir,
            "--max-depth",
            "1",
            "--folding-depth",
            "1",
            "--feature-type",
            "F_CHAR",
            "--vram-cap-mb",
            "500",
            "--require-fit",
        ],
        cwd=repo,
    )

    run(
        [
            tea,
            "40",
            "16",
            "1",
            "1",
            "F_CHAR",
            str(samples),
            "8",
            "0",
            "0",
            "W_ONES",
            f"{dataset_dir}/",
            "handnet_converter_smoke",
            "0",
            "LOG0",
            "Synthetic HandNet converter smoke test",
        ],
        cwd=repo,
    )

    for name in ("Tree_0000.tree", "Tree_0000.tree.labels", "LogFile_handnet_converter_smoke_0000.log"):
        path = dataset_dir / name
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"expected non-empty output file {path}")

    print(f"HandNet converter smoke test passed: {samples} samples in {dataset_dir}")


if __name__ == "__main__":
    main()
