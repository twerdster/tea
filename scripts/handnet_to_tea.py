#!/usr/bin/env python3

import argparse
import io
import json
import math
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio


FEATURE_TYPES = {
    "F_CHAR": {"dtype": np.int8, "feature_max": 127.0, "mat_dtype": "int8"},
    "F_SHORT": {"dtype": np.int16, "feature_max": 32767.0, "mat_dtype": "int16"},
    "F_INT": {"dtype": np.int32, "feature_max": 2147483500.0, "mat_dtype": "int32"},
    "F_FLOAT": {"dtype": np.float32, "feature_max": 3.3e37, "mat_dtype": "single"},
}


class ZipSource:
    def __init__(self, path: Path):
        self.path = path
        self.archive = zipfile.ZipFile(path)

    def names(self):
        return sorted(name for name in self.archive.namelist() if name.endswith(".mat"))

    def read(self, name: str):
        return self.archive.read(name)

    def close(self):
        self.archive.close()


class TarSource:
    def __init__(self, path: Path):
        self.path = path
        if not shutil.which("bsdtar"):
            raise RuntimeError("bsdtar is required to read non-zip archives such as TrainData.rar")

    def names(self):
        result = subprocess.run(["bsdtar", "-tf", str(self.path)], capture_output=True, text=True, check=True)
        return sorted(line for line in result.stdout.splitlines() if line.endswith(".mat"))

    def read(self, name: str):
        return subprocess.check_output(["bsdtar", "-xOf", str(self.path), name])

    def close(self):
        return None


class DirectorySource:
    def __init__(self, path: Path):
        self.path = path

    def names(self):
        return sorted(str(path.relative_to(self.path)) for path in self.path.rglob("Data_*.mat"))

    def read(self, name: str):
        return (self.path / name).read_bytes()

    def close(self):
        return None


def open_source(path: Path):
    if path.is_dir():
        return DirectorySource(path)
    if path.suffix.lower() == ".zip":
        return ZipSource(path)
    return TarSource(path)


def exp_profile(steepness: float, count: int):
    values = np.exp(np.linspace(-steepness, 0.0, count, dtype=np.float32))
    return (values - math.exp(-steepness)) / (1.0 - math.exp(-steepness))


def generate_coords(num_features: int, probe_size: float, seed: int):
    if num_features % 20 != 0:
        raise ValueError("--features must be a multiple of 20 to use the historical HandNet radial pattern")

    rng = np.random.default_rng(seed)
    n_div = 20
    steepness = 1.3
    radii = probe_size * exp_profile(steepness, n_div + 1)
    radii = radii[1:] / 2.0
    radii = radii[rng.permutation(n_div)]
    num_rots = num_features // n_div
    delta = 2.0 * math.pi / float(num_rots)

    base = []
    for index in range(n_div):
        angle = delta * index / float(n_div) + rng.random() * delta * 0.5
        base.append((-math.sin(angle) * radii[index], math.cos(angle) * radii[index]))
    base = np.asarray(base, dtype=np.float32).T

    first_points = []
    for rot_index in range(1, num_rots + 1):
        angle = delta * rot_index
        matrix = np.asarray(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=np.float32,
        )
        first_points.append(matrix @ base)
    first_points = np.concatenate(first_points, axis=1)
    second_points = (rng.random(first_points.shape, dtype=np.float32) - 0.5) * 0.1 * probe_size
    return np.vstack([first_points, second_points]).astype(np.float32)


def select_samples(label, max_samples, rng):
    rows, cols = np.nonzero(label >= 1)
    if max_samples and rows.size > max_samples:
        selected = rng.permutation(rows.size)[:max_samples]
        rows = rows[selected]
        cols = cols[selected]

    aligned = rows.size - (rows.size % 4)
    return rows[:aligned].astype(np.int64), cols[:aligned].astype(np.int64)


def compute_features(depth, rows, cols, coords, feature_max, output_dtype, chunk_size):
    sample_count = rows.size
    feature_count = coords.shape[1]
    z_values = depth[rows, cols].astype(np.float32)
    z_values[z_values == 0] = 1.0
    result_chunks = []

    for start in range(0, feature_count, chunk_size):
        end = min(start + chunk_size, feature_count)
        local = coords[:, start:end]
        row1 = (rows[:, None] + local[0][None, :] / z_values[:, None]).astype(np.int64)
        col1 = (cols[:, None] + local[1][None, :] / z_values[:, None]).astype(np.int64)
        row2 = (rows[:, None] + local[2][None, :] / z_values[:, None]).astype(np.int64)
        col2 = (cols[:, None] + local[3][None, :] / z_values[:, None]).astype(np.int64)

        valid1 = (row1 >= 0) & (row1 < depth.shape[0]) & (col1 >= 0) & (col1 < depth.shape[1])
        valid2 = (row2 >= 0) & (row2 < depth.shape[0]) & (col2 >= 0) & (col2 < depth.shape[1])

        d1 = np.zeros((sample_count, end - start), dtype=np.float32)
        d2 = np.zeros((sample_count, end - start), dtype=np.float32)
        d1[valid1] = depth[row1[valid1], col1[valid1]]
        d2[valid2] = depth[row2[valid2], col2[valid2]]

        values = np.zeros_like(d1)
        only_d2_missing = (d1 != 0) & (d2 == 0)
        only_d1_missing = (d1 == 0) & (d2 != 0)
        both_present = (d1 != 0) & (d2 != 0)
        values[only_d1_missing] = feature_max
        values[only_d2_missing] = -feature_max
        values[both_present] = np.ceil(np.clip((d1[both_present] - d2[both_present]) / 250.0 * feature_max, -feature_max, feature_max))

        if output_dtype == np.float32:
            result_chunks.append(values.astype(np.float32, copy=False))
        else:
            result_chunks.append(values.astype(output_dtype, copy=False))

    return np.concatenate(result_chunks, axis=1)


def remove_existing_outputs(output_dir: Path):
    for path in output_dir.glob("F_*.feat"):
        path.unlink()
    for name in ("Labels.lbl", "Threshholds.thr", "Coords.co", "Poses.pose", "handnet_metadata.json"):
        path = output_dir / name
        if path.exists():
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Convert HandNet Data_*.mat archives into Tea feature-column datasets.")
    parser.add_argument("--archive", required=True, help="HandNet .zip, .rar, or extracted directory containing Data_*.mat files.")
    parser.add_argument("--output", required=True, help="Output Tea dataset directory.")
    parser.add_argument("--features", type=int, default=200, help="Number of generated depth-difference features. Default: 200")
    parser.add_argument("--probe-size", type=float, default=50.0, help="Historical HandNet probe size. Default: 50")
    parser.add_argument("--feature-type", choices=sorted(FEATURE_TYPES), default="F_CHAR", help="Tea feature type. Default: F_CHAR")
    parser.add_argument("--seed", type=int, default=0, help="Feature and sampling seed. Default: 0")
    parser.add_argument("--max-files", type=int, help="Maximum number of Data_*.mat files to convert.")
    parser.add_argument("--start-file", type=int, default=0, help="Start offset into sorted Data_*.mat file list. Default: 0")
    parser.add_argument("--samples-per-frame", type=int, help="Optional cap on sampled hand pixels per frame.")
    parser.add_argument("--feature-chunk-size", type=int, default=64, help="Vectorized feature chunk size. Default: 64")
    parser.add_argument("--force", action="store_true", help="Remove existing Tea output files before converting.")
    args = parser.parse_args()

    archive = Path(args.archive).resolve()
    output = Path(args.output).resolve()
    info = FEATURE_TYPES[args.feature_type]
    coords = generate_coords(args.features, args.probe_size, args.seed)

    output.mkdir(parents=True, exist_ok=True)
    if args.force:
        remove_existing_outputs(output)
    elif any(output.glob("F_*.feat")) or (output / "Labels.lbl").exists():
        raise SystemExit(f"{output} already contains Tea output files; use --force to overwrite")

    source = open_source(archive)
    names = source.names()[args.start_file :]
    if args.max_files:
        names = names[: args.max_files]
    if not names:
        raise SystemExit("No Data_*.mat files found")

    feature_handles = [(output / f"F_{feature_id:04d}.feat").open("wb") for feature_id in range(args.features)]
    labels_handle = (output / "Labels.lbl").open("wb")
    poses_handle = (output / "Poses.pose").open("wb")
    f_min = np.full(args.features, info["feature_max"], dtype=np.float64)
    f_max = np.full(args.features, -info["feature_max"], dtype=np.float64)
    total_samples = 0
    label_hist = {}

    try:
        for file_index, name in enumerate(names):
            mat = sio.loadmat(io.BytesIO(source.read(name)), variable_names=["depth", "lbl"])
            depth = mat["depth"].astype(np.float32, copy=False)
            label = mat["lbl"]
            rng = np.random.default_rng(args.seed + args.start_file + file_index + 1)
            rows, cols = select_samples(label, args.samples_per_frame, rng)
            if rows.size == 0:
                continue

            labels = (label[rows, cols].astype(np.uint16) - 1).astype(np.uint16, copy=False)
            labels.tofile(labels_handle)
            np.full(rows.size, args.start_file + file_index, dtype=np.uint32).tofile(poses_handle)
            values = compute_features(
                depth=depth,
                rows=rows,
                cols=cols,
                coords=coords,
                feature_max=info["feature_max"],
                output_dtype=info["dtype"],
                chunk_size=args.feature_chunk_size,
            )

            sentinel = np.asarray(info["feature_max"], dtype=values.dtype)
            valid = np.abs(values) != sentinel
            for feature_id in range(args.features):
                column = np.ascontiguousarray(values[:, feature_id])
                column.tofile(feature_handles[feature_id])
                valid_column = column[valid[:, feature_id]]
                if valid_column.size:
                    f_min[feature_id] = min(f_min[feature_id], float(valid_column.min()))
                    f_max[feature_id] = max(f_max[feature_id], float(valid_column.max()))

            unique, counts = np.unique(labels, return_counts=True)
            for label_value, count in zip(unique.tolist(), counts.tolist()):
                label_hist[int(label_value)] = label_hist.get(int(label_value), 0) + int(count)
            total_samples += int(rows.size)

            if (file_index + 1) % 100 == 0 or file_index + 1 == len(names):
                print(f"converted {file_index + 1}/{len(names)} files, samples={total_samples}")
    finally:
        for handle in feature_handles:
            handle.close()
        labels_handle.close()
        poses_handle.close()
        source.close()

    missing_valid = f_max < f_min
    f_min[missing_valid] = -info["feature_max"]
    f_max[missing_valid] = info["feature_max"]
    thresholds = np.empty(args.features * 3, dtype=np.float32)
    thresholds[0::3] = info["feature_max"]
    thresholds[1::3] = f_min.astype(np.float32)
    thresholds[2::3] = f_max.astype(np.float32)
    thresholds.tofile(output / "Threshholds.thr")
    coords.T.astype(np.float32).tofile(output / "Coords.co")

    metadata = {
        "archive": str(archive),
        "output": str(output),
        "files_converted": len(names),
        "samples": total_samples,
        "features": args.features,
        "feature_type": args.feature_type,
        "probe_size": args.probe_size,
        "seed": args.seed,
        "samples_per_frame": args.samples_per_frame,
        "label_hist": label_hist,
    }
    (output / "handnet_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
