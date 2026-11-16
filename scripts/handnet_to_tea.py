#!/usr/bin/env python3
"""Bounded, reproducible HandNet -> native Tea column conversion.

Coordinates are pixel*millimetres, depths are millimetres. Coords.co is reusable
across every split; sampling RNGs never affect coordinate generation.
"""
import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import scipy.io as sio

FEATURE_TYPES = {
    "F_CHAR": {"dtype": np.int8, "feature_max": 127.0, "mat_dtype": "int8"},
    "F_SHORT": {"dtype": np.int16, "feature_max": 32767.0, "mat_dtype": "int16"},
    "F_INT": {"dtype": np.int32, "feature_max": 2147483500.0, "mat_dtype": "int32"},
    "F_FLOAT": {"dtype": np.float32, "feature_max": 3.3e37, "mat_dtype": "single"},
}
CLASS_NAMES = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]
FEATURE_PROFILE = {
    "depth_unit": "millimetres", "coordinate_unit": "pixel*millimetres",
    "max_diff": 250.0, "coordinate_rule": "truncate-after-base-plus-offset-over-depth",
    "coordinate_arithmetic": "float32", "coordinate_precision": "float32-complete-sum", "quantization": "ceil-after-clamp",
    "missing": {"both": 0, "first": 127, "second": -127},
}


def write_json_atomic(path, document):
    """Completion markers must become visible as a complete JSONdocument."""
    path=Path(path)
    descriptor,name=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(descriptor,"w") as handle:
            json.dump(document,handle,indent=2,sort_keys=True,allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name,path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


class ZipSource:
    def __init__(self, path):
        self.path = path
        self.archive = zipfile.ZipFile(path)
    def names(self):
        return sorted(n for n in self.archive.namelist() if re.search(r"(?:^|/)Data_\d+\.mat$", n))
    def read(self, name):
        return self.archive.read(name)
    def close(self):
        self.archive.close()


class TarSource:
    def __init__(self, path):
        self.path = path
        if not shutil.which("bsdtar"):
            raise RuntimeError("bsdtar is required for non-zip archives; extract selected frames once for large RAR slices")
    def names(self):
        result = subprocess.run(["bsdtar", "-tf", str(self.path)], capture_output=True, text=True, check=True)
        return sorted(n for n in result.stdout.splitlines() if re.search(r"(?:^|/)Data_\d+\.mat$", n))
    def read(self, name):
        return subprocess.check_output(["bsdtar", "-xOf", str(self.path), name])
    def close(self):
        pass


class DirectorySource:
    def __init__(self, path):
        self.path = path
    def names(self):
        return sorted(str(p.relative_to(self.path)) for p in self.path.rglob("Data_*.mat"))
    def read(self, name):
        path = (self.path / name).resolve()
        if not path.is_relative_to(self.path.resolve()):
            raise ValueError("Frame path escapes source directory")
        return path.read_bytes()
    def close(self):
        pass


def open_source(path):
    path = Path(path)
    if path.is_dir():
        return DirectorySource(path)
    return ZipSource(path) if path.suffix.lower() == ".zip" else TarSource(path)


def exp_profile(steepness, count):
    values = np.exp(np.linspace(-steepness, 0, count, dtype=np.float32))
    return (values - math.exp(-steepness)) / (1 - math.exp(-steepness))


def generate_coords(num_features, probe_size, seed):
    if num_features <= 0 or num_features % 20:
        raise ValueError("--features must be a positive multiple of 20 for the radial pattern")
    if not math.isfinite(probe_size) or probe_size <= 0:
        raise ValueError("probe size must be finite and positive")
    rng = np.random.default_rng(seed)
    radii = probe_size * exp_profile(1.3, 21)[1:] / 2
    radii = radii[rng.permutation(20)]
    num_rots = num_features // 20
    delta = 2 * math.pi / num_rots
    base = []
    for index in range(20):
        angle = delta * index / 20 + rng.random() * delta * .5
        base.append((-math.sin(angle) * radii[index], math.cos(angle) * radii[index]))
    base = np.asarray(base, dtype=np.float32).T
    first = []
    for index in range(1, num_rots + 1):
        angle = delta * index
        rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
        first.append(rotation @ base)
    first = np.concatenate(first, axis=1)
    second = (rng.random(first.shape, dtype=np.float32) - .5) * .1 * probe_size
    return np.vstack([first, second]).astype(np.float32)


def load_coords(path):
    values = np.fromfile(path, dtype="<f4")
    if not values.size or values.size % 4 or not np.isfinite(values).all():
        raise ValueError("Coords.co must contain finite little-endian float32 [row1,col1,row2,col2] records")
    return values.reshape(-1, 4).T.copy()


def stable_seed(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def select_samples(label, max_samples, rng, depth=None, mode="uniform", class_samples=None, align=4, include_background=False):
    eligible = (label >= (0 if include_background else 1)) & (label <= 7)
    class_count = 8 if include_background else 7
    if depth is not None:
        eligible &= np.isfinite(depth) & (depth > 0)
    rows, cols = np.nonzero(eligible)
    if mode == "balanced" or class_samples is not None:
        if class_samples is None:
            if not max_samples:
                raise ValueError("balanced sampling requires --samples-per-frame or --class-samples")
            quotas = [max_samples // class_count] * class_count
        else:
            quotas = class_samples
        selected = []
        classes = label[rows, cols].astype(np.int64)-1
        if include_background:
            classes[classes < 0] = 7
        for class_id, quota in enumerate(quotas):
            members = np.flatnonzero(classes == class_id)
            selected.extend(rng.choice(members, min(len(members), quota), replace=False).tolist())
        selected = np.asarray(selected, dtype=np.int64)
        if max_samples and len(selected) > max_samples:
            selected = rng.choice(selected, max_samples, replace=False)
    elif max_samples and rows.size > max_samples:
        selected = rng.choice(rows.size, max_samples, replace=False)
    else:
        selected = np.arange(rows.size)
    rng.shuffle(selected)
    selected = selected[:len(selected) - len(selected) % align]
    return rows[selected].astype(np.int64), cols[selected].astype(np.int64)


def compute_features(depth, rows, cols, coords, feature_max=127., output_dtype=np.int8, chunk_size=64, max_diff=250.):
    """Literal MEX arithmetic: all coordinate operations float32, trunc total."""
    depth = np.asarray(depth, dtype=np.float32)
    rows, cols = np.asarray(rows), np.asarray(cols)
    coords = np.asarray(coords, dtype=np.float32)
    if depth.ndim != 2 or rows.shape != cols.shape or coords.ndim != 2 or coords.shape[0] != 4:
        raise ValueError("invalid depth, sample or coordinate shape")
    if chunk_size <= 0 or max_diff <= 0 or not np.isfinite(depth).all() or not np.isfinite(coords).all():
        raise ValueError("features require finite depth/coords and positive chunk size/max difference")
    z = depth[rows, cols]
    if np.any(z <= 0):
        raise ValueError("feature samples must have positive centre depth")
    output = np.empty((len(rows), coords.shape[1]), dtype=output_dtype)
    for start in range(0, coords.shape[1], chunk_size):
        end = min(start + chunk_size, coords.shape[1])
        local = coords[:, start:end]
        row1 = (rows.astype(np.float32)[:, None] + local[0][None, :] / z[:, None]).astype(np.int64)
        col1 = (cols.astype(np.float32)[:, None] + local[1][None, :] / z[:, None]).astype(np.int64)
        row2 = (rows.astype(np.float32)[:, None] + local[2][None, :] / z[:, None]).astype(np.int64)
        col2 = (cols.astype(np.float32)[:, None] + local[3][None, :] / z[:, None]).astype(np.int64)
        valid1 = (row1 >= 0) & (row1 < depth.shape[0]) & (col1 >= 0) & (col1 < depth.shape[1])
        valid2 = (row2 >= 0) & (row2 < depth.shape[0]) & (col2 >= 0) & (col2 < depth.shape[1])
        d1 = np.zeros((len(rows), end-start), dtype=np.float32)
        d2 = np.zeros_like(d1)
        d1[valid1], d2[valid2] = depth[row1[valid1], col1[valid1]], depth[row2[valid2], col2[valid2]]
        values = np.zeros_like(d1)
        values[(d1 == 0) & (d2 != 0)] = feature_max
        values[(d1 != 0) & (d2 == 0)] = -feature_max
        present = (d1 != 0) & (d2 != 0)
        values[present] = np.ceil(np.clip((d1[present] - d2[present]) / np.float32(max_diff) * np.float32(feature_max), -feature_max, feature_max))
        output[:, start:end] = values.astype(output_dtype)
    return output


def mat_scalar(value):
    value = np.asarray(value)
    if value.dtype.kind in "US":
        return "".join(value.ravel().tolist())
    if value.size == 1:
        result = value.ravel()[0]
        return result.item() if hasattr(result, "item") else result
    return value.tolist()


def frame_identity(mat, name, split):
    record = {"name": name, "split": split}
    for field in ("session", "idx", "gidx", "sID"):
        if field in mat:
            record[field] = mat_scalar(mat[field])
    # gidx is the source global frame identifier, never a subject id.
    if "gidx" in record:
        record["frame_id"] = f"handnet:gidx:{record['gidx']}"
    else:
        record["frame_id"] = f"{split}:{name}"
    return record


def infer_split(path):
    name = str(path).lower()
    for split in ("validation", "test", "train"):
        if split in name:
            return split
    return "unspecified"


def read_frame_manifest(path, source_names=None, expected_split=None):
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc,dict) or doc.get("schema") != "handnet-frames-v1" or not isinstance(doc.get("frames"), list) or not doc["frames"]:
        raise ValueError("Expected nonempty handnet-frames-v1 manifest")
    split = doc.get("split")
    if split not in ("train", "validation", "test", "unspecified") or (expected_split and split != expected_split):
        raise ValueError("frame manifest split is invalid or conflicts with requested split")
    names = []
    seen_names = set()
    seen_ids = set()
    for frame in doc["frames"]:
        if not isinstance(frame,dict):
            raise ValueError("frame records must be objects")
        name, frame_id = frame.get("name"), frame.get("frame_id")
        if not isinstance(name, str) or not name or not isinstance(frame_id, str) or not frame_id or frame.get("split") != split:
            raise ValueError("frame records require name, unique frame_id and consistent split")
        if name in seen_names or frame_id in seen_ids:
            raise ValueError("duplicate frame name or identity in manifest")
        if source_names is not None and name not in source_names:
            raise ValueError(f"manifest frame is missing from source: {name}")
        names.append(name)
        seen_names.add(name)
        seen_ids.add(frame_id)
    return doc


def remove_existing_outputs(output):
    for path in output.glob("F_*.feat"):
        path.unlink()
    for name in ("Labels.lbl", "Threshholds.thr", "Coords.co", "Poses.pose", "Pixels.pix", "handnet_metadata.json", "frames.json"):
        (output / name).unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, help="HandNet zip/rar or extracted frame directory")
    p.add_argument("--output", required=True)
    p.add_argument("--features", type=int, default=200)
    p.add_argument("--coords", help="Reuse an existing Coords.co; feature count inferred and generation ignored")
    p.add_argument("--probe-size", type=float, default=50000., help="Pixel*mm diameter; default50000 gives max~55px at453mm. Historical50 remains available.")
    p.add_argument("--feature-type", choices=sorted(FEATURE_TYPES), default="F_CHAR")
    p.add_argument("--seed", type=int, default=0, help="Coordinate seed; legacy default for sampling seeds")
    p.add_argument("--sampling-seed", type=int)
    p.add_argument("--frame-seed", type=int)
    p.add_argument("--max-files", type=int)
    p.add_argument("--start-file", type=int, default=0, help="Legacy ordered slice offset; prefer random --frame-seed")
    p.add_argument("--frame-fraction", type=float, default=1.)
    p.add_argument("--split", choices=["train", "validation", "test", "unspecified"])
    p.add_argument("--frame-manifest", help="Reuse exact frames.json selection; split and identities validated")
    p.add_argument("--manifest-only", action="store_true", help="Record selected frames and identities without features")
    p.add_argument("--samples-per-frame", type=int)
    p.add_argument("--sampling", choices=["uniform", "balanced"], default="uniform")
    p.add_argument("--include-background", action="store_true", help="Learn positive-depth rawlabel0 as class7 (eightclasses)")
    p.add_argument("--class-samples", help="Per-frame modeledclass caps (seven, or eightwithbackground appended); no replacement")
    p.add_argument("--sample-alignment", type=int, choices=[1, 4], default=4)
    p.add_argument("--feature-chunk-size", type=int, default=64)
    p.add_argument("--max-output-gib", type=float, default=100., help="Refuse estimated upperbound beyondthissize beforewriting")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.start_file < 0 or args.feature_chunk_size < 1 or not 0 < args.frame_fraction <= 1:
        p.error("invalid start, chunk size or frame fraction")
    if any(v is not None and v <= 0 for v in (args.max_files, args.samples_per_frame)) or args.max_output_gib <= 0:
        p.error("sample/file caps and output bound must be positive")
    quotas = [int(v) for v in args.class_samples.split(",")] if args.class_samples else None
    class_count = 8 if args.include_background else 7
    class_names = CLASS_NAMES + (["background"] if args.include_background else [])
    if quotas is not None and (len(quotas) != class_count or min(quotas) < 0 or not sum(quotas)):
        p.error(f"--class-samples requires {class_count} nonnegative counts with nonzero sum")
    if args.sampling == "balanced" and not args.samples_per_frame and quotas is None:
        p.error("balanced sampling requires samples-per-frame or class-samples")
    archive, output = Path(args.archive).resolve(), Path(args.output).resolve()
    info = FEATURE_TYPES[args.feature_type]
    coords = load_coords(args.coords) if args.coords else generate_coords(args.features, args.probe_size, args.seed)
    features = coords.shape[1]
    split = args.split or infer_split(archive)
    source = open_source(archive)
    try:
        all_names = source.names()
        if args.frame_manifest:
            manifest = read_frame_manifest(args.frame_manifest, set(all_names), args.split)
            split = manifest["split"]
            names = [f["name"] for f in manifest["frames"]]
        else:
            names = all_names[args.start_file:]
            count = min(len(names), args.max_files or len(names), math.ceil(len(names) * args.frame_fraction))
            if args.frame_seed is not None:
                chosen = np.random.default_rng(args.frame_seed).choice(len(names), count, replace=False)
                names = [names[i] for i in sorted(chosen.tolist())]
            else:
                names = names[:count]
        if not names:
            p.error("No selected Data_*.mat frames")
        cap = min(args.samples_per_frame or 320*240, sum(quotas) if quotas else 320*240)
        upper_bytes = len(names) * cap * (features * np.dtype(info["dtype"]).itemsize + 10)
        if not args.manifest_only and upper_bytes > args.max_output_gib * 1024**3:
            p.error(f"output upperbound {upper_bytes/1024**3:.3f}GiB exceeds --max-output-gib")
        output.mkdir(parents=True, exist_ok=True)
        if args.force:
            remove_existing_outputs(output)
        elif any(output.glob("F_*.feat")) or any((output/n).exists() for n in ("Labels.lbl", "frames.json", "handnet_metadata.json")):
            p.error("Output exists; use --force")
        if shutil.disk_usage(output).free < upper_bytes * 1.05 and not args.manifest_only:
            p.error("insufficient free space for bounded output plus5%margin")
        sampling_seed = args.seed if args.sampling_seed is None else args.sampling_seed
        frames, total, hist = [], 0, np.zeros(class_count, dtype=np.int64)
        seen_frame_ids = set()
        f_min = np.full(features, info["feature_max"], dtype=np.float64)
        f_max = -f_min.copy()
        with ExitStack() as stack:
            if not args.manifest_only:
                # 1,200 columns need >the usual macOS256/Linux1024 softlimit.
                # Raise this process only, within its existing hardlimit.
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                needed = features + 64
                if soft < needed:
                    if hard != resource.RLIM_INFINITY and hard < needed:
                        raise ValueError(f"need process filedescriptor limit{needed}, hardlimit is{hard}")
                    resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
                feature_handles = [stack.enter_context((output/f"F_{i:04d}.feat").open("wb")) for i in range(features)]
                labels_handle = stack.enter_context((output/"Labels.lbl").open("wb"))
                poses_handle = stack.enter_context((output/"Poses.pose").open("wb"))
                pixels_handle = stack.enter_context((output/"Pixels.pix").open("wb"))
            for i, name in enumerate(names):
                mat = sio.loadmat(io.BytesIO(source.read(name)), variable_names=["depth", "lbl", "session", "idx", "gidx", "sID"])
                if "depth" not in mat or "lbl" not in mat:
                    raise ValueError(f"{name}: missing depth or lbl")
                depth, label = np.asarray(mat["depth"], dtype=np.float32), mat["lbl"]
                if depth.shape != label.shape or depth.ndim != 2 or depth.size > 320*240 or max(depth.shape)>65535 or not np.isfinite(depth).all() or np.any(depth < 0):
                    raise ValueError(f"{name}: invalid depth/label dimensions or depth values")
                if np.any(label < 0) or np.any(label > 7):
                    raise ValueError(f"{name}: expected raw HandNet labels0..7")
                record = frame_identity(mat, name, split)
                if record["frame_id"] in seen_frame_ids:
                    raise ValueError(f"duplicate source frame identity:{record['frame_id']}")
                seen_frame_ids.add(record["frame_id"])
                if args.frame_manifest:
                    original = manifest["frames"][i]
                    if record["frame_id"] != original["frame_id"]:
                        raise ValueError(f"{name}: frame identity changed since input manifest")
                record["shape"] = list(depth.shape)
                record["source_label_hist"] = np.bincount(label.astype(np.int64).ravel(), minlength=8).tolist()
                record["source_positive_depth_label_hist"] = np.bincount(label[depth > 0].astype(np.int64), minlength=8).tolist()
                record["sample_begin"] = total
                if not args.manifest_only:
                    rng = np.random.default_rng(stable_seed(sampling_seed, record["frame_id"]))
                    rows, cols = select_samples(label, args.samples_per_frame, rng, depth, args.sampling, quotas, args.sample_alignment, args.include_background)
                    raw_labels = label[rows, cols].astype(np.int64)
                    labels = np.where(raw_labels == 0, 7, raw_labels - 1).astype("<u2")
                    labels.tofile(labels_handle)
                    np.full(len(rows), i, dtype="<u4").tofile(poses_handle)
                    np.stack([rows, cols], axis=1).astype("<u2").tofile(pixels_handle)
                    values = compute_features(depth, rows, cols, coords, info["feature_max"], info["dtype"], args.feature_chunk_size)
                    for fid, handle in enumerate(feature_handles):
                        column = np.ascontiguousarray(values[:, fid])
                        column.tofile(handle)
                        valid = column[np.abs(column) != info["feature_max"]]
                        if valid.size:
                            f_min[fid] = min(f_min[fid], float(valid.min()))
                            f_max[fid] = max(f_max[fid], float(valid.max()))
                    hist += np.bincount(labels.astype(np.int64), minlength=class_count)
                    total += len(rows)
                    record["samples"] = len(rows)
                frames.append(record)
                if (i + 1) % 100 == 0 or i + 1 == len(names):
                    print(f"processed {i+1}/{len(names)} frames, samples={total}", flush=True)
        if len({f["frame_id"] for f in frames}) != len(frames):
            raise ValueError("duplicate global frame identities detected")
        frame_doc = {"schema": "handnet-frames-v1", "split": split, "source": str(archive), "frames": frames,
                     "selection": {"frame_seed": args.frame_seed, "frame_fraction": args.frame_fraction, "max_files": args.max_files,
                                   "start_file": args.start_file, "input_manifest": args.frame_manifest},
                     "independence": "Official randomized frame split; not a subject-independent split. session/idx/gidx preserve source identities."}
        write_json_atomic(output/"frames.json",frame_doc)
        if args.manifest_only:
            return
        if total == 0:
            raise ValueError("selection produced no samples")
        missing = f_max < f_min
        f_min[missing], f_max[missing] = -info["feature_max"], info["feature_max"]
        thresholds = np.stack([np.full(features, info["feature_max"]), f_min, f_max], axis=1).astype("<f4")
        thresholds.tofile(output/"Threshholds.thr")
        coords.T.astype("<f4").tofile(output/"Coords.co")
        profile = dict(FEATURE_PROFILE, maximum=info["feature_max"], type=args.feature_type)
        profile["missing"] = {"both": 0, "first": info["feature_max"], "second": -info["feature_max"]}
        metadata = {"schema": "handnet-tea-dataset-v2", "archive": str(archive), "output": str(output), "split": split,
                    "files_converted": len(names), "samples": total, "features": features, "feature_type": args.feature_type,
                    "probe_size": None if args.coords else args.probe_size, "seed": args.seed, "sampling_seed": sampling_seed,
                    "frame_seed": args.frame_seed, "samples_per_frame": args.samples_per_frame, "sampling": args.sampling,
                    "class_samples": quotas, "sample_alignment": args.sample_alignment, "label_hist": {str(i):int(v) for i,v in enumerate(hist)},
                    "classes": class_names, "include_background": args.include_background,
                    "modeled_background_class": 7 if args.include_background else None, "invalid_label": class_count,
                    "raw_label_mapping": dict({str(i+1):i for i in range(7)}, **({"0":7} if args.include_background else {})),
                    "excluded_raw_labels": [] if args.include_background else [0],
                    "centre_policy": "positive finite depth and raw labels0..7" if args.include_background else "positive finite depth and raw labels1..7", "feature_profile": profile,
                    "coords_sha256": hashlib.sha256((output/"Coords.co").read_bytes()).hexdigest(),
                    "frame_manifest": "frames.json", "pixel_file": "Pixels.pix", "pixel_encoding": "little-endian uint16 row,col per sample",
                    "output_upper_bound_bytes": upper_bytes, "complete": True}
        write_json_atomic(output/"handnet_metadata.json",metadata)
        print(json.dumps(metadata, sort_keys=True))
    finally:
        source.close()


if __name__ == "__main__":
    main()
