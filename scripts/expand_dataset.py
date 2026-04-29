#!/usr/bin/env python3

import argparse
import array
import math
import random
from pathlib import Path


DTYPES = {
    "int8": ("b", 1),
    "int16": ("h", 2),
    "int32": ("i", 4),
    "single": ("f", 4),
    "float32": ("f", 4),
    "uint16": ("H", 2),
}


def require_dtype(name: str):
    try:
        return DTYPES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(name for name in DTYPES if name != "uint16"))
        raise ValueError(f"Unsupported dtype '{name}'. Supported values: {supported}") from exc


def feature_files(data_dir: Path):
    files = sorted(data_dir.glob("F_*.feat"))
    if not files:
        raise FileNotFoundError(f"No feature files found under {data_dir}")
    return files


def read_array(path: Path, dtype_name: str):
    typecode, size = DTYPES[dtype_name]
    data = array.array(typecode)
    with path.open("rb") as handle:
        file_size = path.stat().st_size
        if file_size % size != 0:
            raise ValueError(f"{path} size {file_size} is not divisible by dtype size {size}")
        data.fromfile(handle, file_size // size)
    return data


def write_array(path: Path, values):
    with path.open("wb") as handle:
        values.tofile(handle)


def detect_dataset(data_dir: Path, input_dtype: str):
    features = feature_files(data_dir)
    sample_size = require_dtype(input_dtype)[1]
    first_feature = features[0]
    num_samples = first_feature.stat().st_size // sample_size
    labels = read_array(data_dir / "Labels.lbl", "uint16")
    num_classes = max(labels) + 1 if labels else 0
    return {
        "num_features": len(features),
        "num_samples": num_samples,
        "num_classes": num_classes,
        "features": features,
    }


def tile_indices(out_count: int, in_count: int):
    return [index % in_count for index in range(out_count)]


def expand_labels(input_dir: Path, output_dir: Path, indices, out_num_classes: int, distribution: str, seed: int):
    labels = read_array(input_dir / "Labels.lbl", "uint16")
    if not labels:
        raise ValueError("Labels.lbl is empty")

    in_num_classes = max(labels) + 1
    divisors = int(math.ceil(float(out_num_classes) / float(in_num_classes)))
    rng = random.Random(seed)

    out_labels = array.array("H")
    for index in indices:
        label = int(labels[index])
        if distribution == "unbalanced":
            out_label = 1 if label > 0 else 0
        elif distribution == "uniform":
            out_label = rng.randrange(out_num_classes)
        elif distribution == "real":
            out_label = label * divisors + rng.randrange(divisors)
            out_label = min(out_label, out_num_classes - 1)
        else:
            raise ValueError(f"Unknown class distribution '{distribution}'")
        out_labels.append(out_label)

    if out_num_classes - 1 not in out_labels:
        out_labels[0] = out_num_classes - 1
    write_array(output_dir / "Labels.lbl", out_labels)


def expand_feature_file(input_path: Path, output_path: Path, input_dtype: str, output_dtype: str, indices):
    input_values = read_array(input_path, input_dtype)
    output_typecode = require_dtype(output_dtype)[0]
    output_values = array.array(output_typecode)
    output_values.extend(input_values[index] for index in indices)
    write_array(output_path, output_values)


def repeat_float_block(input_path: Path, output_path: Path, values_per_feature: int, out_num_features: int):
    values = read_array(input_path, "float32")
    if len(values) % values_per_feature != 0:
        raise ValueError(
            f"{input_path} contains {len(values)} float32 values, "
            f"which is not divisible by {values_per_feature}"
        )

    in_num_features = len(values) // values_per_feature
    out_values = array.array("f")
    for feature_index in range(out_num_features):
        base = (feature_index % in_num_features) * values_per_feature
        out_values.extend(values[base : base + values_per_feature])
    write_array(output_path, out_values)


def expand_dataset(args):
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detected = detect_dataset(input_dir, args.input_dtype)
    print(
        "Detected:"
        f" inSamples={detected['num_samples']},"
        f" inFeatures={detected['num_features']},"
        f" inClasses={detected['num_classes']}"
    )

    sample_indices = tile_indices(args.out_num_samples, detected["num_samples"])

    for feature_index in range(args.out_num_features):
        input_feature = detected["features"][feature_index % detected["num_features"]]
        output_feature = output_dir / f"F_{feature_index:04d}.feat"
        expand_feature_file(input_feature, output_feature, args.input_dtype, args.output_dtype, sample_indices)
        print(f"Completed: {output_feature}")

    expand_labels(
        input_dir,
        output_dir,
        sample_indices,
        args.out_num_classes,
        args.class_distribution,
        args.seed,
    )

    repeat_float_block(
        input_dir / "Threshholds.thr",
        output_dir / "Threshholds.thr",
        values_per_feature=3,
        out_num_features=args.out_num_features,
    )

    coords_path = input_dir / "Coords.co"
    if coords_path.exists():
        repeat_float_block(
            coords_path,
            output_dir / "Coords.co",
            values_per_feature=4,
            out_num_features=args.out_num_features,
        )

    result = detect_dataset(output_dir, args.output_dtype)
    print(
        "Result:"
        f" outSamples={result['num_samples']},"
        f" outFeatures={result['num_features']},"
        f" outClasses={result['num_classes']}"
    )


def inspect_dataset(args):
    detected = detect_dataset(Path(args.input_dir).resolve(), args.input_dtype)
    print(
        "Detected:"
        f" inSamples={detected['num_samples']},"
        f" inFeatures={detected['num_features']},"
        f" inClasses={detected['num_classes']}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Inspect or expand a Tea dataset without MATLAB.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print detected dataset dimensions.")
    inspect_parser.add_argument("input_dir", help="Directory containing Labels.lbl and F_XXXX.feat files.")
    inspect_parser.add_argument("input_dtype", choices=sorted(name for name in DTYPES if name != "uint16"))
    inspect_parser.set_defaults(func=inspect_dataset)

    expand_parser = subparsers.add_parser("expand", help="Write a derived Tea dataset.")
    expand_parser.add_argument("input_dir", help="Source Tea dataset directory.")
    expand_parser.add_argument("input_dtype", choices=sorted(name for name in DTYPES if name != "uint16"))
    expand_parser.add_argument("output_dir", help="Output directory for the derived dataset.")
    expand_parser.add_argument("out_num_samples", type=int, help="Number of output samples.")
    expand_parser.add_argument("out_num_features", type=int, help="Number of output feature files.")
    expand_parser.add_argument("out_num_classes", type=int, help="Number of output classes.")
    expand_parser.add_argument("output_dtype", choices=sorted(name for name in DTYPES if name != "uint16"))
    expand_parser.add_argument(
        "--class-distribution",
        choices=("real", "uniform", "unbalanced"),
        default="real",
        help="How to derive output labels. Default: real",
    )
    expand_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for synthetic label remapping. Default: 0",
    )
    expand_parser.set_defaults(func=expand_dataset)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
