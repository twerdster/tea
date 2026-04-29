#!/usr/bin/env python3

import argparse
import os
import struct
from typing import Dict, List


def write_u16_list(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack("<{}H".format(len(values)), *values))


def write_f32_list(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack("<{}f".format(len(values)), *values))


def scenario_definition(scenario: str) -> Dict[str, List[float]]:
    if scenario == "root":
        # Use 16 samples so the root split lands cleanly on the trainer's current
        # sample alignment assumptions. Feature 0 cleanly separates the two classes.
        labels = [0] * 8 + [1] * 8
        features = [
            [-1.0] * 8 + [1.0] * 8,
            [0.0, 1.0, 0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.0, 1.0, 0.0, 1.0, -1.0, 0.5, -0.5, 0.25],
            [float(i % 4) - 1.5 for i in range(16)],
            [0.25, -0.25, 0.5, -0.5] * 4,
        ]
    else:
        # Four classes arranged so feature 0 separates the left and right halves,
        # and feature 1 separates classes inside each half.
        labels = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
        features = [
            [-1.0] * 8 + [1.0] * 8,
            [-1.0] * 4 + [1.0] * 4 + [-1.0] * 4 + [1.0] * 4,
            [0.0, 0.25, 0.5, 0.75] * 4,
            [-0.75, -0.25, 0.25, 0.75] * 4,
        ]

    for feature in features:
        if len(feature) != len(labels):
            raise ValueError("Feature length does not match label count")

    # Per-feature threshold metadata: [max positive range proxy, observed min, observed max]
    thresholds = []
    for _ in features:
        thresholds.extend([10.0, -1.0, 1.0])

    return {
        "scenario": scenario,
        "labels": labels,
        "features": features,
        "thresholds": thresholds,
    }


def generate_dataset(output_dir: str, scenario: str = "root") -> Dict[str, List[float]]:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dataset = scenario_definition(scenario)
    labels = dataset["labels"]
    features = dataset["features"]
    thresholds = dataset["thresholds"]

    write_u16_list(os.path.join(output_dir, "Labels.lbl"), labels)
    write_f32_list(os.path.join(output_dir, "Threshholds.thr"), thresholds)

    for feature_id, feature in enumerate(features):
        write_f32_list(os.path.join(output_dir, "F_{:04d}.feat".format(feature_id)), feature)

    dataset["output_dir"] = output_dir
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate a tiny Tea smoke-test dataset.")
    parser.add_argument("output_dir", help="Directory to write Labels.lbl, Threshholds.thr, and F_XXXX.feat files into.")
    parser.add_argument(
        "--scenario",
        choices=("root", "depth1"),
        default="root",
        help="Dataset layout to generate. 'root' is separable at the root, 'depth1' requires a second split.",
    )
    args = parser.parse_args()

    generate_dataset(args.output_dir, args.scenario)


if __name__ == "__main__":
    main()
