#!/usr/bin/env python3
"""Bounded, durable local HandNet experiment; never restarts a failed stage.

Stages: eight pilots, validation selection, three independently sampled final
trees, validation selection, then untouched test evaluation. External monitor
handles diagnosis/recovery and publication. A GPU lease is held only while Tea
is running, so CPU conversion/evaluation does not reserve the GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from run_guarded_training import main as guarded_train, write_json
from tea_model import CLASS_NAMES, DEFAULT_INTRINSICS, TeaForest, TeaTree, export_forest_prefix


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(trees, classes, provenance=None):
    result = {"schema": "depthtrack-tea-forest-v1", "image": {
        "width": 320, "height": 240, "depth_unit": "millimetres", "intrinsics": DEFAULT_INTRINSICS},
        "feature": {"type": "F_CHAR", "max_diff": 250, "maximum": 127,
                    "coordinate_rule": "truncate-after-base-plus-offset-over-depth",
                    "coordinate_precision": "float32-complete-sum",
                    "missing": {"both": 0, "first": 127, "second": -127}},
        "classes": classes, "background_label": len(classes), "unknown_label": 255,
        "aggregation": "mean-normalized-leaf-probabilities", "empty_leaf": "root-prior",
        "trees": trees, "provenance": provenance or {}}
    if len(classes) == 8:
        result["modeled_background_class"] = 7
    return result


def validation_score(result):
    """Macro fingertip F1@15mm; all missed and spurious detections count."""
    local = result["localization"]
    def counts(value):
        array = np.asarray(value, dtype=float)
        if array.shape != (5,) or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("localization counts must be five finite nonnegative values")
        return array
    true_positive = counts(local["sensor_pos_recall_at_mm"]["15"]["hits"])
    present = counts(local["sensor_pos_present"])
    # Predictions on visible fingers with an unknown/invalid sensor position
    # are unscorable in 3D, and must not be charged as localization failures.
    scorable_detected = counts(local.get("sensor_pos_detected", local["detected"]))
    absent_false_positive = counts(local["false_positive_on_absent"])
    detections = scorable_detected + absent_false_positive
    denominator = present + detections
    valid = denominator > 0
    if not valid.any() or not np.isfinite(denominator).all() or np.any(true_positive > present) or np.any(true_positive > scorable_detected):
        raise ValueError("invalid localization selection denominators")
    values = 2 * true_positive[valid] / denominator[valid]
    if not np.isfinite(values).all() or np.any(values > 1):
        raise ValueError("invalid localization F1")
    return float(values.mean())


def verify_saved_predictions(path, predictions, total_samples):
    """Validate native Tea's uint16 sample predictions before retiring scratch."""
    path = Path(path)
    if path.stat().st_size != int(total_samples) * 2:
        raise ValueError("Tea saved prediction file must contain exactly one little-endian uint16 per sample")
    predictions = np.asarray(predictions)
    if predictions.ndim != 1 or len(predictions) > total_samples:
        raise ValueError("invalid external prediction sample count")
    saved = np.fromfile(path, dtype="<u2", count=len(predictions))
    if not np.array_equal(predictions, saved):
        raise ValueError("external traversal disagrees with Tea's saved sample predictions")


class Campaign:
    def __init__(self, root, archives=None, gpu_lock=None):
        self.root = root.resolve()
        self.repo, self.run = self.root / "repo", self.root / "runs/current"
        self.python = sys.executable
        self.archives = Path(archives) if archives else Path.home() / "data/HandNet/OpenDataLab___HandNet/raw/HandNet"
        self.gpu_lock = Path(gpu_lock) if gpu_lock else self.run / "gpu.lock"
        self.state_path = self.run / "CAMPAIGN.json"
        if self.state_path.exists():
            raise RuntimeError("Campaign state exists; inspect before explicit recovery, do not duplicate jobs")
        self.state = {"schema_version": 1, "phase": "starting", "pid": os.getpid(),
                      "started_at": time.time(), "pilots": [], "final_trees": [], "evaluations": []}
        self.budget = self.run / "BUDGET.json"
        if not self.budget.exists():
            write_json(self.budget, {"max_active_training_seconds": 172800,
                                    "used_active_training_seconds": 0,
                                    "max_pilot_configs": 8, "max_final_trees": 3, "runs": []})

    def update(self, phase, **extra):
        self.state.update(phase=phase, updated_at=time.time(), **extra)
        write_json(self.state_path, self.state)

    def command(self, argv, log, max_seconds):
        with Path(log).open("x") as output:
            subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT,
                           check=True, timeout=max_seconds,
                           env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))

    def train(self, dataset, name, pilot):
        info = json.loads((dataset / "handnet_metadata.json").read_text())
        if not info.get("complete") or info["samples"] <= 0 or info["features"] != 1200:
            raise ValueError("incomplete/unexpected dataset")
        directory = self.run / name
        directory.mkdir()
        spec_path = directory / "spec.json"
        binary = self.repo / "Tea"
        spec = {"argv": [str(binary), "1200", "8", "19", "12", "F_CHAR", str(info["samples"]),
                         "16", "0", "0", "W_ONES", str(dataset) + "/", name, "0", "LOG0", name],
                "cwd": str(self.repo), "output_dir": str(directory / "guard"), "checkpoint_dir": str(dataset),
                "max_seconds": 1800 if pilot else 57600, "budget_file": str(self.budget),
                "gpu_lock": str(self.gpu_lock), "min_free_vram_mb": 1536,
                "max_idle_utilization": 30, "binary_sha256": sha256(binary),
                "tea_log": str(dataset / f"LogFile_{name}_0000.log"),
                "dataset_metadata_sha256": sha256(dataset / "handnet_metadata.json")}
        write_json(spec_path, spec)
        self.update("pilot_resource_wait" if pilot else "final_resource_wait", current_run=name,
                    current_spec=str(spec_path), current_tea_log=spec["tea_log"])
        guarded_train(spec_path)
        tree = TeaTree(dataset / "Tree_0000.tree", dataset / "Coords.co")
        if tree.splits != 20 or tree.classes != len(info["classes"]):
            raise ValueError("trained model depth/classes mismatch")
        # Recompute external traversal from original feature columns, independently
        # of Tea's sample routing and self-success report.
        count = min(info["samples"], 1024)
        features = np.stack([np.fromfile(dataset / f"F_{i:04d}.feat", dtype=np.int8, count=count)
                             for i in range(1200)], axis=1)
        predictions = tree.predict_features(features).argmax(axis=1)
        verify_saved_predictions(dataset / "Tree_0000.tree.labels", predictions, info["samples"])
        write_json(directory / "model-validation.json", {"depth_decisions": tree.splits, "classes": tree.classes,
                   "external_prediction_samples": count, "external_predictions_match": True,
                   "tree_sha256": sha256(tree.path), "coords_sha256": sha256(tree.coords_path)})
        forest_path = dataset / "forest.json"
        write_json(forest_path, manifest([{"model": "Tree_0000.tree", "coords": "Coords.co", "weight": 1}], info["classes"],
                                        {"run": name, "training_samples": info["samples"], "seed": info["seed"]}))
        return info, forest_path

    def evaluate(self, forest_path, label, splits, threshold, training, final_test=False, all_frames=False):
        result = self.run / f"{label}.json"
        archive = self.root / ("data/test/TestData" if final_test else "data/validation/ValidationData")
        document = json.loads(forest_path.read_text())
        foreground = "none" if len(document["classes"]) == 8 else "near-component"
        argv = [self.python, "-u", str(self.repo / "scripts/tea_model.py"), "evaluate",
                "--manifest", str(forest_path), "--archive", str(archive), "--output", str(result),
                "--split", "test" if final_test else "validation", "--max-frames", "0" if all_frames else "300",
                "--frame-seed", "420", "--max-splits", str(splits), "--localize", "--foreground", foreground,
                "--probability-threshold", str(threshold)]
        if not final_test and not all_frames:
            argv += ["--frame-manifest", str(self.run / "validation-selection/frames.json")]
        for path in training:
            argv += ["--training-manifest", str(path)]
        self.update("test_evaluation" if final_test else "validation_evaluation", current_evaluation=label)
        self.command(argv, self.run / f"{label}.log", 21600)
        value = json.loads(result.read_text())
        score = validation_score(value)
        item = {"label": label, "result": str(result), "manifest": str(forest_path),
                "splits": splits, "threshold": threshold, "score_macro_f1_at_15mm": score,
                "split": "test" if final_test else "validation", "all_frames": all_frames}
        self.state["evaluations"].append(item)
        self.update(self.state["phase"])
        return item

    def grid(self, path, label, training):
        choices = []
        for depth in [12, 16, 20]:
            for threshold in [.25, .5]:
                choices.append(self.evaluate(path, f"{label}-d{depth}-p{threshold}", depth, threshold, training))
        return max(choices, key=lambda x: (x["score_macro_f1_at_15mm"], -x["splits"], x["threshold"]))

    def execute(self):
        self.update("waiting_for_prepared_datasets")
        write_json(self.run / "runtime.json", {"python": sys.version, "numpy": np.__version__,
                   "platform": platform.platform(), "binary_sha256": sha256(self.repo / "Tea"),
                   "source_sha256": {str(p.relative_to(self.repo)): sha256(p)
                                     for p in sorted(self.repo.glob("*.cu")) + sorted(self.repo.glob("*.h")) +
                                     sorted((self.repo / "scripts").glob("*.py"))}})
        for background in [False, True]:
            for scale in [50, 5000, 25000, 50000]:
                name = f"pilot-{'bg-' if background else ''}scale-{scale}"
                dataset = self.root / "data" / name
                while not (dataset / "handnet_metadata.json").exists():
                    self.update("waiting_for_prepared_datasets", current_dataset=str(dataset))
                    time.sleep(30)
                info, path = self.train(dataset, name, True)
                selected = self.grid(path, name, [dataset / "frames.json"])
                self.state["pilots"].append({"name": name, "scale": scale, "include_background": background,
                                             "samples": info["samples"], "selected": selected})
                self.update("pilot_complete")
        chosen = max(self.state["pilots"], key=lambda x: x["selected"]["score_macro_f1_at_15mm"])
        write_json(self.run / "pilot-selection.json", {"chosen": chosen, "candidates": self.state["pilots"],
                   "selection_metric": "macro fingertipF1@15mm; all missed and spurious detections included; validation only"})
        self.update("pilot_selection_frozen", chosen=chosen)

        training_manifests = []
        final_root = self.root / "models/final"
        final_root.mkdir(parents=True)
        trees, classes = [], None
        for number, seed in enumerate([201, 202, 203]):
            name = f"final-tree-{number}"
            dataset = self.root / "data" / name
            # Require a substantial reserve; never fill the shared system disk.
            upper_bound = int(202198 * .9) * 512 * (1200 + 10)
            if shutil.disk_usage(self.root).free < upper_bound + 16 * 1024**3:
                raise RuntimeError("Insufficient task-scoped staging space; use verified AgentSSD and reassess, not a large internal fallback")
            argv = [self.python, "-u", str(self.repo / "scripts/handnet_to_tea.py"), "--archive",
                    str(self.root / "data/train/TrainData"), "--output", str(dataset), "--features", "1200",
                    "--probe-size", str(chosen["scale"]), "--seed", str(seed), "--sampling-seed", str(seed),
                    "--frame-seed", str(seed), "--frame-fraction", ".9", "--sampling", "balanced",
                    "--samples-per-frame", "512", "--sample-alignment", "1", "--split", "train", "--max-output-gib", "128"]
            if chosen["include_background"]:
                argv.append("--include-background")
            self.update("final_conversion", current_run=name, current_command=argv)
            self.command(argv, self.run / f"{name}-conversion.log", 21600)
            info, path = self.train(dataset, name, False)
            target = final_root / f"tree{number}"
            target.mkdir()
            for source, dest in [("Tree_0000.tree", "model.tree"), ("Coords.co", "Coords.co"),
                                 ("handnet_metadata.json", "training.json"), ("frames.json", "frames.json")]:
                shutil.copy2(dataset / source, target / dest)
            training_manifests.append(target / "frames.json")
            trees.append({"model": f"tree{number}/model.tree", "coords": f"tree{number}/Coords.co", "weight": 1,
                          "model_sha256": sha256(target / "model.tree"), "coords_sha256": sha256(target / "Coords.co")})
            classes = info["classes"]
            self.state["final_trees"].append({"name": name, "seed": seed, "samples": info["samples"], "model": str(target / "model.tree")})
            # These are reproducible, task-generated scratch columns. Retain exact
            # column hashes and all frame/pixel/coordinate manifests before freeing
            # space for the next independent bank. Never touch source archives.
            self.update("hashing_final_feature_bank", current_run=name)
            hashes = {p.name: sha256(p) for p in sorted(dataset.glob("F_*.feat"))}
            write_json(target / "feature-column-sha256.json", hashes)
            for p in dataset.glob("F_*.feat"):
                p.unlink()
            write_json(dataset / "SCRATCH_COLUMNS_RETIRED.json", {"at": time.time(), "reason": "sequential bounded disk staging",
                       "retained_hashes": str(target / "feature-column-sha256.json"), "retained_inputs": str(dataset / "handnet_metadata.json")})
            self.update("final_tree_complete")

        ensemble = final_root / "forest.json"
        write_json(ensemble, manifest(trees, classes, {"training": "three independent90percent frame subsets and feature/pixel seeds201,202,203",
                   "pilot_selection": chosen, "depth_decisions": 20, "test_used_for_selection": False}))
        selected_ensemble = self.grid(ensemble, "final-ensemble-selection", training_manifests)
        single_choices = []
        for number, tree in enumerate(trees):
            path = final_root / f"single{number}.json"
            write_json(path, manifest([tree], classes, {"tree": number}))
            single_choices.append(self.grid(path, f"final-single{number}-selection", training_manifests))
        selected_single = max(single_choices, key=lambda x: x["score_macro_f1_at_15mm"])
        selection = {"ensemble": selected_ensemble, "single": selected_single, "test_used_for_selection": False,
                     "selected_at": time.time(), "selection_metric": "macro fingertipF1@15mm; validation only"}
        write_json(self.run / "FINAL_SELECTION.json", selection)
        self.update("final_selection_frozen", final_selection=selection)
        for kind, selected in [("ensemble", selected_ensemble), ("single", selected_single)]:
            self.update("export_selected_model", current_export=kind)
            deployed = export_forest_prefix(Path(selected["manifest"]), self.root / "models/deploy" / kind, selected["splits"])
            document = json.loads(Path(deployed).read_text())
            document["postprocessing"] = {"foreground": "none" if len(classes) == 8 else "near-component",
                                          "probability_threshold": selected["threshold"], "min_pixels": 3,
                                          "fingertip_classes": [0, 1, 2, 3, 4],
                                          "method": "largest4connected-probability-weighted-centroid"}
            write_json(Path(deployed), document)
            selected["deployed_manifest"] = str(deployed)
        write_json(self.run / "FINAL_SELECTION.json", selection)
        for kind, selected in [("ensemble", selected_ensemble), ("single", selected_single)]:
            self.evaluate(Path(selected["manifest"]), f"final-{kind}-validation-all", selected["splits"], selected["threshold"], training_manifests, all_frames=True)
        # Only now open/extract the test archive. Same frozen selections are used.
        self.command([self.python, str(self.repo / "scripts/extract_handnet_archive.py"),
                      str(self.archives / "TestData.zip"), str(self.root / "data/test")],
                     self.run / "test-extraction.log", 7200)
        for kind, selected in [("ensemble", selected_ensemble), ("single", selected_single)]:
            self.evaluate(Path(selected["manifest"]), f"final-{kind}-test-all", selected["splits"], selected["threshold"], training_manifests, final_test=True, all_frames=True)
        self.update("training_evaluation_complete_pending_integration", completed_at=time.time())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--archives", type=Path, required=True, help="Directory containing original distribution archives")
    parser.add_argument("--gpu-lock", type=Path, required=True, help="Existing cooperative GPU flock file; never modified")
    args = parser.parse_args()
    campaign = Campaign(args.root, args.archives, args.gpu_lock)
    try:
        campaign.execute()
    except BaseException as error:
        campaign.update("failed", error=repr(error))
        raise
