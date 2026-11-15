#!/usr/bin/env python3
"""Prepare a local, public-safe Tea forest bundle; never upload or infer Git HEAD.

Reviewed metadata JSON has exactly:
  selection_metric: "macro-fingertip-f1-at-15mm"
  selected_on: "validation"
  test_used_for_selection: false
  probe_size_pixel_mm: positive number
  training_frame_fraction: number in (0,1]
  trees: one object per model with nonnegative coordinate_seed/sampling_seed/
         frame_seed, positive training_samples/training_frames/features/
         split_decisions. Counts/seeds are supplied by the reviewer, not recovered
         from untrusted nested runtime provenance.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile

import numpy as np

from tea_model import CLASS_NAMES, TeaForest, file_sha256, fingertip_centres, near_component_mask, validate_manifest

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ASSET_BYTES = 512 * 1024 * 1024
MAX_TREES = 16
MAX_PIXELS = 1024 * 1024
POST_METHOD = "largest4connected-probability-weighted-centroid"


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("JSON input must be a regular file of at most1MiB")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON keys are not allowed")
            result[key] = value
        return result
    def invalid_number(value):
        raise ValueError("nonfinite JSON numbers are not allowed")
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid_number)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def integer(value, minimum=0):
    if type(value) is not int or value < minimum or value > 2**53 - 1:
        raise ValueError("reviewed counts and seeds must be bounded integers")
    return value


def number(value, minimum=0, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= minimum or (maximum is not None and value > maximum):
        raise ValueError("reviewed values must be finite numbers in the required range")
    return value


def reviewed_selection(value, trees):
    expected = {"selection_metric", "selected_on", "test_used_for_selection", "probe_size_pixel_mm", "training_frame_fraction", "trees"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("reviewed selection metadata has missing or unapproved keys")
    if value["selection_metric"] != "macro-fingertip-f1-at-15mm" or value["selected_on"] != "validation" or value["test_used_for_selection"] is not False:
        raise ValueError("selection must be the frozen validation fingertip metric; test selection is forbidden")
    number(value["probe_size_pixel_mm"])
    number(value["training_frame_fraction"], maximum=1)
    records = value["trees"]
    fields = {"coordinate_seed", "sampling_seed", "frame_seed", "training_samples", "training_frames", "features", "split_decisions"}
    if not isinstance(records, list) or len(records) != len(trees):
        raise ValueError("reviewed metadata needs one record per ordered tree")
    for record, tree in zip(records, trees):
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("reviewed tree metadata has missing or unapproved keys")
        for key, item in record.items():
            integer(item, 0 if key.endswith("seed") else 1)
        if record["features"] != len(tree.coords) or record["split_decisions"] != tree.splits:
            raise ValueError("reviewed feature/depth counts disagree with the selected model")
    return json.loads(json.dumps(value))


def source_asset(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or any(ord(c) < 32 for c in name):
        raise ValueError("asset paths must be safe relative POSIX paths")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError("absolute paths and parent traversal are forbidden")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("asset symlinks are forbidden")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("asset escapes the source directory or is not a regular file")
    return resolved


def inference_contract(document):
    """Reconstruct known semantics field by field; never copy runtime provenance."""
    validate_manifest(document)
    image, feature = document["image"], document["feature"]
    width, height = image["width"], image["height"]
    if type(width) is not int or type(height) is not int or width * height > MAX_PIXELS:
        raise ValueError("release verification is limited to one megapixel per frame")
    if len(document["trees"]) > MAX_TREES:
        raise ValueError("release is limited to sixteen trees")
    expected_names = CLASS_NAMES[:len(document["classes"])] if len(document["classes"]) <= 7 else CLASS_NAMES + ["background"]
    if document["classes"] != expected_names:
        raise ValueError("release class names must preserve the verified ordered HandNet class map")
    for item in image["intrinsics"].values():
        if type(item) not in (int,float) or not math.isfinite(item):
            raise ValueError("camera intrinsics must be finite numeric values")
    for key in ("background_label", "unknown_label"):
        integer(document[key])
    for key in ("max_diff", "maximum"):
        number(feature[key])
    if any(type(item) not in (int,float) for item in feature["missing"].values()):
        raise ValueError("missing-depth values must be numeric")
    post = document.get("postprocessing")
    if "postprocessing" in document:
        if not isinstance(post, dict):
            raise ValueError("postprocessing must be an object when declared")
        if post.get("method", POST_METHOD) != POST_METHOD or post.get("foreground") not in ("none", "near-component"):
            raise ValueError("unsupported deployment postprocessing")
        number(post.get("probability_threshold"), maximum=1)
        integer(post.get("min_pixels"), 1)
        if post["min_pixels"] > MAX_PIXELS:
            raise ValueError("minimum component size exceeds the verification bound")
        ids = post.get("fingertip_classes")
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 or i >= min(5, len(document["classes"])) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("fingertip classes must be unique ordered HandNet classes0..4")
    result = {
        "schema": document["schema"],
        "image": {"width": width, "height": height, "depth_unit": image["depth_unit"],
                  "intrinsics": {key: image["intrinsics"][key] for key in ("fx", "fy", "cx", "cy")}},
        "feature": {key: feature[key] for key in ("type", "max_diff", "maximum", "coordinate_rule", "coordinate_precision")},
        "classes": list(document["classes"]), "background_label": document["background_label"],
        "unknown_label": document["unknown_label"], "aggregation": document["aggregation"], "empty_leaf": document["empty_leaf"],
    }
    if post is not None:
        result["postprocessing"] = {"foreground": post["foreground"], "probability_threshold": post["probability_threshold"],
                                     "min_pixels": post["min_pixels"], "fingertip_classes": list(ids)}
        if "method" in post:
            result["postprocessing"]["method"] = post["method"]
    result["feature"]["missing"] = {key: feature["missing"][key] for key in ("both", "first", "second")}
    if "modeled_background_class" in document:
        if document["modeled_background_class"] is None:
            raise ValueError("omit absent modeled_background_class; native deployment does not accept null")
        result["modeled_background_class"] = document["modeled_background_class"]
    return result


def synthetic_verification(source, released):
    height, width = source.document["image"]["height"], source.document["image"]["width"]
    rows, cols = np.indices((height, width))
    probes = [
        np.where((rows + cols) % 17 == 0, 0, 300 + (rows * 7 + cols * 13) % 901).astype(np.uint16),
        np.where((rows * 3 + cols) % 19 == 0, 0, np.where((rows // 3 + cols // 3) % 2, 700, 300)).astype(np.uint16),
    ]
    checks = []
    post = source.document.get("postprocessing")
    for depth in probes:
        masks = [("all-positive", None), ("checker-centres", (rows + cols) % 2 == 0),
                 ("near-component", near_component_mask(depth)), ("empty-centres", np.zeros(depth.shape, bool))]
        for mask_name, mask in masks:
            a_labels, a_probs = source.predict(depth, mask=mask)
            b_labels, b_probs = released.predict(depth, mask=mask)
            if not np.array_equal(a_labels, b_labels) or a_probs.tobytes() != b_probs.tobytes():
                raise ValueError("released labels/probabilities differ from the private source")
            eligible = depth > 0
            if mask is not None:
                eligible &= mask
            rr, cc = np.nonzero(eligible)
            for before, after in zip(source.trees, released.trees):
                a_p, a_h = before.predict_depth(depth, rr, cc, return_heaps=True)
                b_p, b_h = after.predict_depth(depth, rr, cc, return_heaps=True)
                if a_p.tobytes() != b_p.tobytes() or not np.array_equal(a_h, b_h):
                    raise ValueError("released tree posterior/heap differs from the private source")
            checks.append({"depth_sha256": hashlib.sha256(depth.astype("<u2").tobytes()).hexdigest(), "mask": mask_name,
                           "queried_centres": len(rr), "labels_exact": True, "probabilities_exact": True, "per_tree_heaps_exact": True})
        # Absence of defaults remains absent; callers must provide class IDs.
        if post is None:
            continue
        # Exercise preserved explicit defaults as well as arbitrary masks.
        default_mask = near_component_mask(depth) if post["foreground"] == "near-component" else None
        a_l, a_p = source.predict(depth, mask=default_mask)
        b_l, b_p = released.predict(depth, mask=default_mask)
        kwargs = {"intrinsics": source.document["image"]["intrinsics"], "threshold": post["probability_threshold"], "min_pixels": post["min_pixels"]}
        a_observations = fingertip_centres(a_l, a_p, depth, **kwargs)
        b_observations = fingertip_centres(b_l, b_p, depth, **kwargs)
        if [a_observations[i] for i in post["fingertip_classes"]] != [b_observations[i] for i in post["fingertip_classes"]]:
            raise ValueError("released default observations differ from the private source")
    return checks


def prepare_release(manifest, output, training_source_revision, selection_metadata):
    """Return the new local forest.json; preserve the private source byte for byte."""
    if not isinstance(training_source_revision, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", training_source_revision):
        raise ValueError("training source revision must be an explicitly reviewed full lowercase Git SHA")
    raw_output = Path(output).expanduser()
    if os.path.lexists(raw_output):
        raise ValueError("release output already exists")
    output = raw_output.resolve()
    if os.path.lexists(output):
        raise ValueError("release output already exists")
    manifest = Path(manifest).resolve(strict=True)
    private_sha = file_sha256(manifest)
    document = read_json(manifest)
    contract = inference_contract(document)
    assets = []
    for tree in document["trees"]:
        number(tree.get("weight",1))
        assets.append((source_asset(manifest.parent, tree["model"]), source_asset(manifest.parent, tree["coords"])))
    if sum(path.stat().st_size for pair in assets for path in pair) > MAX_ASSET_BYTES:
        raise ValueError("release assets exceed the512MiB preparation bound")
    source = TeaForest(manifest)
    metadata = reviewed_selection(selection_metadata, source.trees)
    source_hashes = [(file_sha256(model), file_sha256(coords)) for model, coords in assets]
    for original, hashes in zip(document["trees"], source_hashes):
        for key, actual in zip(("model_sha256", "coords_sha256"), hashes):
            if key in original:
                claimed = original[key]
                if not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed):
                    raise ValueError("declared source asset SHA256 must be64lowercasehex digits")
                if claimed != actual:
                    raise ValueError("source asset does not match its selected manifest SHA256")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".forest-release-", dir=output.parent))
    owns_output = False
    try:
        public = dict(contract)
        public["trees"] = []
        for index, ((model, coords), hashes, original) in enumerate(zip(assets, source_hashes, document["trees"])):
            directory = stage / "assets" / f"tree{index:03d}"
            directory.mkdir(parents=True)
            for before, name in ((model, "model.tree"), (coords, "Coords.co")):
                with before.open("rb") as read, (directory / name).open("xb") as write:
                    shutil.copyfileobj(read, write, length=1024 * 1024)
            model_hash, coord_hash = file_sha256(directory / "model.tree"), file_sha256(directory / "Coords.co")
            if (model_hash, coord_hash) != hashes:
                raise ValueError("source assets changed while preparing the release")
            public["trees"].append({"model": f"assets/tree{index:03d}/model.tree", "coords": f"assets/tree{index:03d}/Coords.co",
                                    "weight": original.get("weight", 1), "model_sha256": model_hash, "coords_sha256": coord_hash})
        public["provenance"] = {
            "training_source_revision": training_source_revision,
            "reviewed_selection": metadata,
            "dataset": {"name": "HandNet", "release": "https://huggingface.co/datasets/twerdster/HandNet",
                        "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                        "credit": ["Aaron Wetzler", "Ron Slossberg", "Ron Kimmel"],
                        "paper": "Rule of Thumb: Deep Derotation for Improved Fingertip Detection, BMVC2015",
                        "paper_url": "https://arxiv.org/abs/1507.05726",
                        "changes": "Repaired Tea trainer, balanced depth-feature training, new forest inference and held-out evaluation.",
                        "code_terms": "The dataset release license does not relicense historical MATLAB code; preserve its academic-use notice."},
        }
        public_manifest = stage / "forest.json"
        write_json(public_manifest, public)
        released = TeaForest(public_manifest)
        if inference_contract(public) != contract:
            raise ValueError("release changed the inference contract")
        checks = synthetic_verification(source, released)
        if file_sha256(manifest) != private_sha or any((file_sha256(model), file_sha256(coords)) != hashes for (model, coords), hashes in zip(assets, source_hashes)):
            raise ValueError("private source changed during preparation")
        verification = {
            "schema": "tea-forest-release-verification-v1", "complete": True,
            "private_source_manifest_sha256": private_sha, "public_manifest_sha256": file_sha256(public_manifest),
            "training_source_revision": training_source_revision,
            "reviewed_selection_sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "private_source_unchanged": True, "inference_semantics_preserved": True, "synthetic_checks": checks,
            "default_observations_exact": True if "postprocessing" in public else "not-declared; caller class IDs/options remain required",
            "actual_frame_python_native_parity": "REQUIRED_BEFORE_PUBLICATION; not performed by this source-only preparer",
            "publication_status": "LOCAL_PREPARATION_ONLY; no upload performed; actual-model verification remains pending",
        }
        write_json(stage / "release-verification.json", verification)
        # mkdir is exclusive, including races/broken symlinks. Publish the
        # completion record last; only this process's newly owned output is removed on failure.
        output.mkdir()
        owns_output = True
        for name in ("assets", "forest.json", "release-verification.json"):
            os.rename(stage / name, output / name)
        return output / "forest.json"
    except BaseException:
        if owns_output:
            shutil.rmtree(output)
        raise
    finally:
        shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Selected private deployment forest.json")
    parser.add_argument("--output", required=True, help="NEW local release directory")
    parser.add_argument("--training-source-revision", required=True, help="Explicit frozen training Git SHA; never inferred from HEAD")
    parser.add_argument("--selection-metadata", required=True, help="Reviewed safe metadata JSON; see the module help schema")
    args = parser.parse_args()
    print(prepare_release(args.manifest, args.output, args.training_source_revision, read_json(args.selection_metadata)))


if __name__ == "__main__":
    main()
