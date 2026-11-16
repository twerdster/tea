#!/usr/bin/env python3
"""Native Tea forest reference and held-out HandNet evaluation (NumPy/SciPy).

This supports newly repaired Tea's 0-based heap histograms. Legacy recovered
models with shifted/anomalous histograms require their historical evaluator.
"""
import argparse
import hashlib
import io
import json
import math
import os
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import ndimage

from handnet_to_tea import CLASS_NAMES, frame_identity, infer_split, load_coords, open_source, read_frame_manifest

NODE_DTYPE = np.dtype([("feature", "<i4"), ("threshold", "<f4")])
LEAF_DTYPE = np.dtype({"names": ["maximum", "label"], "formats": ["<u4", "<u2"], "offsets": [0, 4], "itemsize": 8})
SCHEMA = "depthtrack-tea-forest-v1"
DEFAULT_INTRINSICS = {"fx": 240.30773969, "fy": 240.07488538, "cx": 178.29264758, "cy": 117.95811188}


def feature_values(depth, rows, cols, coordinate_records, maximum=127., max_diff=250.):
    """One separately selected coordinate record per queried pixel, float32 MEX rules."""
    depth = np.asarray(depth, dtype=np.float32)
    records = np.asarray(coordinate_records, dtype=np.float32)
    z = depth[rows, cols]
    if records.shape != (len(rows), 4) or np.any(z <= 0):
        raise ValueError("one coordinate record and positive centre depth required per pixel")
    destinations = []
    for component, base in enumerate((rows, cols, rows, cols)):
        destinations.append((base.astype(np.float32) + records[:, component] / z).astype(np.int64))
    neighbors = []
    for rr, cc in ((destinations[0], destinations[1]), (destinations[2], destinations[3])):
        valid = (rr >= 0) & (rr < depth.shape[0]) & (cc >= 0) & (cc < depth.shape[1])
        values = np.zeros(len(rows), dtype=np.float32)
        values[valid] = depth[rr[valid], cc[valid]]
        neighbors.append(values)
    d1, d2 = neighbors
    values = np.zeros(len(rows), dtype=np.float32)
    values[(d1 == 0) & (d2 != 0)] = maximum
    values[(d1 != 0) & (d2 == 0)] = -maximum
    both = (d1 != 0) & (d2 != 0)
    values[both] = np.ceil(np.clip((d1[both] - d2[both]) / np.float32(max_diff) * np.float32(maximum), -maximum, maximum))
    return values


class TeaTree:
    def __init__(self, model, coords):
        self.path, self.coords_path = Path(model), Path(coords)
        with self.path.open("rb") as handle:
            header = handle.read(16)
        if len(header) != 16:
            raise ValueError("incomplete Tea tree header")
        self.depth, self.classes, self.node_count, self.hist_count = struct.unpack("<4i", header)
        if not 0 <= self.depth <= 25 or not 1 <= self.classes <= 16384:
            raise ValueError("invalid Tea depth or class count")
        if self.node_count != (1 << (self.depth + 1)) - 1 or self.hist_count != 2 * self.node_count + 1:
            raise ValueError("Tea tree heap/header counts inconsistent")
        expected = 16 + self.node_count * 8 + self.hist_count * (8 + self.classes)
        if self.path.stat().st_size != expected:
            raise ValueError(f"Tea model size mismatch: expected{expected}, actual{self.path.stat().st_size}")
        self.nodes = np.memmap(self.path, dtype=NODE_DTYPE, mode="r", offset=16, shape=(self.node_count,))
        leaf_offset = 16 + self.node_count * 8
        self.leaves = np.memmap(self.path, dtype=LEAF_DTYPE, mode="r", offset=leaf_offset, shape=(self.hist_count,))
        self.histograms = np.memmap(self.path, dtype=np.uint8, mode="r", offset=leaf_offset + self.hist_count * 8, shape=(self.hist_count, self.classes))
        self.coords = load_coords(self.coords_path).T.copy()
        if np.any(self.nodes["feature"] < 0) or np.any(self.nodes["feature"] >= len(self.coords)) or not np.isfinite(self.nodes["threshold"]).all():
            raise ValueError("Tea nodes reference invalid features or nonfinite thresholds")
        if np.any(self.leaves["label"] >= self.classes):
            raise ValueError("Tea leaf labels are out of range")
        self.root_prior = self.histograms[0].astype(np.float32)
        if not self.root_prior.sum():
            raise ValueError("empty root histogram: malformed or legacy-shifted Tea model")
        self.root_prior /= self.root_prior.sum()

    @property
    def splits(self):
        return self.depth + 1

    def reached_probabilities(self, heaps):
        probabilities = self.histograms[heaps].astype(np.float32)
        sums = probabilities.sum(axis=1, keepdims=True)
        missing = sums[:, 0] == 0
        probabilities[~missing] /= sums[~missing]
        probabilities[missing] = self.root_prior
        return probabilities

    def write_prefix(self, output, splits):
        """Export a self-contained native tree whose natural depth is the selection.

        Preserve complete-node/histogram prefixes; sourcefullmodel staysunchanged.
        """
        output = Path(output).resolve()
        if not 1 <= splits <= self.splits:
            raise ValueError("exported prefix requires1..modelsplit decisions")
        if output.exists():
            raise ValueError("prefix output exists; preserveimmutablemodelartifacts")
        output.parent.mkdir(parents=True,exist_ok=True)
        count = (1 << splits)-1
        hists = 2*count+1
        fd,name = tempfile.mkstemp(prefix=output.name+".",suffix=".tmp",dir=output.parent)
        try:
            with os.fdopen(fd,"wb") as handle:
                handle.write(struct.pack("<4i",splits-1,self.classes,count,hists))
                self.nodes[:count].tofile(handle)
                self.leaves[:hists].tofile(handle)
                self.histograms[:hists].tofile(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name,output)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
        return TeaTree(output,self.coords_path)

    def predict_features(self, features, max_splits=None, return_heaps=False):
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] < len(self.coords):
            raise ValueError("feature matrix does not cover the model bank")
        heaps = np.zeros(len(features), dtype=np.int64)
        for _ in range(self._split_limit(max_splits)):
            nodes = self.nodes[heaps]
            values = features[np.arange(len(features)), nodes["feature"]]
            heaps = heaps * 2 + 1 + (values > nodes["threshold"])
        probabilities = self.reached_probabilities(heaps)
        return (probabilities, heaps) if return_heaps else probabilities

    def _split_limit(self, max_splits):
        if max_splits is not None and not 0 <= max_splits <= self.splits:
            raise ValueError("requested split count exceeds model or is negative")
        return self.splits if max_splits is None else max_splits

    def predict_depth(self, depth, rows, cols, max_splits=None, return_heaps=False):
        heaps = np.zeros(len(rows), dtype=np.int64)
        for _ in range(self._split_limit(max_splits)):
            nodes = self.nodes[heaps]
            values = feature_values(depth, rows, cols, self.coords[nodes["feature"]])
            heaps = heaps * 2 + 1 + (values > nodes["threshold"])
        probabilities = self.reached_probabilities(heaps)
        return (probabilities, heaps) if return_heaps else probabilities


def validate_manifest(document):
    if not isinstance(document,dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema must be{SCHEMA}")
    image = document.get("image", {})
    if not isinstance(image,dict):
        raise ValueError("image must be an object")
    if not isinstance(image.get("width"), int) or image["width"] <= 0 or not isinstance(image.get("height"), int) or image["height"] <= 0 or image.get("depth_unit") != "millimetres":
        raise ValueError("manifest image needs positive integer width/height and millimetre depths")
    intrinsics = image.get("intrinsics", {})
    if not isinstance(intrinsics,dict):
        raise ValueError("intrinsics must be an object")
    if set(intrinsics) != {"fx", "fy", "cx", "cy"} or any(not isinstance(v, (int,float)) or not math.isfinite(v) for v in intrinsics.values()) or intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
        raise ValueError("manifest intrinsics are incomplete or invalid")
    feature = document.get("feature", {})
    if not isinstance(feature,dict):
        raise ValueError("feature profile must be an object")
    required = {"type": "F_CHAR", "max_diff": 250, "maximum": 127,
                "coordinate_rule": "truncate-after-base-plus-offset-over-depth", "coordinate_precision": "float32-complete-sum",
                "missing": {"both": 0, "first": 127, "second": -127}}
    for key, value in required.items():
        if feature.get(key) != value:
            raise ValueError(f"unsupported or incomplete feature profile:{key}")
    classes = document.get("classes")
    if not isinstance(classes, list) or not classes or len(classes)>254 or any(not isinstance(v,str) or not v for v in classes) or len(set(classes)) != len(classes):
        raise ValueError("manifest classes must be a unique nonempty name list")
    if document.get("background_label") != len(classes) or document.get("unknown_label") != 255:
        raise ValueError("background label must be immediately outside modeled classes; unknown255")
    if "modeled_background_class" in document and document["modeled_background_class"] is not None:
        if len(classes) != 8 or document["modeled_background_class"] != 7 or classes[7] != "background":
            raise ValueError("HandNet modeled background must be class7 of eightclasses")
    if document.get("aggregation") != "mean-normalized-leaf-probabilities" or document.get("empty_leaf") != "root-prior":
        raise ValueError("unsupported forest aggregation or empty-leaf rule")
    trees = document.get("trees")
    if not isinstance(trees,list) or not trees:
        raise ValueError("manifest requires at least one tree")
    for tree in trees:
        if not isinstance(tree,dict):
            raise ValueError("tree entries must be objects")
        if not all(isinstance(tree.get(k),str) and tree[k] for k in ("model", "coords")):
            raise ValueError("tree entry requires model and coords paths")
        weight = tree.get("weight", 1.)
        if not isinstance(weight, (int,float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("tree weight must be finite and positive")
    return document


class TeaForest:
    def __init__(self, manifest):
        self.manifest_path = Path(manifest).resolve()
        self.document = validate_manifest(json.loads(self.manifest_path.read_text()))
        root = self.manifest_path.parent
        self.trees = [TeaTree(root/tree["model"], root/tree["coords"]) for tree in self.document["trees"]]
        self.classes = len(self.document["classes"])
        if any(t.classes != self.classes for t in self.trees):
            raise ValueError("tree and manifest class counts differ")
        self.weights = np.asarray([t.get("weight", 1.) for t in self.document["trees"]], dtype=np.float64)
        self.total_weight = 0.
        for weight in self.weights:
            self.total_weight += float(weight)
        if not math.isfinite(self.total_weight):
            raise ValueError("tree weights overflow")

    def predict(self, depth, rows=None, cols=None, max_splits=None, chunk_size=32768, mask=None):
        depth = np.asarray(depth, dtype=np.float32)
        expected = (self.document["image"]["height"], self.document["image"]["width"])
        if depth.shape != expected or not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError(f"expected finite nonnegative depth shape{expected}")
        full = rows is None and cols is None
        if full:
            rows, cols = np.indices(depth.shape)
            rows, cols = rows.ravel(), cols.ravel()
        if rows is None or cols is None or rows.shape != cols.shape or chunk_size < 1:
            raise ValueError("matching sample rows/cols and positive chunk size required")
        labels = np.full(len(rows), self.document["background_label"], dtype=np.uint8)
        probabilities = np.zeros((len(rows), self.classes), dtype=np.float32)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != depth.shape:
                raise ValueError("foreground mask dimensions differ from raw depth")
        eligible = depth[rows, cols] > 0
        if mask is not None:
            eligible &= mask[rows, cols]
        valid = np.flatnonzero(eligible)
        for start in range(0, len(valid), chunk_size):
            selected = valid[start:start+chunk_size]
            accumulated = np.zeros((len(selected), self.classes), dtype=np.float64)
            for tree, weight in zip(self.trees, self.weights):
                accumulated += weight * tree.predict_depth(depth, rows[selected], cols[selected], max_splits)
            probabilities[selected] = (accumulated/self.total_weight).astype(np.float32)
            labels[selected] = probabilities[selected].argmax(axis=1).astype(np.uint8)
        if full:
            return labels.reshape(depth.shape), probabilities.reshape((*depth.shape, self.classes))
        return labels, probabilities


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def export_forest_prefix(manifest, output, splits, verification_depth=None):
    """Immutable deployable prefix with exact raw-depth label/probability/heap QA.

    Full source files are retained. Output paths are portable and relative.
    Optional verification_depth is an actual raw frame; otherwise an analytic
    frame tests format/routing without making an accuracy claim.
    """
    source=TeaForest(manifest)
    if not 1 <= splits <= min(t.splits for t in source.trees):
        raise ValueError("export split count exceeds a source tree or is below1")
    output=Path(output).resolve()
    if output.exists():
        raise ValueError("deployment output exists; preserveimmutableartifacts")
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=output.name+".",dir=output.parent))
    try:
        document=json.loads(json.dumps(source.document))
        deployed_trees=[]; provenance=[]
        for i,(tree,entry) in enumerate(zip(source.trees,source.document["trees"])):
            directory=stage/f"tree{i}";directory.mkdir()
            cropped=tree.write_prefix(directory/"model.tree",splits)
            shutil.copy2(tree.coords_path,directory/"Coords.co")
            deployed=dict(entry,model=f"tree{i}/model.tree",coords=f"tree{i}/Coords.co",
                          model_sha256=file_sha256(cropped.path),coords_sha256=file_sha256(directory/"Coords.co"))
            deployed_trees.append(deployed)
            provenance.append({"source_model":entry["model"],"source_model_sha256":file_sha256(tree.path),
                               "source_splits":tree.splits,"cropped_model_sha256":deployed["model_sha256"],"selected_splits":splits})
        document["trees"]=deployed_trees
        document.setdefault("provenance",{})["deployment_prefix_export"]={"source_manifest_sha256":file_sha256(manifest),"models":provenance}
        manifest_path=stage/"forest.json"
        manifest_path.write_text(json.dumps(document,indent=2)+"\n")
        cropped_forest=TeaForest(manifest_path)
        height,width=source.document["image"]["height"],source.document["image"]["width"]
        if verification_depth is None:
            rr,cc=np.indices((height,width))
            depth=(300+(rr*7+cc*13)%601).astype(np.uint16)
            depth[(rr+cc)%17==0]=0
            verification_kind="analyticformat/routingprobe; notaccuracydata"
        else:
            depth=np.asarray(verification_depth)
            verification_kind="providedrawdepthframe"
        old_labels,old_probabilities=source.predict(depth,max_splits=splits)
        new_labels,new_probabilities=cropped_forest.predict(depth)
        if not np.array_equal(old_labels,new_labels) or old_probabilities.tobytes()!=new_probabilities.tobytes():
            raise ValueError("croppedforest raw-depth predictions differ from selectedfullprefix")
        rows,cols=np.nonzero(depth>0)
        for original,cropped in zip(source.trees,cropped_forest.trees):
            old_probability,old_heap=original.predict_depth(depth,rows,cols,max_splits=splits,return_heaps=True)
            new_probability,new_heap=cropped.predict_depth(depth,rows,cols,return_heaps=True)
            if old_probability.tobytes()!=new_probability.tobytes() or not np.array_equal(old_heap,new_heap):
                raise ValueError("croppedtree raw-depth heap/probabilities differ")
        verification={"selected_splits":splits,"trees":len(source.trees),"pixels":int(depth.size),"positive_pixels":len(rows),
                      "labels_exact":True,"probabilities_exact":True,"per_tree_heaps_exact":True,"kind":verification_kind,
                      "verification_depth_sha256":hashlib.sha256(np.asarray(depth,dtype="<u2").tobytes()).hexdigest(),
                      "source_manifest_sha256":file_sha256(manifest),"deployment_manifest_sha256":file_sha256(manifest_path)}
        (stage/"prefix-verification.json").write_text(json.dumps(verification,indent=2)+"\n")
        os.replace(stage,output)
        return output/"forest.json"
    except BaseException:
        shutil.rmtree(stage)
        raise


def near_component_mask(depth):
    """Frozen independent foregroundv1; mask query centres, preserve raw neighbours.

    Valid100..1500mm, orderstat floor(.02*(N-1)), near+150mm, largest4connected
    component >=30pixels; equal areas choose first row-major component.
    """
    depth = np.asarray(depth)
    valid = (depth >= 100) & (depth <= 1500)
    values = depth[valid]
    if not len(values):
        return np.zeros(depth.shape,dtype=bool)
    index = (len(values)-1)//50
    near = np.partition(values,index)[index]
    candidate = valid & (depth <= int(near)+150)
    components,count = ndimage.label(candidate)
    if not count:
        return np.zeros(depth.shape,dtype=bool)
    areas = np.bincount(components.ravel()); areas[0]=0
    chosen = int(areas.argmax())
    if areas[chosen] < 30:
        return np.zeros(depth.shape,dtype=bool)
    return components == chosen


def confusion_matrix(truth, prediction, classes=7):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or np.any(truth < 0) or np.any(truth >= classes) or np.any(prediction < 0) or np.any(prediction >= classes):
        raise ValueError("confusion labels outside declared classes")
    return np.bincount((truth.astype(np.int64)*classes + prediction).ravel(), minlength=classes**2).reshape(classes, classes)


def classification_metrics(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    correct = np.diag(confusion)
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    union = support + predicted - correct
    recall = np.divide(correct, support, out=np.full(len(correct), np.nan), where=support > 0)
    precision = np.divide(correct, predicted, out=np.full(len(correct), np.nan), where=predicted > 0)
    iou = np.divide(correct, union, out=np.full(len(correct), np.nan), where=union > 0)
    def nullable(values):
        return [float(v) if np.isfinite(v) else None for v in values]
    return {"confusion_truth_rows": confusion.tolist(), "support": support.tolist(), "predicted": predicted.tolist(),
            "accuracy": float(correct.sum()/support.sum()) if support.sum() else None,
            "macro_iou": float(np.nanmean(iou)) if np.any(union) else None,
            "macro_recall": float(np.nanmean(recall)) if np.any(support) else None,
            "class_iou": nullable(iou), "class_recall": nullable(recall), "class_precision": nullable(precision)}


def fingertip_centres(labels, probabilities, depth, intrinsics, threshold=.5, min_pixels=3):
    """Largest connected confident component, probability-weighted centroid.

    This is explicitly our detector postprocessing, not a reproduced paper score.
    Returns zero-based x/y pixels and pinhole x/y/z millimetres per class0..4.
    """
    centers = []
    for class_id in range(min(5, probabilities.shape[2])):
        mask = (labels == class_id) & (probabilities[:, :, class_id] >= threshold) & (depth > 0)
        components, count = ndimage.label(mask)
        if count == 0:
            centers.append(None)
            continue
        areas = np.bincount(components.ravel()); areas[0] = 0
        component = int(areas.argmax())
        if areas[component] < min_pixels:
            centers.append(None)
            continue
        rows, cols = np.nonzero(components == component)
        weights = probabilities[rows, cols, class_id].astype(np.float64)
        weights /= weights.sum()
        x, y, z = float(weights @ cols), float(weights @ rows), float(weights @ depth[rows,cols])
        xyz = [(x-intrinsics["cx"])*z/intrinsics["fx"], (y-intrinsics["cy"])*z/intrinsics["fy"], z]
        centers.append({"xy": [x,y], "xyz": xyz, "pixels": int(len(rows))})
    return centers


def summary_errors(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.quantile(values,.9))}


def evaluate(args):
    forest = TeaForest(args.manifest)
    if forest.classes not in (7,8):
        raise ValueError("HandNet evaluator requires seven handclasses, optionallyplusbackground")
    if forest.classes == 8 and forest.document.get("modeled_background_class") != 7:
        raise ValueError("eightclass HandNet manifest requires modeled_background_class7")
    source = open_source(Path(args.archive).resolve())
    try:
        split = args.split or infer_split(args.archive)
        names = source.names()
        if args.frame_manifest:
            selection = read_frame_manifest(args.frame_manifest, set(names), args.split)
            names = [f["name"] for f in selection["frames"]]
            split = selection["split"]
        if args.max_frames and len(names) > args.max_frames:
            indices = np.random.default_rng(args.frame_seed).choice(len(names), args.max_frames, replace=False)
            names = [names[i] for i in sorted(indices.tolist())]
        training_ids = set()
        for path in args.training_manifest:
            document = read_frame_manifest(path, expected_split="train")
            training_ids.update(frame["frame_id"] for frame in document["frames"])
        if split == "train" and not args.allow_training_evaluation:
            raise ValueError("held-out evaluator refuses training split unless explicitlymarked --allow-training-evaluation")
        classes = forest.classes
        confusion = np.zeros((classes+1,classes+1),dtype=np.int64)
        oracle_confusion = np.zeros((classes,classes),dtype=np.int64)
        frame_records,label_errors,sensor_errors_px,sensor_errors_mm,correspondence = [],[],[],[],[]
        present,detected,false_positive,absent = np.zeros(5,int),np.zeros(5,int),np.zeros(5,int),np.zeros(5,int)
        sensor_present = np.zeros(5,int)
        sensor_detected = np.zeros(5,int)
        hits_mm = {t:np.zeros(5,int) for t in (10,15,20)}
        hits_px = {t:np.zeros(5,int) for t in (5,10,15)}
        roi_totals = {"true_positive":0,"false_positive":0,"false_negative":0,"true_negative":0}
        learned_foreground_totals = dict(roi_totals)
        intrinsics = forest.document["image"]["intrinsics"]
        for index,name in enumerate(names):
            mat = sio.loadmat(io.BytesIO(source.read(name)),variable_names=["depth","lbl","pos","session","idx","gidx","sID"])
            record = frame_identity(mat,name,split)
            if record["frame_id"] in training_ids:
                raise ValueError(f"training/evaluation frame leakage:{record['frame_id']}")
            depth,truth = np.asarray(mat["depth"],dtype=np.float32),np.asarray(mat["lbl"])
            if depth.shape != truth.shape:
                raise ValueError(f"depth/label shape mismatch:{name}")
            true_hand = (truth >= 1) & (truth <= 7) & (depth > 0)
            if args.foreground == "oracle":
                roi = true_hand
            elif args.foreground == "near-component":
                roi = near_component_mask(depth)
            else:
                roi = depth > 0
            positive = depth > 0
            for key,mask in (("true_positive",roi & true_hand),("false_positive",roi & ~true_hand & positive),
                             ("false_negative",~roi & true_hand),("true_negative",~roi & ~true_hand & positive)):
                roi_totals[key] += int(mask.sum())
            evaluation_mask = (depth > 0) if classes == 8 else true_hand
            rows,cols = np.nonzero(evaluation_mask)
            if args.pixel_limit and len(rows) > args.pixel_limit:
                rng = np.random.default_rng(index+args.frame_seed)
                selected = rng.choice(len(rows),args.pixel_limit,replace=False)
                rows,cols = rows[selected],cols[selected]
            oracle_prediction,_ = forest.predict(depth,rows,cols,max_splits=args.max_splits)
            prediction = oracle_prediction.copy()
            prediction[~roi[rows,cols]] = forest.document["background_label"]
            true_labels = np.where(truth[rows,cols] == 0,7,truth[rows,cols].astype(int)-1)
            is_hand = true_hand[rows,cols]
            oracle_confusion += confusion_matrix(true_labels[is_hand],oracle_prediction[is_hand],classes)
            confusion += confusion_matrix(true_labels,prediction,classes+1)
            record["evaluated_pixels"] = len(rows)
            record["roi_pixels"] = int(roi.sum())
            record["roi_true_hand_pixels"] = int((roi & true_hand).sum())
            if args.localize:
                full_labels,full_probabilities = forest.predict(depth,max_splits=args.max_splits,mask=roi)
                if classes == 8:
                    learned_hand = (full_labels < 7) & (depth > 0)
                    for key,mask in (("true_positive",learned_hand & true_hand),("false_positive",learned_hand & ~true_hand & positive),
                                     ("false_negative",~learned_hand & true_hand),("true_negative",~learned_hand & ~true_hand & positive)):
                        learned_foreground_totals[key] += int(mask.sum())
                centers = fingertip_centres(full_labels,full_probabilities,depth,intrinsics,args.probability_threshold,args.min_pixels)
                record["fingertips"] = centers
                positions = np.asarray(mat.get("pos",np.empty((3,0))),dtype=float)
                for class_id in range(5):
                    rr,cc = np.nonzero(truth == class_id+1)
                    visible = len(rr)>0
                    center = centers[class_id]
                    if visible:
                        present[class_id] += 1
                        if center is not None:
                            detected[class_id] += 1
                            error = float(np.linalg.norm(np.asarray(center["xy"])-[cc.mean(),rr.mean()]))
                            label_errors.append(error)
                            for threshold in hits_px:
                                hits_px[threshold][class_id] += error <= threshold
                    else:
                        absent[class_id] += 1
                        false_positive[class_id] += center is not None
                    if positions.ndim == 2 and positions.shape[0] >= 3 and positions.shape[1] > class_id:
                        xyz = positions[:3,class_id]
                        if visible and np.isfinite(xyz).all() and xyz[2] > 0:
                            sensor_present[class_id] += 1
                            xy = [intrinsics["fx"]*xyz[0]/xyz[2]+intrinsics["cx"],intrinsics["fy"]*xyz[1]/xyz[2]+intrinsics["cy"]]
                            correspondence.append(float(np.linalg.norm(np.asarray([cc.mean(),rr.mean()])-xy)))
                            if center is not None:
                                sensor_detected[class_id] += 1
                                sensor_errors_px.append(float(np.linalg.norm(np.asarray(center["xy"])-xy)))
                                error = float(np.linalg.norm(np.asarray(center["xyz"])-xyz))
                                sensor_errors_mm.append(error)
                                for threshold in hits_mm:
                                    hits_mm[threshold][class_id] += error <= threshold
            frame_records.append(record)
            if (index+1)%25 == 0 or index+1 == len(names):
                print(f"evaluated{index+1}/{len(names)} frames",flush=True)
        metrics = classification_metrics(confusion)
        defined_iou = [v for v in metrics["class_iou"][:7] if v is not None]
        metrics["macro_iou"] = float(np.mean(defined_iou)) if defined_iou else None
        metrics["macro_iou_domain"] = "seven learned hand classes; ROI-rejected handpixels are false negatives"
        tp,fp,fn = (roi_totals[k] for k in ("true_positive","false_positive","false_negative"))
        roi_totals.update(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,iou=tp/(tp+fp+fn) if tp+fp+fn else None)
        result = {"schema":"handnet-tea-evaluation-v1","manifest":str(Path(args.manifest).resolve()),
                  "manifest_sha256":hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),"split":split,
                  "frames":frame_records,"classification":metrics,"oracle_hand_classification":classification_metrics(oracle_confusion),
                  "classes":forest.document["classes"],"confusion_classes":forest.document["classes"]+["roi-rejected-background"],
                  "domain":("all positive-depth pixels; rawlabel0 learnedbackground7" if classes == 8 else "positive-depth rawHandNetlabels1..7; ROI-rejected labels count as errors"),
                  "foreground":args.foreground,"foreground_mask_semantics":"query centres only; feature neighbors always use unmodified raw depth",
                  "foreground_metrics_positive_depth":roi_totals,"pixel_limit_per_frame":args.pixel_limit,"max_splits":args.max_splits,
                  "training_manifest_checks":args.training_manifest,
                  "independence":"Official random frame split; no claim of subject/session independence. Matched training frame IDs rejected."}
        if args.localize:
            if classes == 8:
                tp2,fp2,fn2=(learned_foreground_totals[k] for k in ("true_positive","false_positive","false_negative"))
                learned_foreground_totals.update(precision=tp2/(tp2+fp2) if tp2+fp2 else None,recall=tp2/(tp2+fn2) if tp2+fn2 else None,iou=tp2/(tp2+fp2+fn2) if tp2+fp2+fn2 else None)
                result["learned_foreground_metrics_positive_depth"] = learned_foreground_totals
            def recalls(hits,denominators):
                result = {}
                for threshold,counts in hits.items():
                    per_class = np.divide(counts,denominators,out=np.full(5,np.nan),where=denominators>0)
                    result[str(threshold)] = {"hits":counts.tolist(),"per_class":[float(v) if np.isfinite(v) else None for v in per_class],
                                              "mean_fingertip_recall":float(np.nanmean(per_class)) if np.any(denominators) else None,
                                              "micro_recall":float(counts.sum()/denominators.sum()) if denominators.sum() else None}
                return result
            result["localization"] = {"method":"largest4connected confident predicted component, probability weighted centroid",
                                      "foreground":args.foreground,"probability_threshold":args.probability_threshold,"minimum_pixels":args.min_pixels,
                                      "present":present.tolist(),"detected":detected.tolist(),"absent":absent.tolist(),"false_positive_on_absent":false_positive.tolist(),
                                      "recall_denominator":"all groundtruth-present fingers; absent/missing predictions fail every threshold",
                                      "label_centroid_recall_at_px":recalls(hits_px,present),"sensor_pos_present":sensor_present.tolist(),
                                      "sensor_pos_recall_at_mm":recalls(hits_mm,sensor_present),"sensor_pos_detected":sensor_detected.tolist(),
                                      "to_raw_label_centroid_px":summary_errors(label_errors),"to_pos_first5_projected_px":summary_errors(sensor_errors_px),
                                      "to_pos_first5_mm":summary_errors(sensor_errors_mm),"raw_label_to_pos_correspondence_px":summary_errors(correspondence),
                                      "reference_semantics":"pos[:3,0:5] adjusted sensor centers in camera millimetres (DisplaySession/CreateH5Data); col0 thumb, first5 fingers. Label1..5 expected same order; correspondence residual audits this assumption. This is not the original paper mean-shift detection metric."}
        Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"classification":result["classification"],"foreground":args.foreground,"foreground_metrics":roi_totals,"localization":result.get("localization"),"output":args.output},indent=2))
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--manifest",required=True)
    prefix = sub.add_parser("prefix",help="Exportselecteddepthasnativecompleteartifact")
    prefix.add_argument("--model",required=True)
    prefix.add_argument("--coords",required=True)
    prefix.add_argument("--splits",required=True,type=int)
    prefix.add_argument("--output",required=True)
    export = sub.add_parser("export",help="Export a portable selectedforestprefixwithrawdepthQA")
    export.add_argument("--manifest",required=True)
    export.add_argument("--splits",required=True,type=int)
    export.add_argument("--output",required=True,help="New deployment directory; refusesexisting")
    export.add_argument("--verification-depth",help="Optional one littleendianu16rawframe")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--manifest",required=True)
    evaluate_parser.add_argument("--archive",required=True)
    evaluate_parser.add_argument("--output",required=True)
    evaluate_parser.add_argument("--split",choices=["train","validation","test","unspecified"])
    evaluate_parser.add_argument("--frame-manifest")
    evaluate_parser.add_argument("--training-manifest",action="append",default=[])
    evaluate_parser.add_argument("--max-frames",type=int,default=300,help="0 meansall; default300 randomframes")
    evaluate_parser.add_argument("--frame-seed",type=int,default=420)
    evaluate_parser.add_argument("--pixel-limit",type=int,default=0,help="0 meansall labeledpositivepixels")
    evaluate_parser.add_argument("--max-splits",type=int,help="Evaluate a saved shallower prefix, counting actualsplitdecisions")
    evaluate_parser.add_argument("--localize",action="store_true")
    evaluate_parser.add_argument("--foreground",choices=["oracle","near-component","none"],default="near-component")
    evaluate_parser.add_argument("--probability-threshold",type=float,default=.5)
    evaluate_parser.add_argument("--min-pixels",type=int,default=3)
    evaluate_parser.add_argument("--allow-training-evaluation",action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        depth=None
        if args.verification_depth:
            doc=validate_manifest(json.loads(Path(args.manifest).read_text()))
            depth=np.fromfile(args.verification_depth,dtype="<u2").reshape(doc["image"]["height"],doc["image"]["width"])
        print(export_forest_prefix(args.manifest,args.output,args.splits,depth))
    elif args.command == "prefix":
        tree=TeaTree(args.model,args.coords).write_prefix(args.output,args.splits)
        print(json.dumps({"model":str(tree.path),"splits":tree.splits,"sha256":hashlib.sha256(tree.path.read_bytes()).hexdigest()}))
    elif args.command == "inspect":
        forest = TeaForest(args.manifest)
        print(json.dumps({"classes":forest.document["classes"],"trees":[{"model":str(t.path),"splits":t.splits,"nodes":t.node_count,"root_prior":t.root_prior.tolist()} for t in forest.trees]},indent=2))
    else:
        if args.max_frames < 0 or args.pixel_limit < 0 or not 0 <= args.probability_threshold <= 1 or args.min_pixels < 1:
            parser.error("invalid evaluation bound/postprocessingparameter")
        evaluate(args)


if __name__ == "__main__":
    main()
