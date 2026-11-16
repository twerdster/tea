#!/usr/bin/env python3
"""Independent analytic/data-format regressions; no GPU or real data required."""
import copy
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import scipy.io as sio

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import handnet_to_tea as convert
import tea_model as model

REPO = Path(__file__).resolve().parents[1]


def write_tree(path, depth=0):
    """Analytic stump: firstfeature>0=>class1, elseclass0. Empty rightleaf testsprior."""
    n = (1 << (depth+1))-1
    nodes = np.zeros(n,dtype=model.NODE_DTYPE)
    nodes["feature"] = 0
    nodes["threshold"] = 0
    leaves = np.zeros(2*n+1,dtype=model.LEAF_DTYPE)
    hist = np.tile(np.array([255,0],np.uint8),(2*n+1,1))
    hist[0] = [64,192]
    hist[n:] = [0,255]
    hist[n] = [255,0]
    leaves["maximum"] = 1
    leaves["label"] = hist.argmax(axis=1)
    with open(path,"wb") as h:
        h.write(struct.pack("<4i",depth,2,n,2*n+1))
        nodes.tofile(h); leaves.tofile(h); hist.tofile(h)
    return hist


def make_manifest(directory, models=None, shape=(3,4)):
    doc = {"schema":model.SCHEMA,"image":{"width":shape[1],"height":shape[0],"depth_unit":"millimetres","intrinsics":dict(model.DEFAULT_INTRINSICS)},
           "feature":{"type":"F_CHAR","max_diff":250,"maximum":127,"coordinate_rule":"truncate-after-base-plus-offset-over-depth","coordinate_precision":"float32-complete-sum","missing":{"both":0,"first":127,"second":-127}},
           "classes":["first","second"],"background_label":2,"unknown_label":255,"aggregation":"mean-normalized-leaf-probabilities","empty_leaf":"root-prior",
           "trees":models or [{"model":"tree.tree","coords":"Coords.co","weight":1}]}
    path = Path(directory)/"forest.json"
    path.write_text(json.dumps(doc))
    return path,doc


class FeatureTests(unittest.TestCase):
    def test_depth_features_independent_analytic(self):
        depth = np.array([[100,110,120],[200,220,240],[300,330,360]],np.float32)
        # Center[1,1] z220. First offset -0.2px truncates total .8 to0.
        # If offset were truncated before adding it would incorrectlysample row1.
        coords = np.array([[-44,0,10000,0],[0,220,0,0],[0,0,0,10000],[0,0,0,0]],np.float32)
        got = convert.compute_features(depth,np.array([1]),np.array([1]),coords)
        # (110-220)/250*127=>ceil(-55.88)=-55; (240-220)=>11.
        np.testing.assert_array_equal(got,[[-55,11,127,-127]])
        repeated = model.feature_values(depth,np.repeat(1,4),np.repeat(1,4),coords.T)
        np.testing.assert_array_equal(repeated,got[0])

    def test_zero_and_missing_sentinel(self):
        depth = np.ones((2,2),np.float32)*400
        coordinates = np.array([[40000],[40000],[40000],[40000]],np.float32)
        self.assertEqual(convert.compute_features(depth,np.array([0]),np.array([0]),coordinates)[0,0],0)
        depth[0,0] = 0
        with self.assertRaises(ValueError):
            convert.compute_features(depth,np.array([0]),np.array([0]),coordinates)

    def test_coordinate_scale_and_determinism(self):
        small = convert.generate_coords(40,50,2)
        large = convert.generate_coords(40,50000,2)
        np.testing.assert_allclose(large,small*1000,rtol=1e-6,atol=.003)
        np.testing.assert_array_equal(large,convert.generate_coords(40,50000,2))
        self.assertGreater(np.abs(large[:2]).max()/453,50)
        self.assertLess(np.abs(small[:2]).max()/453,.06)

    def test_balanced_nonreplacement_sampling(self):
        label = np.arange(700).reshape(20,35)%7+1
        depth = np.ones_like(label)*500
        rows,cols = convert.select_samples(label,70,np.random.default_rng(4),depth,"balanced",align=1)
        self.assertEqual(len(rows),70)
        np.testing.assert_array_equal(np.bincount(label[rows,cols])[1:],[10]*7)
        self.assertEqual(len(set(zip(rows.tolist(),cols.tolist()))),70)
        depth[:] = 0
        rows,cols = convert.select_samples(label,70,np.random.default_rng(4),depth,"balanced")
        self.assertEqual(len(rows),0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_tree(self.root/"tree.tree")
        np.zeros((1,4),dtype="<f4").tofile(self.root/"Coords.co")
        self.manifest,self.doc = make_manifest(self.root)
    def tearDown(self):
        self.tmp.cleanup()

    def test_heap_strict_threshold_and_probability(self):
        tree = model.TeaTree(self.root/"tree.tree",self.root/"Coords.co")
        p,h = tree.predict_features(np.array([[-1],[0],[1]]),return_heaps=True)
        np.testing.assert_array_equal(h,[1,1,2])
        np.testing.assert_array_equal(p,[[1,0],[1,0],[0,1]])
        prior = tree.predict_features(np.zeros((1,1)),max_splits=0)
        np.testing.assert_allclose(prior,[[.25,.75]])

    def test_empty_leaf_root_prior(self):
        path = self.root/"tree.tree"
        with path.open("r+b") as h:
            h.seek(-2,2); h.write(b"\0\0")
        tree = model.TeaTree(path,self.root/"Coords.co")
        np.testing.assert_allclose(tree.predict_features(np.ones((1,1))),[[.25,.75]])

    def test_ensemble_normalized_weighting(self):
        write_tree(self.root/"second.tree")
        with (self.root/"second.tree").open("r+b") as h:
            h.seek(-4,2); h.write(bytes([0,255,255,0]))
        make_manifest(self.root,[{"model":"tree.tree","coords":"Coords.co","weight":1},{"model":"second.tree","coords":"Coords.co","weight":3}])
        forest = model.TeaForest(self.manifest)
        depth = np.ones((3,4),np.uint16)*400; depth[0,0] = 0
        labels,probability = forest.predict(depth)
        self.assertEqual(labels[0,0],2)
        np.testing.assert_allclose(probability[0,0],[0,0])
        np.testing.assert_allclose(probability[1,1],[.25,.75])
        self.assertEqual(labels[1,1],1)

    def test_class_tie_after_float32_aggregation(self):
        write_tree(self.root/"second.tree")
        with (self.root/"second.tree").open("r+b") as h:
            h.seek(-4,2);h.write(bytes([0,255,255,0]))
        make_manifest(self.root,[{"model":"tree.tree","coords":"Coords.co","weight":1},{"model":"second.tree","coords":"Coords.co","weight":1+1e-10}])
        labels,probability=model.TeaForest(self.manifest).predict(np.ones((3,4),np.uint16)*400)
        np.testing.assert_array_equal(probability[1,1],[.5,.5])
        self.assertEqual(int(labels[1,1]),0)

    def test_bad_headers_and_missing_manifest_fields(self):
        for malformed in [[],{"schema":model.SCHEMA,"image":{}},dict(self.doc,classes=[["invalid"]]),dict(self.doc,trees=["invalid"])]:
            with self.assertRaises(ValueError):model.validate_manifest(malformed)
        for field in ["aggregation","empty_leaf","feature","classes"]:
            bad=copy.deepcopy(self.doc); del bad[field]
            with self.assertRaises(ValueError):model.validate_manifest(bad)
        with (self.root/"tree.tree").open("ab") as h:h.write(b"x")
        with self.assertRaises(ValueError):model.TeaTree(self.root/"tree.tree",self.root/"Coords.co")

    def test_deep_heap_route(self):
        # Avoid allocating a20-level model in unit tests; depth5 exercisesdeepheap.
        write_tree(self.root/"deep.tree",depth=5)
        tree = model.TeaTree(self.root/"deep.tree",self.root/"Coords.co")
        _,heaps = tree.predict_features(np.array([[0],[1]]),return_heaps=True)
        np.testing.assert_array_equal(heaps,[63,126])

    def test_native_prefix_preserves_selected_predictions(self):
        write_tree(self.root/"deep.tree",depth=5)
        tree=model.TeaTree(self.root/"deep.tree",self.root/"Coords.co")
        cropped=tree.write_prefix(self.root/"prefix.tree",3)
        self.assertEqual(cropped.splits,3)
        features=np.array([[-1],[0],[1]])
        expected,heap=tree.predict_features(features,max_splits=3,return_heaps=True)
        actual,actual_heap=cropped.predict_features(features,return_heaps=True)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(actual_heap,heap)
        with self.assertRaises(ValueError):tree.write_prefix(self.root/"prefix.tree",3)

    def test_portable_prefix_export_raw_depth_qa(self):
        write_tree(self.root/"tree.tree",depth=5)
        depth=np.ones((3,4),np.uint16)*400;depth[0,0]=0
        exported=model.export_forest_prefix(self.manifest,self.root/"deploy",3,depth)
        selected=model.TeaForest(exported)
        self.assertEqual(selected.trees[0].splits,3)
        verification=json.loads((self.root/"deploy/prefix-verification.json").read_text())
        self.assertTrue(verification["per_tree_heaps_exact"])
        self.assertTrue(verification["probabilities_exact"])
        self.assertTrue((self.root/"tree.tree").exists())
        self.assertEqual(json.loads(exported.read_text())["trees"][0]["model"],"tree0/model.tree")
        with self.assertRaises(ValueError):model.export_forest_prefix(self.manifest,self.root/"deploy",3)
        with self.assertRaises(ValueError):model.export_forest_prefix(self.manifest,self.root/"bad",0)
        with self.assertRaises(ValueError):model.export_forest_prefix(self.manifest,self.root/"bad",7)

    def test_confusion_iou_and_missing_class(self):
        confusion = model.confusion_matrix(np.array([0,0,1,1]),np.array([0,1,1,1]),3)
        metrics = model.classification_metrics(confusion)
        self.assertEqual(metrics["accuracy"],.75)
        self.assertAlmostEqual(metrics["macro_iou"],(.5+2/3)/2)
        self.assertEqual(metrics["class_iou"][2],None)

    def test_localization_coordinates(self):
        labels=np.ones((3,4),dtype=np.uint8)*2
        labels[1,1:3]=0
        probabilities=np.zeros((3,4,2),np.float32); probabilities[1,1:3,0]=1
        intrinsics={"fx":100,"fy":100,"cx":0,"cy":0}
        centers=model.fingertip_centres(labels,probabilities,np.ones((3,4))*500,intrinsics,min_pixels=2)
        np.testing.assert_allclose(centers[0]["xy"],[1.5,1])
        np.testing.assert_allclose(centers[0]["xyz"],[7.5,5,500])
        self.assertIsNone(centers[1])


class ForegroundTests(unittest.TestCase):
    def test_near_component_bounds_tie_and_minimum(self):
        depth=np.zeros((8,20),np.uint16)
        depth[:3,:10]=400
        depth[5:,10:]=400
        depth[4,0]=50
        depth[4,1]=1600
        mask=model.near_component_mask(depth)
        self.assertEqual(int(mask.sum()),30)
        self.assertTrue(mask[0,0])
        self.assertFalse(mask[5,10])
        depth[:]=0;depth[:2,:10]=400
        self.assertEqual(int(model.near_component_mask(depth).sum()),0)

    def test_eight_class_background_and_missing_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            nodes=np.zeros(1,dtype=model.NODE_DTYPE)
            leaves=np.zeros(3,dtype=model.LEAF_DTYPE);leaves["maximum"]=1;leaves["label"]=7
            hist=np.zeros((3,8),np.uint8);hist[:,7]=255
            with (root/"tree.tree").open("wb") as h:
                h.write(struct.pack("<4i",0,8,1,3));nodes.tofile(h);leaves.tofile(h);hist.tofile(h)
            np.zeros((1,4),dtype="<f4").tofile(root/"Coords.co")
            manifest,doc=make_manifest(root)
            doc.update(classes=convert.CLASS_NAMES+["background"],background_label=8,modeled_background_class=7)
            manifest.write_text(json.dumps(doc))
            depth=np.ones((3,4),np.uint16)*400;depth[0,0]=0
            forest=model.TeaForest(manifest)
            labels,probs=forest.predict(depth)
            self.assertEqual(int(labels[0,0]),8)
            self.assertTrue(np.all(labels[depth>0]==7))
            # One present thumb, model predicts onlybackground: missingprediction
            # must scorezero recall, ratherthan beingdroppedfromdenominator.
            label=np.zeros((3,4),np.uint8);label[1,1:3]=1
            source=root/"ValidationData";source.mkdir()
            pos=np.ones((3,6),np.float32);pos[2,:]=400
            sio.savemat(source/"Data_0000001.mat",{"depth":depth,"lbl":label,"pos":pos,"gidx":123})
            output=root/"eval.json"
            subprocess.run([sys.executable,str(REPO/"scripts/tea_model.py"),"evaluate","--manifest",str(manifest),"--archive",str(source),"--output",str(output),"--foreground","none","--localize","--min-pixels","1"],check=True,capture_output=True)
            result=json.loads(output.read_text())
            self.assertEqual(result["classification"]["support"][7],9)
            self.assertEqual(result["localization"]["sensor_pos_present"][0],1)
            self.assertEqual(result["localization"]["sensor_pos_recall_at_mm"]["15"]["mean_fingertip_recall"],0)
            self.assertEqual(result["localization"]["detected"][0],0)


class ConversionTests(unittest.TestCase):
    def test_real_cli_selection_identity_and_bank_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); archive=root/"TrainData.zip"
            with zipfile.ZipFile(archive,"w") as z:
                for i in range(10):
                    b=io.BytesIO()
                    sio.savemat(b,{"depth":np.full((10,14),450+i,np.uint16),"lbl":np.arange(140,dtype=np.uint8).reshape(10,14)%7+1,
                                  "session":"synthetic","idx":i,"gidx":i+100})
                    z.writestr(f"TrainData/Data_{i:07d}.mat",b.getvalue())
            common=[sys.executable,str(REPO/"scripts/handnet_to_tea.py"),"--archive",str(archive),"--features","40","--samples-per-frame","56","--sampling","balanced","--frame-seed","17","--max-files","4","--split","train"]
            subprocess.run(common+["--output",str(root/"one")],check=True,capture_output=True)
            subprocess.run(common+["--output",str(root/"two"),"--seed","999","--coords",str(root/"one/Coords.co"),"--frame-manifest",str(root/"one/frames.json"),"--sampling-seed","0"],check=True,capture_output=True)
            first=json.loads((root/"one/handnet_metadata.json").read_text())
            second=json.loads((root/"two/handnet_metadata.json").read_text())
            self.assertEqual(first["samples"],224)
            self.assertEqual(first["coords_sha256"],second["coords_sha256"])
            self.assertEqual(first["sampling_seed"],second["sampling_seed"])
            for name in ("Coords.co","Labels.lbl","Pixels.pix","F_0000.feat"):
                self.assertEqual((root/"one"/name).read_bytes(),(root/"two"/name).read_bytes())
            frames=convert.read_frame_manifest(root/"one/frames.json",expected_split="train")
            self.assertEqual(len(frames["frames"]),4)
            self.assertEqual(len({f["gidx"] for f in frames["frames"]}),4)
            with self.assertRaises(ValueError):convert.read_frame_manifest(root/"one/frames.json",expected_split="test")
            bad=copy.deepcopy(frames);bad["frames"][0]["frame_id"]=""
            (root/"bad.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):convert.read_frame_manifest(root/"bad.json")
            result=subprocess.run(common+["--output",str(root/"bounded"),"--max-output-gib","0.000001"],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/"bounded/Labels.lbl").exists())

    def test_background_cli_excludes_invalid_depth(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/"TrainData";source.mkdir()
            label=np.arange(160,dtype=np.uint8).reshape(10,16)%8
            depth=np.full(label.shape,500,np.uint16)
            depth[(label==0) & (np.indices(label.shape)[0]<5)]=0
            sio.savemat(source/"Data_0000001.mat",{"depth":depth,"lbl":label,"gidx":987})
            output=root/"output"
            subprocess.run([sys.executable,str(REPO/"scripts/handnet_to_tea.py"),"--archive",str(source),"--output",str(output),"--include-background","--sampling","balanced","--samples-per-frame","64","--sample-alignment","1","--features","20"],check=True,capture_output=True)
            metadata=json.loads((output/"handnet_metadata.json").read_text())
            self.assertEqual(metadata["samples"],64)
            self.assertEqual(metadata["modeled_background_class"],7)
            self.assertEqual(metadata["classes"][-1],"background")
            self.assertEqual(metadata["label_hist"]["7"],8)
            pixels=np.fromfile(output/"Pixels.pix",dtype="<u2").reshape(-1,2)
            self.assertTrue(np.all(depth[pixels[:,0],pixels[:,1]]>0))
            labels=np.fromfile(output/"Labels.lbl",dtype="<u2")
            np.testing.assert_array_equal(np.bincount(labels,minlength=8),[8]*8)


if __name__ == "__main__":
    unittest.main()
