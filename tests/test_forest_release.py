#!/usr/bin/env python3
"""Privacy/path boundaries and unchanged inference in a local release bundle."""
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_forest_release as release
from tea_model import TeaForest, file_sha256

REVISION = "27ae728e9cb012d2255337d9b3583bbe7f66b3cc"
SECRET = "/private/runtime/do-not-publish"


def stump(path, opposite=False):
    # Native stump with contrasting children, independently encoded using struct.
    histograms = [[127,127], [0,255] if opposite else [255,0], [255,0] if opposite else [0,255]]
    with path.open("wb") as handle:
        handle.write(struct.pack("<4iif", 0,2,1,3,0,0.))
        for histogram in histograms:
            handle.write(struct.pack("<IH2x", 1, int(histogram[1] > histogram[0])))
        handle.write(bytes(sum(histograms,[])))


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.private = self.root / "private"
        self.private.mkdir()
        for index in range(2):
            directory = self.private / f"tree{index}"
            directory.mkdir()
            stump(directory / "model.tree", bool(index))
            np.array([[400,0,0,0]], dtype="<f4").tofile(directory / "Coords.co")
        self.document = {
            "schema":"depthtrack-tea-forest-v1",
            "image":{"width":8,"height":8,"depth_unit":"millimetres","intrinsics":{"fx":240.,"fy":241.,"cx":4.,"cy":4.},"debug_path":SECRET},
            "feature":{"type":"F_CHAR","max_diff":250,"maximum":127,"coordinate_rule":"truncate-after-base-plus-offset-over-depth","coordinate_precision":"float32-complete-sum","missing":{"both":0,"first":127,"second":-127},"debug":{"path":SECRET}},
            "classes":["thumb","index"],"background_label":2,"unknown_label":255,"aggregation":"mean-normalized-leaf-probabilities","empty_leaf":"root-prior",
            "trees":[{"model":"tree0/model.tree","coords":"tree0/Coords.co","weight":1,"private":{"path":SECRET}},
                     {"model":"tree1/model.tree","coords":"tree1/Coords.co","weight":1+1e-10,"source_model":SECRET}],
            "postprocessing":{"method":release.POST_METHOD,"foreground":"none","probability_threshold":.25,"min_pixels":3,"fingertip_classes":[1,0],"debug_path":SECRET},
            "provenance":{"pilot":{"manifest":SECRET,"result":SECRET},"host":"do-not-publish-host"},
        }
        self.metadata = {"selection_metric":"macro-fingertip-f1-at-15mm","selected_on":"validation","test_used_for_selection":False,
                         "probe_size_pixel_mm":5000,"training_frame_fraction":.9,
                         "trees":[dict(coordinate_seed=seed,sampling_seed=seed,frame_seed=seed,training_samples=1000,training_frames=100,features=1,split_decisions=1) for seed in [201,202]]}
        self.manifest = self.private / "forest.json"
        self.save()
        (self.private / "not-an-asset.bin").write_bytes(b"private image data")

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.manifest.write_text(json.dumps(self.document))

    def test_public_bundle_drops_recursive_private_values_and_preserves_semantics(self):
        original = {p.relative_to(self.private).as_posix():file_sha256(p) for p in self.private.rglob("*") if p.is_file()}
        public = release.prepare_release(self.manifest,self.root/"release",REVISION,self.metadata)
        result = json.loads(public.read_text())
        files = {p.relative_to(public.parent).as_posix() for p in public.parent.rglob("*") if p.is_file()}
        self.assertEqual(files,{"forest.json","release-verification.json","assets/tree000/model.tree","assets/tree000/Coords.co","assets/tree001/model.tree","assets/tree001/Coords.co"})
        bundle_text = public.read_text()+(public.parent/"release-verification.json").read_text()
        self.assertNotIn(SECRET,bundle_text)
        self.assertNotIn("do-not-publish-host",bundle_text)
        self.assertEqual(result["provenance"]["training_source_revision"],REVISION)
        self.assertEqual(result["trees"][1]["weight"],1+1e-10)
        self.assertEqual(result["postprocessing"]["fingertip_classes"],[1,0])
        self.assertEqual(result["postprocessing"]["min_pixels"],3)
        verification=json.loads((public.parent/"release-verification.json").read_text())
        self.assertEqual(verification["private_source_manifest_sha256"],original["forest.json"])
        self.assertEqual(verification["public_manifest_sha256"],file_sha256(public))
        self.assertEqual(len(verification["synthetic_checks"]),8)
        self.assertIn("REQUIRED_BEFORE_PUBLICATION",verification["actual_frame_python_native_parity"])
        self.assertTrue(all(record["per_tree_heaps_exact"] for record in verification["synthetic_checks"]))
        self.assertEqual(original,{p.relative_to(self.private).as_posix():file_sha256(p) for p in self.private.rglob("*") if p.is_file()})
        depth=np.full((8,8),700,np.uint16)
        before=TeaForest(self.manifest).predict(depth)
        after=TeaForest(public).predict(depth)
        np.testing.assert_array_equal(before[0],after[0])
        self.assertEqual(before[1].tobytes(),after[1].tobytes())

    def test_absent_postprocessing_defaults_stay_absent(self):
        del self.document["postprocessing"]
        self.save()
        public=release.prepare_release(self.manifest,self.root/"release",REVISION,self.metadata)
        self.assertNotIn("postprocessing",json.loads(public.read_text()))
        report=json.loads((public.parent/"release-verification.json").read_text())
        self.assertIn("not-declared",report["default_observations_exact"])

    def test_path_escape_symlink_and_existing_output_refused(self):
        for index,value in enumerate([str(self.private/"tree0/model.tree"),"../outside.tree","tree0/../../outside.tree","C:\\secret\\model.tree","tree0/secret\nmodel.tree"]):
            with self.subTest(value=value):
                self.document["trees"][0]["model"]=value
                self.save()
                with self.assertRaises(ValueError):release.prepare_release(self.manifest,self.root/f"bad{index}",REVISION,self.metadata)
                self.assertFalse((self.root/f"bad{index}").exists())
        self.document["trees"][0]["model"]="tree0/link.tree"
        (self.private/"tree0/link.tree").symlink_to(self.private/"tree0/model.tree")
        self.save()
        with self.assertRaises(ValueError):release.prepare_release(self.manifest,self.root/"symlink",REVISION,self.metadata)
        self.document["trees"][0]["model"]="tree0/model.tree"
        self.save()
        existing=self.root/"existing";existing.mkdir();(existing/"keep").write_text("preserve")
        with self.assertRaises(ValueError):release.prepare_release(self.manifest,existing,REVISION,self.metadata)
        self.assertEqual((existing/"keep").read_text(),"preserve")
        broken=self.root/"broken";broken.symlink_to(self.root/"missing")
        with self.assertRaises(ValueError):release.prepare_release(self.manifest,broken,REVISION,self.metadata)
        self.assertTrue(broken.is_symlink())
        self.assertFalse((self.root/"missing").exists())

    def test_unsafe_reviewed_metadata_wrong_depth_and_large_assets_refused(self):
        invalid=[]
        bad=copy.deepcopy(self.metadata);bad["notes"]={"path":SECRET};invalid.append(bad)
        bad=copy.deepcopy(self.metadata);bad["test_used_for_selection"]=True;invalid.append(bad)
        bad=copy.deepcopy(self.metadata);bad["trees"][0]["coordinate_seed"]=True;invalid.append(bad)
        bad=copy.deepcopy(self.metadata);bad["trees"][0]["split_decisions"]=2;invalid.append(bad)
        for index,bad in enumerate(invalid):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):release.prepare_release(self.manifest,self.root/f"bad{index}",REVISION,bad)
        with self.assertRaises(ValueError):release.prepare_release(self.manifest,self.root/"badrevision",SECRET,self.metadata)
        from unittest.mock import patch
        with patch.object(release,"MAX_ASSET_BYTES",1):
            with self.assertRaises(ValueError):release.prepare_release(self.manifest,self.root/"oversize",REVISION,self.metadata)
        self.assertFalse((self.root/"oversize").exists())

    def test_declared_selected_asset_hashes_must_match(self):
        for key in ("model_sha256", "coords_sha256"):
            for claimed in ("0"*64,"A"*64,"invalid"):
                with self.subTest(key=key,claimed=claimed):
                    self.document["trees"][0][key]=claimed
                    self.save()
                    before=self.manifest.read_bytes()
                    with self.assertRaises(ValueError):release.prepare_release(self.manifest,self.root/"mismatch",REVISION,self.metadata)
                    self.assertEqual(self.manifest.read_bytes(),before)
                    self.assertFalse((self.root/"mismatch").exists())
            self.document["trees"][0][key]=file_sha256(self.private/self.document["trees"][0]["model" if key=="model_sha256" else "coords"])
        self.save()
        public=release.prepare_release(self.manifest,self.root/"matching",REVISION,self.metadata)
        prepared=json.loads(public.read_text())["trees"][0]
        self.assertEqual(prepared["model_sha256"],self.document["trees"][0]["model_sha256"])
        self.assertEqual(prepared["coords_sha256"],self.document["trees"][0]["coords_sha256"])

    def test_cli_requires_explicit_training_revision(self):
        metadata=self.root/"reviewed.json";metadata.write_text(json.dumps(self.metadata))
        script=Path(release.__file__)
        result=subprocess.run([sys.executable,str(script),"--manifest",str(self.manifest),"--output",str(self.root/"release"),"--selection-metadata",str(metadata),"--training-source-revision",REVISION],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads((self.root/"release/forest.json").read_text())["provenance"]["training_source_revision"],REVISION)


if __name__=="__main__":unittest.main()
