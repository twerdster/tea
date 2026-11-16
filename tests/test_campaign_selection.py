#!/usr/bin/env python3
"""Selection and saved-label checks, independent of CUDA/training resources."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_handnet_campaign as campaign


def result(present=100,detected=100,hits=100,false_positive=0):
    return {"localization":{"sensor_pos_recall_at_mm":{"15":{"hits":[hits]*5}},
                            "sensor_pos_present":[present]*5,"sensor_pos_detected":[detected]*5,
                            "detected":[detected]*5,"false_positive_on_absent":[false_positive]*5}}


class SelectionTests(unittest.TestCase):
    def test_perfect_missing_wrong_and_spurious(self):
        self.assertEqual(campaign.validation_score(result()),1)
        self.assertEqual(campaign.validation_score(result(detected=0,hits=0)),0)
        self.assertEqual(campaign.validation_score(result(hits=0)),0)
        self.assertEqual(campaign.validation_score(result(hits=50)),.5)
        self.assertAlmostEqual(campaign.validation_score(result(false_positive=100)),2/3)

    def test_background_accuracy_cannot_override_tip_score(self):
        miss=result(detected=0,hits=0);miss["classification"]={"accuracy":.999}
        reliable=result(hits=70,detected=70)
        noisy=result(hits=90,false_positive=900)
        self.assertGreater(campaign.validation_score(reliable),campaign.validation_score(miss))
        self.assertGreater(campaign.validation_score(reliable),campaign.validation_score(noisy))

    def test_unknown_sensor_positions_excluded_from_prediction_universe(self):
        value=result(present=80,detected=80,hits=80)
        value["localization"]["detected"]=[100]*5
        self.assertEqual(campaign.validation_score(value),1)

    def test_invalid_denominators_and_counts_rejected(self):
        invalid=[result(present=0,detected=0,hits=0),result(hits=101),result(detected=1,hits=2)]
        wrongshape=result();wrongshape["localization"]["sensor_pos_present"]=[1,2]
        invalid.append(wrongshape)
        nonfinite=result();nonfinite["localization"]["sensor_pos_present"][0]=float("nan")
        invalid.append(nonfinite)
        negative=result();negative["localization"]["false_positive_on_absent"][0]=-1
        invalid.append(negative)
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):campaign.validation_score(value)

    def test_saved_predictions_reject_truncation_excess_and_disagreement(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"predictions.u16"
            expected=np.array([0,7,1,2],dtype="<u2")
            expected.tofile(path)
            campaign.verify_saved_predictions(path,expected,4)
            campaign.verify_saved_predictions(path,expected[:2],4)
            for payload in (expected.tobytes()[:-1],expected.tobytes()+b"\0\0"):
                path.write_bytes(payload)
                with self.assertRaises(ValueError):campaign.verify_saved_predictions(path,expected,4)
            expected.tofile(path)
            with self.assertRaises(ValueError):campaign.verify_saved_predictions(path,np.array([7,0,1,2]),4)

    def test_train_reads_native_saved_predictions_as_uint16(self):
        # Actual binary columns and saved labels; only the expensive trainer and
        # parsed model are replaced. This catches a u32 reader combining labels.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);repo=root/"repo";run=root/"runs/current";data=root/"data"
            repo.mkdir();run.mkdir(parents=True);data.mkdir()
            (repo/"Tea").write_bytes(b"unusedfakebinary")
            expected=np.arange(16,dtype="<u2")%8
            for feature in range(1200):
                expected.astype(np.int8).tofile(data/f"F_{feature:04d}.feat")
            (data/"Coords.co").write_bytes(b"unusedtestcoords")
            metadata={"complete":True,"samples":16,"features":1200,"classes":campaign.CLASS_NAMES+["background"],"seed":42}
            (data/"handnet_metadata.json").write_text(json.dumps(metadata))
            def fake_guard(path):
                (data/"Tree_0000.tree").write_bytes(b"unusedfakemodel")
                expected.tofile(data/"Tree_0000.tree.labels")
            class FakeTree:
                splits=20;classes=8
                def __init__(self,path,coords):self.path,self.coords_path=Path(path),Path(coords)
                def predict_features(self,values):return np.eye(8)[values[:,0]]
            with mock.patch.object(campaign,"guarded_train",fake_guard),mock.patch.object(campaign,"TeaTree",FakeTree):
                task=campaign.Campaign(root)
                info,manifest=task.train(data,"mock-pilot",True)
            verification=json.loads((run/"mock-pilot/model-validation.json").read_text())
            self.assertTrue(verification["external_predictions_match"])
            self.assertEqual(verification["external_prediction_samples"],16)
            self.assertEqual(json.loads(manifest.read_text())["modeled_background_class"],7)


if __name__=="__main__":
    unittest.main()
