"""Failure and lease lifecycle regressions; no GPU/device or training launched."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/run_guarded_training.py"
SPEC = importlib.util.spec_from_file_location("campaign_guard_tests", SCRIPT)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class GuardFailureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tea-guard-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.lock_path = self.root/"gpu.lock"
        self.lock_path.write_text("preserve cooperative owner metadata\n")
        self.budget = self.root/"budget.json"
        self.budget.write_text(json.dumps({"max_active_training_seconds": 60,
                                          "used_active_training_seconds": 0}))
        self.spec_path = self.root/"spec.json"
        self.spec = {"argv": [sys.executable,"-c","pass"], "cwd": str(self.root),
                     "output_dir": str(self.root/"run"), "checkpoint_dir": str(self.root/"run"),
                     "max_seconds": 5, "gpu_lock": str(self.lock_path),
                     "budget_file": str(self.budget), "tea_log": str(self.root/"LogFile_test_0000.log")}
        self.save_spec()
        self.signals = patch.object(guard.signal,"signal")
        self.signals.start()
        self.addCleanup(self.signals.stop)

    def save_spec(self):
        self.spec_path.write_text(json.dumps(self.spec))

    def status(self):
        return json.loads((self.root/"run/status.json").read_text())

    def assert_lease_free(self):
        with self.lock_path.open("r") as descriptor:
            fcntl.flock(descriptor, fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        self.assertEqual(self.lock_path.read_text(),"preserve cooperative owner metadata\n")

    def idle(self):
        return patch.object(guard,"gpu_state",return_value={"index":0,"free_mb":4096,"utilization":0})

    def test_existing_state_is_never_overwritten_or_relaunched(self):
        (self.root/"run").mkdir()
        state = self.root/"run/status.json"
        state.write_text('{"phase":"failed","evidence":"keep"}\n')
        original = state.read_bytes(),self.budget.read_bytes()
        with patch.object(guard.subprocess,"Popen") as launch, patch.object(guard,"gpu_state") as telemetry:
            with self.assertRaisesRegex(RuntimeError,"already exists"):
                guard.main(self.spec_path)
        launch.assert_not_called();telemetry.assert_not_called()
        self.assertEqual((state.read_bytes(),self.budget.read_bytes()),original)
        self.assert_lease_free()

    def test_existing_log_survives_failed_launch(self):
        (self.root/"run").mkdir()
        (self.root/"run/training.log").write_text("prior failure evidence\n")
        with self.idle(), patch.object(guard.subprocess,"Popen") as launch:
            with self.assertRaises(FileExistsError):
                guard.main(self.spec_path)
        launch.assert_not_called()
        self.assertEqual((self.root/"run/training.log").read_text(),"prior failure evidence\n")
        self.assertEqual(self.status()["phase"],"failed")
        self.assert_lease_free()

    def test_exhausted_budget_never_reads_gpu_or_starts_process(self):
        self.budget.write_text(json.dumps({"max_active_training_seconds":60,"used_active_training_seconds":60}))
        before = self.budget.read_bytes()
        with patch.object(guard.subprocess,"Popen") as launch, patch.object(guard,"gpu_state") as telemetry:
            with self.assertRaisesRegex(RuntimeError,"budget exhausted"):
                guard.main(self.spec_path)
        launch.assert_not_called();telemetry.assert_not_called()
        self.assertEqual(self.budget.read_bytes(),before)
        self.assert_lease_free()

    def test_telemetry_failure_is_not_success_and_releases_lease(self):
        with patch.object(guard,"gpu_state",side_effect=RuntimeError("telemetry unavailable")), patch.object(guard.subprocess,"Popen") as launch:
            with self.assertRaisesRegex(RuntimeError,"telemetry unavailable"):
                guard.main(self.spec_path)
        launch.assert_not_called()
        self.assertEqual(self.status()["phase"],"failed")
        self.assertEqual(json.loads(self.budget.read_text())["used_active_training_seconds"],0)
        self.assert_lease_free()

    def test_busy_gpu_wait_does_not_launch_or_hold_lease(self):
        with patch.object(guard,"gpu_state",return_value={"index":0,"free_mb":100,"utilization":80}), patch.object(guard.time,"sleep",side_effect=InterruptedError("bounded wait ended")), patch.object(guard.subprocess,"Popen") as launch:
            with self.assertRaises(InterruptedError):
                guard.main(self.spec_path)
        launch.assert_not_called()
        state = self.status()
        self.assertEqual(state["phase"],"failed")
        self.assertEqual(state["wait_reason"],"GPU resources busy")
        self.assertEqual(state["active_training_seconds"],0)
        self.assert_lease_free()

    def test_budget_accounting_write_failure_still_releases_lease(self):
        original_write = guard.write_json
        def write(path,value):
            if Path(path)==self.budget:
                raise OSError("simulated accounting disk error")
            return original_write(path,value)
        with self.idle(),patch.object(guard.time,"sleep"),patch.object(guard,"write_json",side_effect=write):
            with self.assertRaisesRegex(OSError,"accounting disk error"):
                guard.main(self.spec_path)
        state = self.status()
        self.assertEqual(state["phase"],"failed")
        self.assertTrue(state["accounting_failed"])
        self.assertEqual(state["tea_log"],self.spec["tea_log"])
        self.assert_lease_free()

    def test_final_status_write_failure_still_releases_lease(self):
        original_write = guard.write_json
        def write(path,value):
            if Path(path).name=="status.json" and value["phase"]=="training_complete_pending_validation":
                raise OSError("simulated final status disk error")
            return original_write(path,value)
        with self.idle(),patch.object(guard.time,"sleep"),patch.object(guard,"write_json",side_effect=write):
            with self.assertRaisesRegex(OSError,"final status disk error"):
                guard.main(self.spec_path)
        self.assertGreater(json.loads(self.budget.read_text())["used_active_training_seconds"],0)
        self.assert_lease_free()

    def test_time_limit_terminates_only_owned_child_group(self):
        class Child:
            pid = 999999999
            finished = False
            def poll(self):
                return 0 if self.finished else None
            def wait(self,timeout=None):
                self.finished=True
                return 0
        child = Child()
        with self.idle(),patch.object(guard.subprocess,"Popen",return_value=child),patch.object(guard.os,"killpg") as kill,patch.object(guard.time,"monotonic",side_effect=[0,10,10]):
            with self.assertRaisesRegex(RuntimeError,"time limit"):
                guard.main(self.spec_path)
        kill.assert_called_once_with(child.pid,guard.signal.SIGTERM)
        self.assertTrue(child.finished)
        self.assertTrue(self.status()["limit_reached"])
        self.assertEqual(self.status()["phase"],"failed")
        self.assert_lease_free()


if __name__=="__main__":
    unittest.main()
