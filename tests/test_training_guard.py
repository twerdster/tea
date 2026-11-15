import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("guard", Path(__file__).parents[1] / "scripts/run_guarded_training.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class GuardTest(unittest.TestCase):
    def run_case(self, exit_code):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gpu.lock").touch()
            budget = root / "budget.json"
            budget.write_text(json.dumps({"max_active_training_seconds": 60, "used_active_training_seconds": 0}))
            configuration = {"argv": [sys.executable, "-c", f"raise SystemExit({exit_code})"],
                             "cwd": str(root), "output_dir": str(root / "run"),
                             "checkpoint_dir": str(root / "run"), "max_seconds": 5,
                             "gpu_lock": str(root / "gpu.lock"), "budget_file": str(budget)}
            config = root / "config.json"
            config.write_text(json.dumps(configuration))
            with patch.object(guard, "gpu_state", return_value={"index": 0, "free_mb": 4096, "utilization": 0}), patch.object(guard.time, "sleep"):
                if exit_code:
                    with self.assertRaises(RuntimeError):
                        guard.main(config)
                else:
                    guard.main(config)
            state = json.loads((root / "run/status.json").read_text())
            self.assertEqual(state["exit_code"], exit_code)
            self.assertEqual(state["phase"], "failed" if exit_code else "training_complete_pending_validation")
            self.assertGreater(json.loads(budget.read_text())["used_active_training_seconds"], 0)
            with self.assertRaises(RuntimeError):
                guard.main(config)

    def test_success_retains_pending_validation_state(self):
        self.run_case(0)

    def test_failure_is_preserved_and_not_restarted(self):
        self.run_case(7)


if __name__ == "__main__":
    unittest.main()
