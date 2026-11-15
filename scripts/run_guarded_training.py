#!/usr/bin/env python3
"""Run one bounded Tea configuration under a cooperative GPU lease.

The JSON specification contains argv, cwd, output_dir, max_seconds,
gpu_lock (an existing lock file), min_free_vram_mb, and max_total_seconds.
The durable status is usable during a resource wait as well as training.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def gpu_state():
    command = ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    rows = [[int(v.strip()) for v in line.split(",")] for line in result.stdout.splitlines()]
    if not rows:
        raise RuntimeError("GPU telemetry unavailable")
    return {"index": rows[0][0], "free_mb": rows[0][1], "utilization": rows[0][2]}


def main(spec_path):
    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    spec = json.loads(spec_path.read_text())
    out = Path(spec["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path, log_path = out / "status.json", out / "training.log"
    if status_path.exists():
        raise RuntimeError("Run status already exists; inspect and preserve evidence before recovery")
    if not isinstance(spec["argv"], list) or not spec["argv"]:
        raise ValueError("argv must be a nonempty argument list")
    max_seconds = float(spec.get("max_seconds", 57600))
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 57600:
        raise ValueError("Invalid per-run budget")
    total_path = Path(spec["budget_file"])
    budget = json.loads(total_path.read_text())
    remaining = float(budget["max_active_training_seconds"]) - float(budget.get("used_active_training_seconds", 0))
    if remaining <= 0:
        raise RuntimeError("Training batch budget exhausted")
    max_seconds = min(max_seconds, remaining)
    state = {"schema_version": 1, "phase": "resource_wait", "pid": os.getpid(),
             "created_at": time.time(), "updated_at": time.time(), "spec": str(spec_path.resolve()),
             "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
             "argv": spec["argv"], "cwd": spec["cwd"], "log": str(log_path),
             "tea_log": spec.get("tea_log"),
             "active_training_seconds": 0, "max_seconds": max_seconds,
             "gpu_lock": spec["gpu_lock"], "latest_checkpoints": []}
    child = None
    acquired = False
    training_start = None
    lock = open(spec["gpu_lock"], "r")  # Never change the cooperative lock's contents.
    try:
        write_json(status_path, state)
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                acquired = False
            telemetry = gpu_state()
            state.update(updated_at=time.time(), gpu=telemetry)
            state["wait_reason"] = "cooperative GPU lease busy" if not acquired else "GPU resources busy"
            if acquired and telemetry["free_mb"] >= spec.get("min_free_vram_mb", 1536) and telemetry["utilization"] <= spec.get("max_idle_utilization", 30):
                break
            if acquired:
                fcntl.flock(lock, fcntl.LOCK_UN)
                acquired = False
            write_json(status_path, state)
            time.sleep(30)

        # Another sequential run may have consumed budget while this one waited.
        budget = json.loads(total_path.read_text())
        remaining = float(budget["max_active_training_seconds"]) - float(budget.get("used_active_training_seconds", 0))
        if remaining <= 0:
            raise RuntimeError("Training batch budget exhausted during resource wait")
        max_seconds = min(max_seconds, remaining)
        state["max_seconds"] = max_seconds
        training_start = time.monotonic()
        state.update(phase="training", started_at=time.time(), wait_reason=None)
        with log_path.open("x", buffering=1) as log:
            environment = os.environ.copy()
            environment.pop("DROP_CACHE", None)
            child = subprocess.Popen(spec["argv"], cwd=spec["cwd"], stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True, env=environment)
            state["child_pid"] = child.pid
            try:
                state["child_start_ticks"] = Path(f"/proc/{child.pid}/stat").read_text().split()[21]
            except FileNotFoundError:
                state["child_start_ticks"] = None
            while True:
                elapsed = time.monotonic() - training_start
                state.update(updated_at=time.time(), active_training_seconds=elapsed)
                if elapsed >= max_seconds:
                    state["limit_reached"] = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise RuntimeError("Active training time limit reached")
                state["latest_checkpoints"] = [{"name": p.name, "bytes": p.stat().st_size,
                                                 "modified_at": p.stat().st_mtime}
                                                for p in sorted(Path(spec["checkpoint_dir"]).glob("Tree_*.tree"))]
                write_json(status_path, state)
                code = child.poll()
                if code is not None:
                    state["exit_code"] = code
                    if code:
                        raise RuntimeError(f"Tea exited with status {code}")
                    break
                time.sleep(5)
        state.update(phase="training_complete_pending_validation", updated_at=time.time())
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state.update(phase="failed", error=repr(error), updated_at=time.time())
        raise
    finally:
        try:
            if training_start is not None:
                elapsed = time.monotonic() - training_start
                state["active_training_seconds"] = elapsed
                budget["used_active_training_seconds"] = float(budget.get("used_active_training_seconds", 0)) + elapsed
                budget.setdefault("runs", []).append({"spec_sha256": state["spec_sha256"], "seconds": elapsed,
                                                      "phase": state["phase"]})
                try:
                    write_json(total_path, budget)
                except BaseException as error:
                    state.update(phase="failed", error=repr(error), accounting_failed=True, updated_at=time.time())
                    raise
        finally:
            try:
                write_json(status_path, state)
            finally:
                try:
                    if acquired:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                finally:
                    lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    main(parser.parse_args().spec)
