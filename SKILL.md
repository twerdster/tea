# Tea maintenance and training workflow

Read SUMMARY.md, this file, README.md and docs/deep-training.md before changes.
Review and synchronize SUMMARY.md and SKILL.md before every commit.

Preserve source archives, historical models and unrelated workloads. Keep repos
and environments on a native filesystem; use verified task storage for large
ordinary data. Record actual paths, capacity checks, run authorization and budgets
in the private runtime handoff, not public model manifests. Never change global
caches, drivers, kernels or other jobs to make a training run fit.

Validate trainer changes against independent CPU split/histogram expectations:
odd and empty branches, vector tails, folded batches, feature streaming, tiny
weighted masses, large counts and serialized posteriors. `make trainer-kernels`
and `tests/trainer_regression.py` need CUDA; `tests/trainer_depth20.py` is a bounded
full-depth check. CPU tooling/supervisor checks use
`python -m unittest discover -s tests -p 'test_*.py'`. Runtime exit success alone
does not establish classifier correctness.

Freeze training frame identities, class mapping, sampling/coordinate seeds,
feature profile and model hashes. `treeNum` is a filename index, not diversity.
Train distinct trees from independent reproducible data/feature selections.
Use held-out validation for settings, depth and confidence thresholds; freeze
selection before opening test inputs. Include missed and spurious fingertips in
metrics. Background accuracy, training self-success, oracle segmentation and
detector-relative tracking residuals are not end-to-end fingertip accuracy.

Deployment needs exact per-tree Coords.co and feature arithmetic, normalized
probability aggregation, source calibration and explicit preprocessing. Use
verified native prefix export for a selected shallower tree. Independent foreground
masks restrict queried centres only; feature neighbors retain original depth.
Balanced-training votes are not proven calibrated probabilities. Test the native
DepthTrack adapter on identical inputs; do not substitute historical models or
claim deep Metal/mobile/FPGA execution from CPU checks.

Before unattended work, activate and verify the relevant recurring monitor and
record service/PID/start identity, actual Tea log, status, checkpoint paths,
expected cadence and budget. The guarded supervisor coordinates an existing GPU
lease and releases it during resource waits. Treat unreachable infrastructure as
monitoring failure, and expected resource waits explicitly. Preserve failure
evidence, diagnose and validate recovery; never blindly restart or duplicate a job.
Keep monitoring through recovery and final evaluation. Pause only after all watched
work completes and results are reported, or the user cancels.

Publish reviewed source, distributable selected model artifacts and safe provenance.
Exclude runtime ledgers, private machine paths and source data archives. Update the
DepthTrack integration and HandNet submodule deliberately after validation.
