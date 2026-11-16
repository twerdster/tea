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

## Preparing a public model candidate

Use `scripts/prepare_forest_release.py` with the selected deployment manifest, a
new output directory, explicit `--training-source-revision` and a reviewed
`--selection-metadata` JSON file. For the active campaign the training revision is
`27ae728e9cb012d2255337d9b3583bbe7f66b3cc`; later packaging/documentation commits do
not change it. The CLI help defines the strict safe metadata schema for validation
selection, probe size, frame fraction and ordered per-tree seeds/counts/depth.

The preparer copies only model and Coords assets into relative release paths,
recomputes hashes, drops nested private runtime provenance and reconstructs known
image/feature/class/postprocessing fields. It refuses existing outputs, traversal,
asset symlinks, unsafe reviewed metadata and oversized inputs. Do not weaken these
checks to publish a private manifest. Preserve the private source and its hash in
the separate release verification record. Missing postprocessing declarations stay
missing; explicit selected defaults, class order and unnormalized tree weights
must retain their exact meaning.

Run `python tests/test_forest_release.py -v` after preparer changes. Its synthetic
checks establish sanitization and inference equivalence only. Before uploading any
candidate, separately verify the actual selected model on real depth frames against
native DepthTrack, validate declared observation defaults, review attribution/data
terms and all public files, and record a fresh final public-manifest hash. This utility performs no upload and does not substitute for required validation or
the user's existing task authorization. It does not establish trained-model accuracy.

## Pilot results workflow — 2026-11-15

`docs/handnet-pilots.md` and `docs/evidence/handnet-pilots-2026-11-15.json` record the
completed eight pilots and 48 evaluations on one frozen 300-frame validation subset.
Keep their scores linked to the frozen training source `27ae728`. The selected
class mode/scale is eight classes with background and probe 25000; 20 decisions and
cutoff 0.5 are pilot settings, not the final forest's frozen settings. Re-evaluate
final depth/cutoff and single-tree versus ensemble choice after all three full-size
trees exist, before opening test data.

Maintain the distinction between seven-class depth-ROI inference and eight-class
learned-background inference, their pixel domains, fingertip F1 and segmentation
metrics. Do not present modest pilot scores or three-training-frame numerical CPU
parity as full-size/test accuracy, temporal accuracy or latency. Public aggregates
must omit private paths, host/account details and runtime ledgers. Update this
report only from verified model/evaluation evidence; keep final result claims
pending until the final selected models have been evaluated.

## Completed forest results workflow — 2026-11-16

The original three-tree campaign and both frozen 10,000-frame test evaluations
completed. `docs/handnet-forest-results.md` and
`docs/evidence/handnet-forest-2026-11-16.json` report the audited result: forest
F1@15mm 0.37494755678731473 versus the validation-selected seed 202 single
0.32798765233655836, using 20 decisions/cutoff 0.25/minimum support 3/foreground none.
Keep the 300-frame selection subset separate from the descriptive full 2,773-frame
validation cohort, which includes it. Preserve the earlier pilot report as dated
history; do not replace its cutoff 0.5 with the later final cutoff.

Final aggregation must retain hit/presence/detection/absent false-positive denominators,
the adjusted sensor-position reference semantics, the natural positive-depth
classification domain and modest localization/foreground quality. Raw-reference
metric arithmetic checks are separate from classifier inference validation. Exact
selected-model CPU parity on three training frames with synthetic 0.1s timestamps
does not establish tracking accuracy, latency or Metal/FPGA/mobile support.

Model downloads use the `handnet-forest-2026-11-16` release and its two reviewed ZIP
hashes; verify fresh public manifests after sanitization rather than reusing a
prefix-verification hash recorded before postprocessing fields were added. Keep
training revision 27ae728 explicit and separate from packaging/documentation commits.
Review attribution/data terms and public contents before publication. Private
runtime ledgers, raw frame identities and machine paths stay excluded.

The optional 200m/256 recipe remains unlaunched pending authorized run-count
reconciliation. Its prospective recipe is preserved, but comparisons after
original-test outcome disclosure are post-exposure exploratory engineering.

## Documentation maintenance and upstream merges

Keep the README's requirements and first-run commands concise; use the guides for
advanced build/CLI/conversion workflows. Preserve upstream operator explanations,
source notices, attribution and schematic figures while reconciling claims with
verified current evidence. No standalone software license is assigned by a docs
change. Check diagram labels/captions as well as Markdown links.

Separate source/assertion inspection, generated fixtures, executed regressions,
held-out application results and benchmark timing. Rebuild MkDocs after docs
edits; do not start training or benchmarks for prose-only changes. Record checks
that were not rerun without erasing the evidence of independently verified prior
runs. Current coverage includes deep/weighted native regressions and real HandNet
held-out results; it does not establish every historical configuration, multiple
GPU execution, historical MATLAB parity or a paper-metric reproduction.
