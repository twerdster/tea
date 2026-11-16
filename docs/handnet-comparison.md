# HandNet 200m/256 comparison

The completed **200m-sample/256-feature seed204 tree** scored **0.330230 mean
fingertip F1@15mm** on all **10,000 test frames**, versus **0.340204** for the
matched original seed201 tree. This recipe was **0.997 percentage points lower**.
The original three-tree forest remains at **0.374948** and stays the default.
This is **post-exposure exploratory engineering** after disclosure of the original
test results. The [aggregate JSON](evidence/handnet-200m-256-2026-11-16.json)
contains full-precision results, denominators and selected hashes.

`scripts/run_handnet_comparison.py` implements one additional seed 204 HandNet
tree with 200,216,884 sampled pixels, 256 frozen features and at most 20 split
decisions. It preserves the original three trees and their budget. It does not
implement a new three-tree batch, extra pilots, test evaluation, publication or
automatic feature deletion. Source availability does not establish that a run
started or completed; current authorization and operational evidence are private.

The recipe is fixed: the original tree0 bag's 181,979 ordered training-frame
identities, balanced cap 1,352 per frame (at most 169 per each of eight classes),
learned background, pixel seed 204 and original frame selector seed 201. The
4,096-byte coordinate asset is reused exactly, SHA256
`d30587695928cb350c274e6d7d91a472e1b6c27d8ffcfa54c6d76e43a81bf03c`.
Do not regenerate it from the seed: coordinate-generation arithmetic can differ
across platforms/versions. Actual Python/NumPy/SciPy versions and coordinate
hashes are recorded. Training uses frozen revision
`27ae728e9cb012d2255337d9b3583bbe7f66b3cc`, folding 12, 16 thresholds and
`maxDepth=19`. The maintained trainer's internal feature pool remains 8.

## Private configuration and review receipts

The CLI accepts `--runtime-config`, `--authorization`, `--source-review` and
`--monitor-ack`, with mutually exclusive `--check` and `--run` modes. Omitted
mode means a read-only check. Missing authorization refuses before runtime
context, directories, locks or commands. A check also requires the configured
own service and all launch receipts; it is not an authorization bypass.

Keep these JSON inputs outside public source. The strict runtime schema is
`tea-handnet-comparison-runtime-v1`, with these fields:

| Field | Meaning |
| --- | --- |
| `project_root` | Existing verified native filesystem workspace |
| `training_repo` | Frozen training checkout within that workspace |
| `proposal_dir` | Immutable recipe/coordinate/protocol inputs within the workspace |
| `hostname`, `username` | Exact intended Linux execution identity |
| `python_executable` | Canonical installed interpreter path |
| `gpu_lock` | Existing cooperative GPU lease, opened read-only by the frozen guard |
| `service_name` | Own bounded user systemd service |
| `worker_path` | Canonical path of this reviewed published worker |
| `monitor` | Collector-owned `tea-handnet-comparison-monitor-v1` configuration |

All paths are explicit absolute normalized paths, verified canonical before
mutation. The published worker can live in a separate checkout from the frozen
training code. Both source-review and monitor receipts bind the exact worker
SHA256 **and complete private runtime-config SHA256**. The worker rechecks its
bytes, runtime-config file and frozen inputs during supervision. Never replace
the frozen checkout with a newer source tree to satisfy publication review.

Authorization must explicitly record one additional seed 204 run, all three
original final slots still counted, a total-four slot ceiling, no extra pilots,
at most 16h new active training and the original 48h aggregate ceiling. It must
reference the user's instruction. A pending or unsupported batch scope refuses.
The published-source receipt records the exact source revision/URL, worker hash
and completed review. The monitor receipt records an operational integration
review, matching paths and a fresh verified ACTIVE two-hour schedule. Receipt
booleans do not substitute for that operational review.

`scripts/collect_handnet_comparison.py` is the read-only companion collector.
Its config/report binds the same runtime JSON, source inventory, worker and
run. Before leaving training unattended, verify its actual service/PID/start,
progress/log/checkpoints, numerical outputs, GPU lease, memory and disk checks.
The original campaign-only monitor does not cover this additional worker.

## Accounting and failure handling

The original ledger is unchanged: 11 settled runs totaling 9,012.027812974062
active seconds, under the original 48h ceiling. The new residual ledger begins
at zero with a 163,787.97218702594-second ceiling. Only the frozen generic guard
settles the new charge. Do not copy old charges into the new ledger or add the
same guard elapsed time again. Monitoring aggregates old settled + new settled
+ distinct new unsettled live guard charge, each once. Resource waits release
the GPU lease and are not active training time.

An own controller lock spans conversion through selection. Exclusive output
creation is a durable single-attempt reservation. Existing output, unknown
reservation, residual ledger, unsettled spec/status, partial bank or unknown
empty run/data directory blocks reuse and requires reconciliation. The worker
never retries automatically or signals unrelated processes.

The own service must enforce 24GiB memory, `KillMode=control-group` and at least
60s stop timeout. SIGTERM/INT follows the failure path; Runner grants 35s for
the frozen guard to stop its separately-sessioned trainer and settle. SIGKILL,
host loss or an uninterruptible child can bypass cleanup/accounting. Preserve
process/start identities and charge evidence, mark accounting unresolved and
diagnose before recovery. Never infer zero charge from a missing settlement.

Initial staging requires at least 77.950845GiB free on the verified native
filesystem; command supervision requires 16GiB reserve. Conversion and each CPU
validation have 6h wall ceilings. Guard resource waiting has no wall timeout
equal to the active-training budget. Features and checkpoints remain in place
after success or failure. Verified retained-artifact SSD backup and monitor
acknowledgment must precede any separately designed cleanup.

## Validation and reporting

All six predeclared settings must complete: 12/16/20 decisions crossed with
cutoff 0.25/0.5, foreground none, minimum support 3, first five finger classes and
all pixels on the same 300 validation frames. All three original training bags
and the complete new bag participate in identity leakage checks. Select
`max(mean fingertip F1@15mm, -decisions, cutoff)`; partial grids cannot win.
Counts are bounded integers, scorable detections use sensor_pos_detected, and
absent-finger false positives remain in the denominator. The macro average has
five entries; an undefined finger denominator contributes 0.

Bank verification checks full lengths, labels, poses and pixel bounds, plus at
most 1,024 distributed raw samples across all 256 feature columns. Every saved
level is checked for structure and finite positive normalized posteriors;
sampled external tree traversal must match saved labels. This does not claim
independent regeneration of every feature byte. A shallower selected prefix is
exported with final postprocessing and fresh manifest hashes.

Independent raw-groundtruth/position metric audit, actual selected-model native
parity, verified backup and report review remain required after completion.
Earlier original test outcomes are already disclosed: this comparison is
**post-exposure exploratory engineering**, not independent confirmation. The
worker never accesses the test split or publishes a result.

The CPU suite uses temporary directories, tiny synthetic MAT/native models and
short fake commands. Run:

```sh
python -m unittest discover -s tests -p 'test_handnet_comparison*.py' -v
```

These tests establish control/contract behavior;
they do not establish real training, native parity, runtime sufficiency or
monitor readiness. A test-only `TEA_TEST_SCRIPTS` override can select the frozen
converter/evaluator directory for the synthetic artifact checks.

## Completed training and validation selection — 2026-11-16

One additional eight-class tree completed **200,216,884 balanced pixel samples**,
**256 saved features** and **20 split decisions**. Its primary comparator is
original **seed201/tree0**, using the same ordered bag of **181,979 training
frames**. The original best single, seed202, and three-tree forest are separate
secondary references.

| Recipe | New seed204 | Matched original seed201 |
| --- | ---: | ---: |
| Training frames | 181,979 | 181,979 |
| Actual sampled pixels | 200,216,884 | 84,759,330 |
| Signed-int8 features per sample | 256 | 1,200 |
| Balanced cap per class per frame | 169 | 64 |
| Total sample cap per frame | 1,352 | 512 |
| Frame selector seed | 201 | 201 |
| Coordinate and pixel seed | 204 | 201 |
| Split decisions | 20 | 20 |

Both use eight classes including background, probe size 25,000 pixel·mm,
16 split thresholds, folding depth 12 and the frozen Tea `27ae728`
implementation. The new feature-column payload is **51,255,522,304 bytes**
(51.26 GB decimal / 47.74 GiB), before labels and other metadata. This changes
sample count, feature count and coordinate/pixel seeds together; it does not
isolate a causal effect of sample count.

The same frozen **300 validation frames** were used for six settings per model:
12/16/20 decisions crossed with cutoffs 0.25/0.5. Independent raw-reference
arithmetic auditing covered all **12 configurations**. Both selected **20
decisions, cutoff 0.25, support 3 and foreground `none`** using the declared tie
rule. Selection-subset mean fingertip F1@15mm was **0.333550** for seed204 and
**0.331467** for seed201. These settings and assets were frozen before the new
full-cohort evaluations; earlier original test outcomes were already disclosed.
The comparison remains **post-exposure exploratory engineering**.

## Full-cohort results and independent audit

Both frozen models completed all **2,773 validation frames** and **10,000 test
frames** with unchanged settings. Full validation is descriptive and includes
the 300 selection frames. All four training bags, full validation and test
identities were checked; no train/validation/test overlap was found. The four
training bags contain 202,010 unique training-frame identities in total.

| Cohort | Frames | New seed204 F1@15mm | Matched seed201 F1@15mm |
| --- | ---: | ---: | ---: |
| Validation used for selection | 300 | 0.333550 | 0.331467 |
| Full validation, descriptive | 2,773 | 0.337228 | 0.347738 |
| Test, frozen settings | 10,000 | **0.330230** | **0.340204** |

The new recipe scores lower on full validation and test, including lower test F1
for every finger. No model or setting was changed after these results. This
does not establish a sample-count effect: features and seeds changed too, and
the comparison was conducted after the original test disclosure. It does not
establish statistical significance or a best-possible Tea result.

| Test measure | New seed204 | Matched seed201 |
| --- | ---: | ---: |
| Mean fingertip recall within 15 mm | 34.15% | 35.18% |
| Detected/scorable fingertip errors | 46,768 | 46,768 |
| Mean detected-finger error | 48.04 mm | 44.26 mm |
| Median detected-finger error | 18.98 mm | 18.60 mm |
| 90th percentile detected-finger error | 88.98 mm | 79.41 mm |
| Seven hand-class mean IoU | 23.46% | 23.10% |
| Learned foreground precision | 64.30% | 64.51% |
| Learned foreground recall | 94.65% | 95.23% |
| Learned foreground IoU | 62.04% | 62.49% |
| Predictions on absent-finger occasions | 3,232 / 3,232 | 3,232 / 3,232 |

Both models predict a centre for all 46,768 scorable present fingers and all
3,232 absent-finger occasions. Detection coverage therefore does not establish
reliable presence detection. Error summaries condition on detected/scorable
fingers; missing predictions, wrong locations and absent-finger false positives
remain in the F1 denominator. For each of the five fingers:

```text
F1@15mm = 2 × hits within 15 mm
          / (scorable presence + scorable-present predictions + absent predictions)
```

The reported score averages these five values. Ground truth is HandNet's
adjusted sensor centres `pos[:3, :5]` in camera millimetres; predictions use the
largest confident 4-connected class component and its probability-weighted
centroid. Classification uses all positive-depth pixels, with raw label 0
learned as background. These operational results do not reproduce the historical
paper's mean-shift detector or metric. Balanced votes are not established as
calibrated probabilities. Official randomized frames do not establish subject
or session independence.

For context, the separately published original forest scores **0.374948** and
its validation-selected seed202 single scores **0.327988** on the same test
cohort. Their settings, source, metric and exact cohort order were checked for
reuse. These are secondary references; the matched primary baseline here is
seed201, and the original validation selection is preserved.

Independent raw-reference audits reconstructed identities, ground-truth supports,
visibility, counts, localization hits/errors, F1 and classification count
conservation. All inputs and outputs remained hash-stable and finite. This audit
uses supplied predictions; it does not independently rerun inference or
reconstruct predicted components. Image-feasibility checks allow **1e-8 pixel**
numerical slack for float64 centroid roundoff; original coordinates and scoring
values are unchanged. Selected-model native inference checks are separate below.

## Selected artifact and native CPU integration

The optional [seed204 release](https://github.com/twerdster/tea/releases/tag/handnet-200m-256-seed204-2026-11-16)
uses a separate tag. It preserves the original forest release and default
integration. Its selected manifest declares the exact feature profile,
coordinates, calibration and postprocessing defaults.

| Artifact | SHA256 |
| --- | --- |
| [Seed204 ZIP](https://github.com/twerdster/tea/releases/download/handnet-200m-256-seed204-2026-11-16/tea-handnet-200m-256-seed204.zip) | `53b8fc95c5c8bc3f24c191e5ffda8a80787f64635e45d9e1a16c9dfbdf9006c6` |
| Public manifest | `606237ac0db02df7ca4f0c04d7bdc6855edece0f7b30927e3495bc473b331a5d` |
| Native model | `22c5f0763282d8cb816b021cc10f370c0ad398c1aa0e26fe83d4fb34ec67f8df` |
| Coordinates | `d30587695928cb350c274e6d7d91a472e1b6c27d8ffcfa54c6d76e43a81bf03c` |

Verify [SHA256SUMS.txt](https://github.com/twerdster/tea/releases/download/handnet-200m-256-seed204-2026-11-16/SHA256SUMS.txt)
before unpacking. The ZIP contains only `forest.json`,
`assets/tree000/model.tree` and `assets/tree000/Coords.co`.
It includes HandNet attribution and dataset terms in the manifest; original
archives, raw frame identities and private runtime records are excluded.
HandNet is credited to Aaron Wetzler, Ron Slossberg and Ron Kimmel under its
[dataset release](https://huggingface.co/datasets/twerdster/HandNet) terms.
Dataset CC BY 4.0 does not relicense historical MATLAB code or assign a blanket
software/model license.

The actual selected model passed Linux/macOS native/Python checks on the same
**three training depth frames**, in both `none` and `near-component` modes.
Labels, posterior bytes, heaps and masks matched exactly. The maximum observation
difference was **4.760636329592671 × 10⁻¹³ mm**; complete tracker states/history
agreed with maximum numeric difference **4.760636329592671 × 10⁻¹³** across mixed
units. Linux/macOS native outputs agreed exactly. Synthetic **0.1 s timestamps**
check numerical integration, not measured chronology, tracking accuracy or
latency. Deep Metal/mobile/FPGA execution is not established.
See the [DepthTrack guide](https://github.com/twerdster/depthtrack/blob/main/docs/tea-forest.md)
and [parity aggregate](https://github.com/twerdster/depthtrack/blob/main/docs/tea-handnet-seed204-validation.json).
