# Guarded 200m/256 comparison

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
