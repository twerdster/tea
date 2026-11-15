# Deep trees and three-tree HandNet forests

The repaired trainer has independent CPU/GPU regression coverage, including the
full dense allocation and serialization path for **20 split decisions**. This
establishes trainer correctness on generated data. Real HandNet performance must
be measured with held-out predictions.

## Depth, diversity and split search

Tea CLI `maxDepth=d` trains levels `0..d` inclusive: use **19 for 20 decisions**.
It allocates 1,048,575 split nodes and 2,097,151 histogram slots. `foldingDepth`
limits the histogram batch size, not inference depth. Accepted bounds are
`0 <= foldingDepth <= maxDepth <= 20`.

One native invocation trains one tree. `treeNum` selects the output filename;
changing it alone does not deliberately diversify a forest. Prepare three
reproducible feature banks/pixel samples with independent coordinate and sampling
seeds, record each tree's source frame manifest and preserve its `Coords.co`.
Use sequential runs in separate output directories.

Every supplied feature is tested at every level. Thresholds follow the configured
deterministic grid; there is no per-node RNG, bootstrap option, minimum-leaf-size
or pruning option. More features/thresholds cost more computation and do not
promise better held-out accuracy. Select settings within the experiment budget.
Class-balanced sampling changes training priors; evaluate on the natural held-out
pixel distribution. `W_APRIORI` weights split entropy by inverse class frequency;
`W_FILE` supplies positive integer divisors. Leaf probabilities remain raw class
counts. Weighted entropy is normalized before taking logarithms so deep nodes
with very small inverse-frequency mass remain numerically meaningful.

## Prepare a bounded pilot

First run the build and correctness suite:

```bash
make -j2 Tea CUDA_ARCH=sm_86
make trainer-kernels CUDA_ARCH=sm_86
python tests/trainer_regression.py --tea ./Tea
python tests/trainer_depth20.py --tea ./Tea
```

Choose the CUDA architecture supported by the installed toolkit. `sm_86` was used
for the RTX 4090 checks because that host's compiler did not recognize `sm_89`.
The tests use generated data and do not download HandNet or launch a real run.

Convert a bounded pilot with explicit [feature units, labels and seeds](handnet.md).
Compare probe scales 50/5000/25000/50000 as validation hypotheses; the historical
50 setting and newer 50000 default are not evidence of the best scale. Reuse
frame identities and sampled pixels for paired scale comparisons.

Read the actual sample count only after successful conversion:

```bash
python - <<'PY'
import json
metadata = json.load(open('/work/tea/pilot-25000/handnet_metadata.json'))
assert metadata['complete']
print(metadata['samples'], metadata['label_hist'])
PY
```

Estimate device memory before acquiring the project's GPU lease:

```bash
python scripts/estimate_training_memory.py \
  --dataset /work/tea/pilot-25000 --max-depth 19 --folding-depth 8 \
  --feature-type F_CHAR --vram-cap-mb 1024 --require-fit
```

Inside the authorized GPU queue/lease, a native 20-decision invocation is:

```bash
./Tea 1200 8 19 8 F_CHAR ACTUAL_SAMPLE_COUNT 32 0 0 W_ONES \
  /work/tea/pilot-25000/ pilot25000 0 LOG0 "bounded HandNet pilot"
```

Use the project's bounded supervisor for unattended work:
`scripts/run_guarded_training.py /path/to/run-spec.json`. Its specification
records the exact argv/cwd/output directory, existing cooperative GPU lock,
per-run timeout, available-VRAM threshold and shared training budget file. A busy
lease/GPU is a monitored resource wait. Never stop or crowd an unrelated job.
The legacy `DROP_CACHE` variable is ignored; the trainer never changes global
filesystem caches. Keep the run ledger, process identity, logs, checkpoint paths, authorized budget
and periodic monitor active through training, recovery and final evaluation.

## Memory and storage

For **100,000,000 samples, eight classes, `F_CHAR`, pool size 8, one GPU worker and
20 decisions (CLI19)**, the current native allocation formulas give:

| Component | Allocation |
| --- | ---: |
| Persistent host arrays/pool/nodes | 1,740.61 MiB |
| Host peak while exporting a snapshot | 2,035.35 MiB |
| Device buffers, folding depth 8 | 355.12 MiB |
| Device buffers, folding depth 12 | 355.35 MiB |
| Device buffers, folding depth 19 | 387.10 MiB |
| Final native tree | 40.00 MiB |

These are array allocations, not process RSS or total GPU use. Add CUDA context,
allocator, thread, logger and I/O-cache overhead and leave working headroom. The
current Python estimator assumes a conservative 48-byte training node; the native
layout is 24 bytes on the tested build. With its 32 MiB safety margin, estimator
results for the three folding depths are approximately 435.12/435.35/467.10 MiB.
Host memory is not included in that estimator.

The host calculation includes two 400 MB index arrays, a 200 MB sample array,
eight 100 MB pinned feature columns and the node array. Snapshot peak adds the
40 MiB model, 64 MiB full uint32 histograms and a 200 MB output-label array. The
36 MiB reduction scratch is released before snapshot export.

On-disk features scale as `samples * features * bytes_per_feature`: 100 million
samples and 1200 `F_CHAR` features require 120 GB decimal (111.76 GiB) before
labels/poses/pixels and other metadata. Verify task-scoped storage and quotas
before conversion. Feature banks can be regenerated from frozen source identities,
coordinates and seeds; preserve models/manifests/results when reclaiming an
explicitly disposable bank. Do not delete source data or active-run files.

## Atomic checkpoints and recovery

Each completed level writes an immutable `Tree_NNNN.depthDD.tree` and atomically
replaces `Tree_NNNN.tree`. The latter's `.labels` file contains training predictions
and is separately published atomically. A `.tmp` is never a completed checkpoint.
`treePrefix` selects the log filename, not the model filename.

Depth files are inference checkpoints, not resumable trainer state. Existing
completed depth files cause the trainer to refuse overwrite. Preserve failure
logs/status and use a new run directory for a diagnosed replacement. Verify the
replacement before unattended execution and account for consumed training budget.

The final tree includes intermediate histograms. `tea_model.py evaluate
--max-splits N` can evaluate a shorter prefix without retraining, with `N` counting
actual decisions. Compare saved-depth models/prefixes on validation to decide
whether deeper trees help or overfit. Do not select depth with final test data.

## Freeze and evaluate the forest

Create a `depthtrack-tea-forest-v1` manifest with relative per-tree model/coordinate
paths, checksums, positive aggregation weights, split manifests, seeds and the
complete float32 feature profile. Eight-class models declare learned background
class 7 and invalid-centre label 8. Record image dimensions/intrinsics and preserve
the provenance of their calibration. Validate with:

```bash
python scripts/tea_model.py inspect --manifest /models/forest.json
```

Each reached byte histogram is normalized by its own sum. The forest averages
these probabilities using the declared weights; raw byte totals and `maxVal`
do not give a tree extra weight. Repaired empty leaves already inherit a populated
ancestor posterior; the reference's malformed/legacy empty fallback is the root
prior. Historical models with shifted histogram conventions are kept separate.

Evaluate individual trees and their three-tree combination on the same frozen
validation frames, foreground policy and pixel domain. Pass all training frame
manifests to overlap checks. Report hand-class macro IoU, per-class metrics,
foreground errors and fingertip localization recall/coverage, including missing
predictions. Official frame splits do not establish participant independence.
After selecting configuration and localization thresholds, evaluate the untouched
test split once for final reporting. Archive the manifest, evaluation command,
metrics, overlays and failure cases alongside the selected checkpoints.
