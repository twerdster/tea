# Tea

[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA-76B900?logo=nvidia&logoColor=white)](TreeBuilder.cu)
[![C++11](https://img.shields.io/badge/C%2B%2B-11-00599C?logo=c%2B%2B&logoColor=white)](CMakeLists.txt)
[![CMake](https://img.shields.io/badge/build-CMake-064F8C?logo=cmake&logoColor=white)](CMakeLists.txt)

Tea trains a randomized decision tree from precomputed feature columns. CUDA workers accumulate
class histograms and score candidate splits; the host selects splits, reorders sample indices
and writes tree snapshots and predicted labels. The related BMVC 2015 paper [*Rule of Thumb: Deep Derotation for Improved Fingertip Detection*](https://arxiv.org/abs/1507.05726)
is by Aaron Wetzler, Ron Slossberg and Ron Kimmel. This repository includes tools for converting
that work's HandNet data into Tea feature columns.

![Schematic of feature columns, CUDA split evaluation, host selection and tree outputs](docs/images/tea-training-pipeline.svg)

*Implementation schematic. It contains no measured throughput or accuracy result.*

The completed [HandNet forest report](docs/handnet-forest-results.md) records
**0.374948 mean fingertip F1@15mm** for the three-tree ensemble versus **0.327988**
for the validation-selected single tree on all **10,000 official test frames**.
Both use 20 split decisions, 1,200 int8 features, cutoff 0.25 and minimum component
support 3. The report includes denominators, limitations and
[selected model downloads](https://github.com/twerdster/tea/releases/tag/handnet-forest-2026-11-16).

## Current implementation

The source uses C++ standard-library threading, Make and CMake build paths, and Python tools for
data generation, inspection, expansion, HandNet conversion and bounded synthetic benchmark runs.
These are implemented workflows, not a claim that all historical configurations have been validated.
Independent CPU/GPU regressions cover routing, histogram tails, weighted split scoring,
serialization and a bounded full 20-decision tree. The fast smoke suite still consists of two
shallow cases; the separate converter smoke uses four synthetic MAT frames. Real-data results
come from the separately audited held-out HandNet campaign, rather than those generated checks.

Tea reads a directory containing:

- `Labels.lbl`: one binary `uint16` class ID per sample.
- `Threshholds.thr`: three binary `float32` threshold metadata values per feature. The spelling is
  part of the file format.
- `F_0000.feat`, `F_0001.feat`, …: one binary feature column per file, with datatype selected by the CLI.

A run writes `Tree_0000.tree`, `Tree_0000.tree.labels` and `LogFile_<prefix>_0000.log` to the dataset
directory for tree number 0. Current snapshots publish atomically as training advances; completed
`Tree_0000.depthDD.tree` files are immutable inference checkpoints, not resumable build state.
CLI depth 19 produces 20 split decisions.
See [dataset format](docs/dataset-format.md) and [CLI/output details](docs/cli-and-outputs.md).

## Requirements and first run

The documented training target is Linux with an NVIDIA GPU with a working driver, the CUDA toolkit (`nvcc`) and a
C++11-capable host compiler. Python 3 drives the tests and data tools; CMake is optional. OpenMP
can be disabled. HandNet conversion additionally requires NumPy and SciPy, and `bsdtar` for RAR
input. This is not a CPU-only trainer.

```bash
bash scripts/check_env.sh
make Tea
make smoke
```

The Makefile defaults to `CUDA_ARCH=sm_86`; select the architecture for the installed GPU and
compiler, for example `make Tea CUDA_ARCH=sm_89`. Use `make Tea OPENMP_FLAGS=` to disable OpenMP.
The CMake alternative is `bash scripts/build.sh`, followed by
`python3 tests/run_smoke_tests.py --tea ./build-cmake/Tea`.

To generate and train a tiny root-only example (argument order is documented in
[CLI/output details](docs/cli-and-outputs.md)):

```bash
python3 tests/gen_smoke_dataset.py /tmp/tea-example-root --scenario root
./Tea 4 4 0 0 F_FLOAT 16 4 0 0 W_ONES /tmp/tea-example-root/ example_root 0 LOG1 "manual root example"
```

See [getting started](docs/getting-started.md) for compiler overrides and documentation setup,
[examples](examples/README.md) for CLI cases, [HandNet conversion](docs/handnet.md) for archive
preprocessing, and [benchmarking](docs/benchmarking.md) for CSV-producing synthetic sweeps.

## Validation scope

The [test reference](tests/README.md) separates shallow data-path checks, independent native
regressions and Python tooling tests. `make smoke` verifies known synthetic training labels at
depths 0 and 1. `make handnet-smoke` checks generated conversion layout and nonempty shallow
training outputs; it does not verify its predicted labels.

The repaired trainer also supports:

```bash
make trainer-kernels
python3 tests/trainer_regression.py --tea ./Tea
python3 tests/trainer_depth20.py --tea ./Tea
python3 -m unittest discover -s tests -p 'test_*.py'
```

The independent native expectations cover odd/empty branches, folded and streamed features,
weighted entropy, serialized posteriors and the full dense 20-decision layout. They establish
checked code behavior. The [final HandNet evaluation](docs/handnet-forest-results.md) separately
checks natural-distribution segmentation/localization with frozen validation selection and all
10,000 test frames. Training self-success and generated-data correctness are not held-out scores.

The training/evaluation source remains
[`27ae728`](https://github.com/twerdster/tea/tree/27ae728e9cb012d2255337d9b3583bbe7f66b3cc).
Later release packaging and documentation commits do not change the trained weights. This
upstream documentation merge rebuilt MkDocs; it did not rerun CUDA tests or launch training.

## Limitations

Features must be prepared before training; the core trainer does not extract image features.
Multiple-GPU execution and every historical numerical configuration are not established by the
reported regressions. The memory cap is an estimate, not a guarantee that all runtime allocations
fit. Use the [deep-training workflow](docs/deep-training.md) for bounds and verification.

The reported HandNet scores remain modest, with substantial localization and absent-finger false
positives. Official random-frame splits do not establish subject/session independence. Numerical
CPU parity on three training frames with synthetic timestamps establishes neither tracking
accuracy nor latency, and it does not validate deep Metal/mobile/FPGA inference. These results do
not reproduce the historical paper's mean-shift detector or metric. The original MATLAB/MEX
preprocessor has not been independently matched against the maintained Python feature contract.

## Documentation and attribution

The [MkDocs source](docs/index.md) describes the operator chain and workflows. Build it with
`bash scripts/build_docs.sh` after installing `requirements-docs.txt`; the existing GitHub Pages
workflow is configured to publish it. See [site maintenance](docs/docs-site.md) for details.

Existing author credits and source notices are retained. No standalone software license is
included in this checkout; these documentation changes do not assign one.
