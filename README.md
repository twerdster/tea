# Tea

[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++11](https://img.shields.io/badge/C%2B%2B-11-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-supported-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)

Tea is a CUDA/C++ decision-tree trainer for very large precomputed feature datasets. It was used for large-scale training in the BMVC 2015 paper [*Rule Of Thumb: Deep derotation for improved fingertip detection*](https://arxiv.org/abs/1507.05726) by Aaron Wetzler, Ron Slossberg, and Ron Kimmel. This repository now includes:

- a modernized host-side threading layer based on the C++ standard library
- a supported local build flow with both `make` and CMake
- independent CPU/GPU routing, histogram and serialization regressions, including 20-decision trees
- small generated example datasets and copy-pasteable example commands
- reproducible class-aware HandNet conversion, native forest inference and held-out evaluation

The maintained workflow covers buildability, native trainer correctness and reproducible dataset/inference contracts. Historical MATLAB tooling is replaced by Python conversion and evaluation utilities. Real model quality requires held-out evaluation; synthetic success is not an accuracy claim.

The completed [HandNet forest evaluation](docs/handnet-forest-results.md) reports
**0.374948 mean fingertip F1@15mm** for the three-tree ensemble versus **0.327988**
for the validation-selected single tree on all **10,000 official test frames**.
Both use 20 split decisions, 1,200 int8 features, cutoff 0.25 and minimum component
support 3. The report includes limitations, exact denominators and
[selected model downloads](https://github.com/twerdster/tea/releases/tag/handnet-forest-2026-11-16).

## Documentation

The repository now includes a multi-page documentation site under `docs/`, built with MkDocs.

Install the docs dependencies:

```bash
python3 -m pip install -r requirements-docs.txt
```

Serve the site locally:

```bash
bash scripts/serve_docs.sh
```

Build the static site:

```bash
bash scripts/build_docs.sh
```

The GitHub Pages workflow is defined in `.github/workflows/docs.yml`.

## Repository Status

The repository has been updated to build on a modern Linux toolchain. The active workflow in this repo is:

- build the `Tea` binary
- run the supported smoke tests
- use the small example datasets to verify shallow-tree training

Independent CPU/GPU regressions now cover odd/singleton routing, histogram tails, deep empty branches, streaming feature pools, weighted split entropy and serialized posteriors. A bounded nine-sample test exercises the full 20-decision dense model. See [the test workflow](tests/README.md). Exactly 20 decisions uses CLI `maxDepth=19`.

## Requirements

You need:

- Linux
- an NVIDIA GPU with a working driver
- the CUDA toolkit, including `nvcc`
- Python 3
- a C++11-capable host compiler
- CMake if you want the CMake-based build path
- NumPy and SciPy if you want to run the HandNet converter

OpenMP is optional. If it is available, Tea uses it for a small amount of host-side parallel work. If it is not available, the code still builds.

## Quick Start

Check the local toolchain and GPU visibility:

```bash
bash scripts/check_env.sh
```

Build with the Makefile:

```bash
make Tea
```

Run the supported smoke tests:

```bash
make smoke
```

That command:

- generates tiny test datasets under `/tmp`
- runs Tea on a depth-0 root-separable case
- runs Tea on a depth-1 balanced case
- verifies the expected output files and final predicted labels

## Build Options

### Makefile

Default build:

```bash
make Tea
```

Override the CUDA compiler:

```bash
make Tea NVCC=/path/to/nvcc
```

Override the CUDA architecture:

```bash
make Tea CUDA_ARCH=sm_89
```

Disable OpenMP if needed:

```bash
make Tea OPENMP_FLAGS=
```

Show available Makefile helpers:

```bash
make help
```

### CMake

Configure and build:

```bash
bash scripts/build.sh
```

By default, that builds into `build-cmake/`.

Override the CUDA architecture:

```bash
CUDA_ARCH=89 bash scripts/build.sh
```

You can also use raw CMake commands directly:

```bash
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cmake -j
```

Disable OpenMP in the CMake path if needed:

```bash
TEA_ENABLE_OPENMP=OFF bash scripts/build.sh
```

## Smoke Tests

The supported smoke-test runner is [tests/run_smoke_tests.py](tests/run_smoke_tests.py).

Run it directly:

```bash
python3 tests/run_smoke_tests.py --tea ./Tea
```

List the available supported cases:

```bash
python3 tests/run_smoke_tests.py --list
```

Or through the wrapper:

```bash
bash scripts/run_smoke.sh ./Tea
```

The smoke suite covers:

- `root-depth0`: a depth-0 dataset that is perfectly separable at the root
- `depth1-balanced`: a depth-1 dataset that needs one additional split

The runner checks:

- successful process exit
- expected output files exist
- final predicted labels match the known labels exactly
- the log reports `Depth X: 100.0000%` at the target depth

More detail is in [tests/README.md](tests/README.md).

## HandNet conversion and deep forests

The maintained converter supports frozen frame identities, reusable coordinates,
separate frame/pixel/feature seeds and class-aware sampling. An eight-class pilot
can learn positive-depth background as well as the seven hand classes:

```bash
python scripts/handnet_to_tea.py \
  --archive /data/HandNet/TrainData --output /work/tea/pilot-25000 \
  --split train --features 1200 --probe-size 25000 --seed 101 \
  --frame-seed 20261115 --max-files 2000 \
  --sampling balanced --include-background --samples-per-frame 512 \
  --sampling-seed 101 --sample-alignment 1 --max-output-gib 4
```

The historical probe 50 and scales 5000/25000/50000 are validation hypotheses;
50000 is now the converter default, not an established best setting. Depths are
millimetres and coordinates pixel·millimetres. Conversion and inference share
float32 complete-sum coordinate arithmetic, truncation toward zero and the
`ceil(clamp((d1-d2)/250*127,-127,127))` feature definition. Reuse each tree's exact
`Coords.co` across splits and deployment.

Tea trains one tree per invocation. CLI depth 19 means 20 split decisions;
`treeNum` is a filename index, not a random seed. Three diverse trees require
independently seeded banks/samples and a frozen forest manifest. Use the
[deep-training workflow](docs/deep-training.md) for memory bounds, GPU queueing,
atomic checkpoints and held-out model selection. The [HandNet guide](docs/handnet.md)
details class semantics, natural-distribution evaluation and localization metrics.

```bash
python scripts/tea_model.py inspect --manifest /models/forest.json
python scripts/tea_model.py evaluate \
  --manifest /models/forest.json --archive /data/HandNet/ValidationData \
  --split validation --output /results/validation.json --localize --foreground none \
  --training-manifest /work/tea/pilot-25000/frames.json
```

Trainer correctness checks are separate from the completed
[held-out forest evaluation](docs/handnet-forest-results.md). The earlier depth 8
modernization result matched the dominant class and predates native repairs.

## Dataset Utility

[scripts/expand_dataset.py](scripts/expand_dataset.py) can inspect and derive Tea-format datasets.

Inspect an existing dataset:

```bash
python3 scripts/expand_dataset.py inspect /path/to/data single
```

Write a derived dataset:

```bash
python3 scripts/expand_dataset.py expand /path/to/data single /tmp/tea-expanded 1000 200 7 single
```

That utility can:

- tile samples to a new sample count
- change feature file datatype across `int8`, `int16`, `int32`, and `single`
- expand feature count by repeating feature columns
- derive new label distributions for quick benchmarking

## Benchmarking

Use [scripts/benchmark.py](scripts/benchmark.py) to generate bounded synthetic datasets, run Tea over sample/feature/depth sweeps, and write a CSV summary.

Example:

```bash
python3 scripts/benchmark.py \
  --tea ./Tea \
  --work-dir /tmp/tea-bench-run \
  --samples 250000,1000000,2000000 \
  --features 32,128,256 \
  --depths 1,2,4,6 \
  --feature-type F_CHAR \
  --vram-cap-mb 500
```

That script:

- generates synthetic Tea datasets directly in columnar `F_XXXX.feat` format
- estimates Tea's per-device VRAM requirement before each run
- refuses any case that exceeds the configured cap
- writes results to `results/benchmark_results.csv`

## Manual Examples

Additional copy-pasteable examples live in [examples/README.md](examples/README.md).

### Root-only Example

Generate a tiny dataset:

```bash
python3 tests/gen_smoke_dataset.py /tmp/tea-example-root --scenario root
```

Train a depth-0 tree:

```bash
./Tea 4 4 0 0 F_FLOAT 16 4 0 0 W_ONES /tmp/tea-example-root/ example_root 0 LOG1 "manual root example"
```

### Depth-1 Example

Generate a shallow two-level dataset:

```bash
python3 tests/gen_smoke_dataset.py /tmp/tea-example-depth1 --scenario depth1
```

Train a depth-1 tree:

```bash
./Tea 4 4 1 1 F_FLOAT 16 4 0 0 W_ONES /tmp/tea-example-depth1/ example_depth1 0 LOG1 "manual depth1 example"
```

## Input Dataset Format

Tea expects a directory containing:

- `Labels.lbl`
  Binary `uint16` labels, one per sample
- `Threshholds.thr`
  Binary `float32` threshold metadata, three floats per feature
- `F_0000.feat`, `F_0001.feat`, ...
  Binary feature columns, one file per feature

The supported smoke-data generator writes exactly that structure, so it is the easiest reference for the on-disk layout:

- [tests/gen_smoke_dataset.py](tests/gen_smoke_dataset.py)

## Output Files

A training run writes:

- `Tree_0000.tree`
  Serialized tree snapshot for the current build depth
- `Tree_0000.depthDD.tree`
  Immutable completed-depth model for validation and recovery evidence
- `Tree_0000.tree.labels`
  Predicted training label per sample for that tree snapshot
- `LogFile_<prefix>_0000.log`
  Run log with timing and self-success information

Current model/label files publish atomically after each level. Completed depth files are immutable; use a fresh run directory for a replacement. Depth files are inference checkpoints, not resumable training state. The model filename uses the tree index; the prefix selects the log.

## Repository Layout

- [TreeBuilder.cu](TreeBuilder.cu)
  Main training entry point
- [histograms.cu](histograms.cu), [entropies.cu](entropies.cu), [utilities.cu](utilities.cu)
  CUDA kernels and GPU-side utilities
- [tests/README.md](tests/README.md)
  Supported test workflow
- [examples/README.md](examples/README.md)
  Manual example commands
- [scripts/expand_dataset.py](scripts/expand_dataset.py)
  Dataset inspection and expansion helper
- [scripts/handnet_to_tea.py](scripts/handnet_to_tea.py)
  HandNet `.mat` archive converter
- [scripts/estimate_training_memory.py](scripts/estimate_training_memory.py)
  Standalone Tea GPU-memory estimator for existing datasets
- [scripts/benchmark.py](scripts/benchmark.py)
  Synthetic benchmark runner with a configurable VRAM cap
- [HandNet/README.md](HandNet/README.md)
  HandNet conversion notes; the old MATLAB/MEX files are no longer maintained

## Recommended Path

If you are starting fresh, use:

1. `make Tea` or `bash scripts/build.sh`
2. `make smoke`
3. `make handnet-smoke` if you plan to use HandNet data
4. the examples in [examples/README.md](examples/README.md)

## Troubleshooting

If Tea reports `cudaErrorNoDevice`, first confirm that your shell can see the GPU:

```bash
nvidia-smi
```

If the build fails because of the selected architecture, override it explicitly:

```bash
make Tea CUDA_ARCH=sm_89
```

If OpenMP causes trouble in your environment, disable it:

```bash
make Tea OPENMP_FLAGS=
```

## Current focus

The maintained workflow includes modern CUDA builds, independent CPU/GPU trainer
regressions, reproducible class-aware HandNet conversion and a native forest
reference shared with DepthTrack. Correctness coverage includes the full dense
20-decision model and atomic per-depth checkpoints. Real model quality is evaluated
on held-out data using the [deep-training workflow](docs/deep-training.md).
