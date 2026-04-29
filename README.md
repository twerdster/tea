# Tea

Tea is a CUDA/C++ decision-tree trainer for very large precomputed feature datasets. It was originally built as a research codebase for multi-GPU tree training, and this repository now includes:

- a modernized host-side threading layer based on the C++ standard library
- a supported local build flow with both `make` and CMake
- shallow automated smoke tests for depth `0` and depth `1`
- small generated example datasets and copy-pasteable example commands
- a Python dataset expansion utility instead of the old MATLAB helper

The current supported workflow is intentionally focused on buildability, shallow correctness checks, and ease of setup. Deeper benchmark scripts from the original project are still present, but they are treated as legacy performance assets rather than the primary test surface.

## Repository Status

The repository has been updated to build on a modern Linux toolchain. The active workflow in this repo is:

- build the `Tea` binary
- run the supported smoke tests
- use the small example datasets to verify shallow-tree training

The supported automated test coverage currently stops at depth `1`. That is deliberate: it validates the modernized build/runtime path without immediately stepping into older deeper-tree edge cases that deserve separate work.

## Requirements

You need:

- Linux
- an NVIDIA GPU with a working driver
- the CUDA toolkit, including `nvcc`
- Python 3
- a C++11-capable host compiler
- CMake if you want the CMake-based build path

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

The supported smoke-test runner is [tests/run_smoke_tests.py](/home/aaron/repos/tea/tests/run_smoke_tests.py).

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

More detail is in [tests/README.md](/home/aaron/repos/tea/tests/README.md).

## Dataset Utility

The old MATLAB dataset-expansion helper has been replaced by [scripts/expand_dataset.py](/home/aaron/repos/tea/scripts/expand_dataset.py).

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

Use [scripts/benchmark.py](/home/aaron/repos/tea/scripts/benchmark.py) to generate bounded synthetic datasets, run Tea over sample/feature/depth sweeps, and write a CSV summary.

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

Additional copy-pasteable examples live in [examples/README.md](/home/aaron/repos/tea/examples/README.md).

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

- [tests/gen_smoke_dataset.py](/home/aaron/repos/tea/tests/gen_smoke_dataset.py)

## Output Files

A training run writes:

- `Tree_0000.tree`
  Serialized tree snapshot for the current build depth
- `Tree_0000.tree.labels`
  Predicted label per sample for that tree snapshot
- `LogFile_<prefix>_0000.log`
  Run log with timing and self-success information

Note that Tea snapshots the tree after each level and rewrites the same output filenames for the current run prefix.

## Repository Layout

- [TreeBuilder.cu](/home/aaron/repos/tea/TreeBuilder.cu)
  Main training entry point
- [histograms.cu](/home/aaron/repos/tea/histograms.cu), [entropies.cu](/home/aaron/repos/tea/entropies.cu), [utilities.cu](/home/aaron/repos/tea/utilities.cu)
  CUDA kernels and GPU-side utilities
- [tests/README.md](/home/aaron/repos/tea/tests/README.md)
  Supported test workflow
- [examples/README.md](/home/aaron/repos/tea/examples/README.md)
  Manual example commands
- [scripts/expand_dataset.py](/home/aaron/repos/tea/scripts/expand_dataset.py)
  Dataset inspection and expansion helper
- [scripts/benchmark.py](/home/aaron/repos/tea/scripts/benchmark.py)
  Synthetic benchmark runner with a configurable VRAM cap

## Recommended Path

If you are starting fresh, use:

1. `make Tea` or `bash scripts/build.sh`
2. `make smoke`
3. the examples in [examples/README.md](/home/aaron/repos/tea/examples/README.md)

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

## Current Focus

The repository is now organized around:

- clean builds on a modern toolchain
- supported shallow smoke tests
- clear examples and setup

The next layers of work after that are deeper regression coverage and longer legacy benchmark validation, but those are intentionally separate from the current easy-to-use path.
