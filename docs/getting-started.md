# Getting Started

This page describes the build commands and small checks available in the repository. A successful local run is required before treating a toolchain as validated.

## Requirements

You need:

- Linux
- an NVIDIA GPU with a working driver
- the CUDA toolkit, including `nvcc`
- Python 3
- a C++11-capable host compiler
- CMake if you want the CMake build path

OpenMP is optional. Tea uses it for a small amount of host-side parallel work when available.

## Check The Local Environment

Tea includes a helper that checks the toolchain and GPU visibility:

```bash
bash scripts/check_env.sh
```

That script checks:

- `nvcc`
- `g++`
- `python3`
- `cmake`
- `nvidia-smi`

If `nvidia-smi` fails, investigate device/driver visibility before running training. Its failure alone does not distinguish a missing GPU from a tooling or permission problem.

## Build With The Makefile

The simplest build path is:

```bash
make Tea
```

Useful overrides:

```bash
make Tea CUDA_ARCH=sm_89
make Tea NVCC=/path/to/nvcc
make Tea OPENMP_FLAGS=
```

## Build With CMake

The repository also supports a CMake-based build:

```bash
bash scripts/build.sh
```

By default, this writes to `build-cmake/`.

Useful overrides:

```bash
CUDA_ARCH=89 bash scripts/build.sh
TEA_ENABLE_OPENMP=OFF bash scripts/build.sh
```

## Verify The Build

Run the supported smoke tests:

```bash
make smoke
```

Or run them directly:

```bash
python3 tests/run_smoke_tests.py --tea ./Tea
```

The smoke suite currently covers:

- a depth-0 root-separable dataset
- a depth-1 balanced dataset

These cases check exact training-label predictions on small synthetic inputs. Passing them validates those cases in the tested environment; it does not establish compatibility or correctness for every historical training regime. See [Testing](testing.md) for the assertions and coverage limits.

The independent trainer suites additionally check deep/odd/empty routing, weighted
splits, serialization and a bounded full 20-decision tree. Follow the
[test reference](https://github.com/twerdster/tea/blob/main/tests/README.md) after
the fast smoke checks. CLI depth 19 means 20 split decisions; see
[deep training](deep-training.md) for memory, checkpoints and held-out selection.

If you plan to use HandNet conversion, install the extra Python tooling and run its converter smoke test:

```bash
python3 -m pip install -r requirements-tools.txt
make handnet-smoke
```

## Build The Documentation Site

Install docs dependencies:

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
