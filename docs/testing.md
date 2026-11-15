# Testing

Tea has a Python-driven smoke-test surface for the maintained build and data paths.

## Supported Tests

The supported runner is:

- `tests/run_smoke_tests.py`

Current maintained cases:

- `root-depth0`
- `depth1-balanced`
- `handnet-smoke`

These are generated on the fly by:

- `tests/gen_smoke_dataset.py`
- `tests/run_handnet_converter_test.py`

## What The Smoke Tests Verify

Each smoke case checks:

- process exit succeeds
- expected output files exist
- output files are non-empty
- predicted labels match the expected labels exactly
- the run log contains `Depth X: 100.0000%` at the target depth

That gives you a fast correctness check for the maintained shallow path.

The HandNet converter smoke test additionally checks:

- a synthetic HandNet-style `.zip` can be converted through the Python path
- the generated Tea files have the expected sizes and zero-based labels
- the memory estimator accepts the generated dataset under a 500 MB cap
- Tea can train on the converted data

## Run The Full Smoke Suite

```bash
make smoke
```

Or:

```bash
python3 tests/run_smoke_tests.py --tea ./Tea
```

Run the HandNet converter smoke test:

```bash
make handnet-smoke
```

## Run A Single Case

```bash
python3 tests/run_smoke_tests.py --tea ./Tea --case depth1-balanced
```

## List The Cases

```bash
python3 tests/run_smoke_tests.py --list
```

## Native deep and weighted regression coverage

The generated shallow smoke cases remain fast build checks. The repaired trainer
also has independent CPU/GPU kernel and full serialization regressions:

```bash
make trainer-kernels
python tests/trainer_regression.py --tea ./Tea
python tests/trainer_depth20.py --tea ./Tea
```

Coverage includes odd/singleton partitioning, all histogram methods and feature
types, folded empty nodes, streaming feature reloads, weighted entropy and every
serialized posterior. The bounded nine-sample depth test exercises exactly 20
decisions (CLI 19), 1,048,575 split nodes, 2,097,151 posteriors and all 20 immutable
checkpoints. See the [test reference](https://github.com/twerdster/tea/blob/main/tests/README.md) for details.

These tests establish checked code behavior. Real HandNet segmentation and
localization quality require the separate [held-out protocol](handnet.md).
