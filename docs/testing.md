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

## Why The Tests Stop At Depth 1

The repository is an older research codebase. The current goal of the maintained test surface is:

- prove the modernized build still works
- prove the supported dataset and runtime path still works
- keep verification fast and repeatable

That is why the supported automated path stops at shallow trees for now.

Deeper configurations are better treated as benchmark workloads and staged regressions, not as the first mandatory correctness layer.
