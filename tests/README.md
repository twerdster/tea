# Tests

The active test workflow in this repository is the smoke-test suite in [run_smoke_tests.py](/home/aaron/repos/tea/tests/run_smoke_tests.py). It covers:

- a depth-0 root-separable dataset
- a depth-1 balanced dataset that requires a second split

Both cases are generated on the fly by [gen_smoke_dataset.py](/home/aaron/repos/tea/tests/gen_smoke_dataset.py), run through the `Tea` binary, and then verified by checking:

- the process exits successfully
- the expected `.tree`, `.labels`, and log files are written
- the expected output files are non-empty
- the predicted labels exactly match the known labels for the final tree depth
- the log contains a `Depth X: 100.0000%` success line for the target depth

## Run The Smoke Tests

With a locally built binary:

```bash
python3 tests/run_smoke_tests.py --tea ./Tea
```

List the available cases:

```bash
python3 tests/run_smoke_tests.py --list
```

Run only one shallow case:

```bash
python3 tests/run_smoke_tests.py --tea ./Tea --case depth1-balanced
```

Using the Makefile helper:

```bash
make smoke
```

## Generate A Single Dataset

Root-only smoke dataset:

```bash
python3 tests/gen_smoke_dataset.py /tmp/tea-smoke-root --scenario root
```

Depth-1 smoke dataset:

```bash
python3 tests/gen_smoke_dataset.py /tmp/tea-smoke-depth1 --scenario depth1
```

## Related Utility

To derive a larger dataset from an existing Tea dataset, use:

```bash
python3 scripts/expand_dataset.py inspect /path/to/data single
python3 scripts/expand_dataset.py expand /path/to/data single /tmp/tea-expanded 1000 200 7 single
```

## HandNet Converter Smoke Test

The HandNet converter has a separate no-data smoke test:

```bash
make handnet-smoke
```

It creates a synthetic HandNet-style `.zip`, converts it with `scripts/handnet_to_tea.py`, checks the generated Tea files, estimates memory under a 500 MB cap, and runs Tea on the converted dataset.
