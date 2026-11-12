# Testing

Tea has two generated training cases and one separate generated HandNet conversion scenario.
Their presence describes the available checks; this documentation revision did not run the CUDA
binary, and this checkout contains no current raw execution record.

## Training cases

[run_smoke_tests.py](https://github.com/twerdster/tea/blob/main/tests/run_smoke_tests.py) uses
[gen_smoke_dataset.py](https://github.com/twerdster/tea/blob/main/tests/gen_smoke_dataset.py) to generate `root-depth0` and
`depth1-balanced`. Both run GPU 0 with `F_FLOAT` inputs, `W_ONES`, and equal folding/maximum depths.
For each case the runner checks:

- successful process exit;
- nonempty `.tree`, `.tree.labels` and log files;
- one predicted label per input sample, exactly matching the generated training labels;
- a `Depth X: 100.0000%` log line at the requested depth.

The percentage is self-success on deliberately separable training data. It is not a held-out
accuracy, a throughput measurement or coverage of all numerical configurations.

```bash
make smoke
python3 tests/run_smoke_tests.py --list
python3 tests/run_smoke_tests.py --tea ./Tea --case depth1-balanced
```

`make smoke` runs both training cases. The Python command accepts a different binary path, such
as `./build-cmake/Tea`, and repeatable `--case` arguments.

## HandNet conversion scenario

[run_handnet_converter_test.py](https://github.com/twerdster/tea/blob/main/tests/run_handnet_converter_test.py) creates four synthetic
32×32 HandNet-style MAT frames in a ZIP archive. It converts 256 samples per frame into 1,024
samples with 40 `F_CHAR` features, then checks file sizes and zero-based labels spanning 0–6.
It runs the memory estimator with a 500 MB cap, trains to depth 1 and requires nonempty tree,
predicted-label and log outputs.

```bash
python3 -m pip install -r requirements-tools.txt
make handnet-smoke
```

This scenario does not compare predicted labels with expected values or require a particular
self-success percentage. It uses generated input, so no HandNet download is needed, but CUDA
training and the Python dependencies are still required.

## Coverage limits

The three scenarios do not establish deeper-tree correctness, all feature types, other weighting
modes, multiple GPU execution, real HandNet prediction accuracy or current-toolchain compatibility.
A larger run described in a guide is not part of this regression suite. Keep performance and
application evaluation records separate from these small data-path checks.
