# Testing

Tea has two fast generated training cases, a separate generated HandNet conversion scenario,
and independent native deep/weighted regression suites. These code checks are separate from
[real held-out HandNet results](handnet-forest-results.md). This documentation merge did not
rerun CUDA tests; it verified the static documentation and CPU tooling as recorded.

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

## Coverage limits

Generated checks establish their exercised cases, not every historical configuration
or multiple-GPU execution. Application quality and timing require separate evidence.
The official random-frame HandNet evaluation does not establish subject independence,
and numerical CPU parity does not establish temporal accuracy or latency.
