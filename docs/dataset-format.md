# Dataset Format

Tea trains from a directory containing labels, threshold metadata, and one file per feature column.

## Required Files

The minimum dataset layout is:

```text
dataset/
  Labels.lbl
  Threshholds.thr
  F_0000.feat
  F_0001.feat
  ...
```

The filename `Threshholds.thr` is intentionally spelled that way because it matches the historical code and file loader.

## Labels

`Labels.lbl` contains one `uint16` label per sample.

Important points:

- labels are zero-based class IDs
- the trainer infers `numClasses` from the maximum observed label
- every feature file must contain exactly the same number of samples

## Threshold Metadata

`Threshholds.thr` contains three `float32` values per feature:

1. a positive-range proxy or max-value proxy
2. the observed minimum feature value
3. the observed maximum feature value

Tea uses these values when generating candidate thresholds during training.

## Feature Files

Each `F_XXXX.feat` file stores one feature column.

The feature file type must match the `featureType` CLI argument:

- `F_CHAR`
- `F_SHORT`
- `F_INT`
- `F_FLOAT`

For example:

- if the trainer is invoked with `F_FLOAT`, each feature file must contain `float32`
- if the trainer is invoked with `F_CHAR`, each feature file must contain signed 8-bit values

## Sample Alignment Note

The maintained smoke tests use sample counts that are multiples of `4`. That remains the safest path for new datasets because some scatter/reordering assumptions in the legacy code were written around aligned sample handling.

## Small Example Dataset

Generate a tiny root-separable dataset:

```bash
python3 tests/gen_smoke_dataset.py /tmp/tea-example-root --scenario root
```

Generate a tiny depth-1 dataset:

```bash
python3 tests/gen_smoke_dataset.py /tmp/tea-example-depth1 --scenario depth1
```

These are the fastest reference examples for understanding the on-disk layout.

## Expand Or Inspect A Dataset

Tea now includes a Python dataset utility that replaces the old MATLAB helper.

Inspect a dataset:

```bash
python3 scripts/expand_dataset.py inspect /path/to/data single
```

Write a derived dataset:

```bash
python3 scripts/expand_dataset.py expand /path/to/data single /tmp/tea-expanded 1000 200 7 single
```

The utility can:

- tile samples to a new sample count
- change the stored feature datatype
- repeat feature columns to a new feature count
- derive new label distributions for benchmarking experiments
