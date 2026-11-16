# Benchmarking

Tea includes a Python benchmark runner for bounded, repeatable performance sweeps.

## Purpose

The benchmark runner exists to answer questions like:

- how does cost scale with sample count?
- how does cost scale with feature count?
- how much does depth change total training time?
- how much GPU memory will a given run likely require?

## Script

The maintained benchmark entry point is:

- `scripts/benchmark.py`

It generates synthetic Tea datasets directly in the repository's native file format, runs the `Tea` binary, parses logs, and writes a CSV summary.

## Estimated VRAM cap

The runner estimates per-device VRAM usage before each run and rejects any case above the configured cap.

Default:

```text
500 MB
```

The cap limits the estimator's calculation, not actual allocations or other processes' memory use. A case below the estimate can still fail at runtime.

For existing Tea datasets, use the standalone estimator:

```bash
python3 scripts/estimate_training_memory.py \
  --dataset /tmp/tea-handnet-validation \
  --max-depth 3 \
  --folding-depth 3 \
  --feature-type F_CHAR \
  --vram-cap-mb 500 \
  --require-fit
```

## Example Sweep

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

This produces:

- generated datasets under `/tmp/tea-bench-run/datasets/`
- CSV results under `/tmp/tea-bench-run/results/benchmark_results.csv`

## What The CSV Contains

The output CSV includes fields such as:

- suite name
- sample count
- feature count
- tree depth
- threshold count
- pool size
- estimated VRAM
- final depth reached
- final self-success percentage
- tree timing parsed from the log
- full wall-clock elapsed time

## Synthetic Dataset Design

The runner builds synthetic datasets that are intentionally easy to split so the benchmark is measuring the trainer rather than a complex external preprocessing pipeline.

That means the benchmark is best interpreted as:

- a throughput and scaling tool
- a way to compare runtime changes on controlled synthetic inputs

It is not meant to stand in for a published application-level accuracy benchmark.

## Practical Advice

For stable and interpretable runs:

- use a single GPU unless you are explicitly measuring multi-GPU behavior
- keep `foldingDepth == maxDepth` for maintained benchmark cases
- start with `F_CHAR` unless you want datatype comparisons
- keep the VRAM cap conservative on shared systems

## Relationship to the historical workflow

The Python runner replaces the earlier MATLAB command-generation layer with explicit dataset-generation, subprocess and CSV-writing steps. This documentation revision did not execute a sweep, and no synthetic timing value here should be read as an application accuracy result.

The separate [HandNet forest report](handnet-forest-results.md) records real
held-out segmentation/localization results. It does not turn this synthetic
benchmark page into a measured application-throughput claim.
