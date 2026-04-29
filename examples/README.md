# Examples

## Quick Build

Check that the local environment has the expected tools:

```bash
bash scripts/check_env.sh
```

Makefile build:

```bash
make Tea
```

CMake build:

```bash
bash scripts/build.sh
```

Override the CUDA architecture if needed:

```bash
CUDA_ARCH=89 bash scripts/build.sh
```

## Run The Supported Smoke Tests

With the Makefile build:

```bash
make smoke
```

With the CMake build:

```bash
bash scripts/build.sh
bash scripts/run_smoke.sh ./build-cmake/Tea
```

With an explicit binary path:

```bash
python3 tests/run_smoke_tests.py --tea ./Tea
```

## Inspect Or Expand A Dataset

Inspect an existing Tea dataset:

```bash
python3 scripts/expand_dataset.py inspect /path/to/data single
```

Write a derived dataset:

```bash
python3 scripts/expand_dataset.py expand /path/to/data single /tmp/tea-expanded 1000 200 7 single
```

## Convert HandNet Data

Install the converter dependencies:

```bash
python3 -m pip install -r requirements-tools.txt
```

Convert a HandNet archive:

```bash
python3 scripts/handnet_to_tea.py \
  --archive /path/to/ValidationData.zip \
  --output /tmp/tea-handnet-validation \
  --features 200 \
  --feature-type F_CHAR \
  --samples-per-frame 2000 \
  --force
```

Check the estimated memory for a depth-3 run:

```bash
python3 scripts/estimate_training_memory.py \
  --dataset /tmp/tea-handnet-validation \
  --max-depth 3 \
  --folding-depth 3 \
  --feature-type F_CHAR \
  --vram-cap-mb 500 \
  --require-fit
```

Run the converter smoke test:

```bash
make handnet-smoke
```

For larger `TrainData.rar` experiments, extract a bounded file list first and convert the extracted directory. See `docs/handnet.md` for a 10,000-frame depth-8 example.

## Benchmark Tea

Run a bounded benchmark sweep and write a CSV:

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

## Manual Root-Only Example

Generate a tiny dataset:

```bash
python3 tests/gen_smoke_dataset.py /tmp/tea-example-root --scenario root
```

Train a depth-0 tree:

```bash
./Tea 4 4 0 0 F_FLOAT 16 4 0 0 W_ONES /tmp/tea-example-root/ example_root 0 LOG1 "manual root example"
```

## Manual Depth-1 Example

Generate a dataset that needs a second split:

```bash
python3 tests/gen_smoke_dataset.py /tmp/tea-example-depth1 --scenario depth1
```

Train a depth-1 tree:

```bash
./Tea 4 4 1 1 F_FLOAT 16 4 0 0 W_ONES /tmp/tea-example-depth1/ example_depth1 0 LOG1 "manual depth1 example"
```

After either run, inspect:

- `Tree_0000.tree`
- `Tree_0000.tree.labels`
- `LogFile_<prefix>_0000.log`
