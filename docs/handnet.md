# HandNet

The Python converter reads HandNet `Data_*.mat` files and writes native Tea datasets. It is
intended to reproduce useful parts of the historical MATLAB/MEX preprocessing, but its outputs
have not been compared with that implementation. The current implementation and generated-input
check are:

- `scripts/handnet_to_tea.py`
- `tests/run_handnet_converter_test.py`

## What The Converter Does

The converter reads HandNet frames containing at least:

- `depth`
- `lbl`

It then:

- samples pixels where `lbl >= 1`
- converts labels from HandNet's `1..7` range to Tea's zero-based `0..6` range
- generates radial depth-difference coordinates intended to follow the historical feature construction
- computes one Tea feature-column file per generated feature
- writes Tea threshold metadata, coordinates, poses, labels, and JSON metadata

The output directory contains:

```text
Labels.lbl
Threshholds.thr
Coords.co
Poses.pose
F_0000.feat
F_0001.feat
...
handnet_metadata.json
```

## Install Tooling

Install the Python dependencies:

```bash
python3 -m pip install -r requirements-tools.txt
```

For `.rar` archives such as `TrainData.rar`, the converter also needs `bsdtar` on `PATH`. `.zip` archives are handled with Python's standard library.

## Convert A Dataset

Example using a validation archive:

```bash
python3 scripts/handnet_to_tea.py \
  --archive /path/to/ValidationData.zip \
  --output /tmp/tea-handnet-validation \
  --features 200 \
  --feature-type F_CHAR \
  --samples-per-frame 2000 \
  --force
```

The `--samples-per-frame` cap is optional. It is useful for bounded experiments because full pixel-level HandNet datasets can become very large.

## Large TrainData Slices

`TrainData.rar` can be passed directly to the converter, but large slices are much faster if you extract the selected `.mat` files once and convert the extracted directory. RAR random access is expensive when each frame is read separately.

Example: extract the first 10,000 training frames:

```bash
python3 - <<'PY'
import subprocess
from pathlib import Path

archive = Path("/path/to/TrainData.rar")
out = Path("/tmp/tea-handnet-train-10000-names.txt")
result = subprocess.run(["bsdtar", "-tf", str(archive)], check=True, capture_output=True, text=True)
names = [line for line in result.stdout.splitlines() if line.endswith(".mat")][:10000]
out.write_text("\n".join(names) + "\n")
print(len(names), names[0], names[-1])
PY

mkdir -p /tmp/tea-handnet-train-10000-extracted
bsdtar -xf /path/to/TrainData.rar \
  -C /tmp/tea-handnet-train-10000-extracted \
  -T /tmp/tea-handnet-train-10000-names.txt
```

Convert that extracted slice:

```bash
python3 scripts/handnet_to_tea.py \
  --archive /tmp/tea-handnet-train-10000-extracted/TrainData \
  --output /tmp/tea-handnet-train-10000-d8 \
  --features 200 \
  --feature-type F_CHAR \
  --samples-per-frame 2000 \
  --force
```

Then estimate and train a depth-8 tree:

```bash
python3 scripts/estimate_training_memory.py \
  --dataset /tmp/tea-handnet-train-10000-d8 \
  --max-depth 8 \
  --folding-depth 8 \
  --feature-type F_CHAR \
  --vram-cap-mb 500 \
  --require-fit

./Tea 200 32 8 8 F_CHAR <samples> 8 0 0 W_ONES \
  /tmp/tea-handnet-train-10000-d8/ handnet_train_10000_d8 0 LOG0 \
  "HandNet TrainData 10000-frame depth-8 test"
```

The earlier repository guide reported a completed depth-8 run on 10,000 TrainData frames (19,949,016 samples and 200 `F_CHAR` features), with an estimated 110 MB per device. No raw run log, device record or output artifact accompanies that report in this checkout, so those figures are a historical account rather than a reproduced result. The same account reported self-success near the dominant class rate; that would not establish fingertip detection accuracy.

## Estimate Training Memory

Before running Tea, estimate the per-device GPU memory requirement:

```bash
python3 scripts/estimate_training_memory.py \
  --dataset /tmp/tea-handnet-validation \
  --max-depth 3 \
  --folding-depth 3 \
  --feature-type F_CHAR \
  --vram-cap-mb 500 \
  --require-fit
```

This reads `Labels.lbl`, infers sample and class counts, asks `nvidia-smi` for currently free memory, and exits nonzero if the run is over the configured cap or currently free VRAM.

## Train A Tree

Read the sample count from the converter metadata:

```bash
python3 - <<'PY'
import json
print(json.load(open("/tmp/tea-handnet-validation/handnet_metadata.json"))["samples"])
PY
```

Then pass that sample count to Tea. The argument after the sample count (`8` below) is the number
of candidate thresholds, not the number of classes; see [CLI parameters](cli-and-outputs.md):

```bash
./Tea 200 32 3 3 F_CHAR <samples> 8 0 0 W_ONES \
  /tmp/tea-handnet-validation/ handnet_validation_d3 0 LOG0 \
  "HandNet validation depth-3 test"
```

## Converter Smoke Test

The repo includes a no-data HandNet converter test. It creates a tiny synthetic HandNet-style `.zip`, converts it, estimates memory, and runs Tea:

```bash
make handnet-smoke
```

This generated-input scenario checks the conversion layout, labels, estimated memory and nonempty training outputs. It does not check predicted-label correctness. Real HandNet archives and application metrics require separate integration and evaluation records.

## Accuracy Note

A depth-3 run on raw pixel labels would be a data-path check, not a full application benchmark. HandNet labels are highly imbalanced at the pixel level, so a shallow tree can report self-success near the dominant class rate. Meaningful application evaluation should add held-out prediction tooling, class-balancing decisions, deeper trees or ensembles, and task-specific metrics.
