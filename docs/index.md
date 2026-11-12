# Tea

Tea is a CUDA/C++ program that trains a randomized decision tree from feature columns prepared
offline. The related BMVC 2015 paper [*Rule of Thumb: Deep Derotation for Improved Fingertip Detection*](https://arxiv.org/abs/1507.05726)
is by Aaron Wetzler, Ron Slossberg and Ron Kimmel. The repository's converter maps HandNet data
from that work to the feature-column format used by Tea.

## Data flow

1. Read labels and threshold metadata, then load feature columns from disk.
2. Send columns to CUDA workers and reorder them for the current sample index lists.
3. Evaluate candidate thresholds by accumulating class histograms and entropy scores.
4. Select the best split for each active node on the host.
5. Partition sample indices and repeat for the next depth.
6. Write a tree snapshot, training-sample predictions and a log.

The core trainer searches over precomputed features. Image preprocessing and feature extraction
belong to separate tools, including the Python HandNet converter.

## Implementation and validation scope

The repository includes Make/CMake builds, C++ standard-library host threading and Python tools
for generated datasets, conversion, memory estimation and synthetic timing sweeps. Linux and an
NVIDIA CUDA device are required for training.

The automated training suite has two synthetic cases at depths 0 and 1. A separate synthetic
HandNet conversion scenario checks the data path and nonempty training outputs; it does not
check predicted-label correctness. See [Testing](testing.md) for the exact assertions and limits.
There is no current raw build/test log in this checkout. The documentation revision did not run
GPU training or reproduce the paper's results.

## Read next

- [Getting Started](getting-started.md): requirements, build commands and first smoke checks.
- [Training Workflow](training-workflow.md): the implementation's operator chain.
- [Dataset Format](dataset-format.md): binary input layout.
- [CLI and Outputs](cli-and-outputs.md): positional arguments and output files.
- [HandNet](handnet.md): MAT archive conversion and evaluation qualifications.
- [Benchmarking](benchmarking.md): synthetic runtime sweeps and CSV fields.

Large datasets, multi-GPU execution, deep trees and alternative weighting modes need validation
beyond the small synthetic checks. Timing on generated data is not an application accuracy result.
