# Tea

Tea is a CUDA/C++ program that trains a randomized decision tree from feature columns prepared
offline. The related BMVC 2015 paper [*Rule of Thumb: Deep Derotation for Improved Fingertip Detection*](https://arxiv.org/abs/1507.05726)
is by Aaron Wetzler, Ron Slossberg and Ron Kimmel. The repository's converter maps HandNet data
from that work to the feature-column format used by Tea.

The completed [three-tree HandNet report](handnet-forest-results.md) records
0.374948 mean fingertip F1@15mm on all 10,000 test frames versus 0.327988 for the
validation-selected single tree. It includes exact settings, denominators,
limitations and selected model downloads.

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

Two shallow smoke cases check generated training labels. A separate generated
HandNet scenario checks conversion layout and nonempty outputs, without verifying
that scenario's predicted labels. Independent native suites additionally cover
routing/histogram edge cases, weights, folding, streaming and the full dense
20-decision tree. See [Testing](testing.md) for exact assertions and limits.

Real held-out segmentation/localization quality comes from the separate audited
HandNet campaign: settings were selected on 300 validation frames before opening the
10,000 test frames. The complete 2,773-frame validation cohort is descriptive and
includes that selection subset. This documentation merge performs static/CPU
checks; it does not rerun CUDA training or reproduce the historical paper metric.

## Read next

- [Getting Started](getting-started.md): requirements, build commands and first smoke checks.
- [Training Workflow](training-workflow.md): the implementation's operator chain.
- [Dataset Format](dataset-format.md): binary input layout.
- [CLI and Outputs](cli-and-outputs.md): positional arguments and output files.
- [HandNet](handnet.md): MAT archive conversion and evaluation qualifications.
- [Benchmarking](benchmarking.md): synthetic runtime sweeps and CSV fields.

Reported checks do not establish every historical configuration or multiple-GPU execution.
Official random-frame HandNet splits do not establish subject/session independence.
Generated-data timing and numerical CPU parity are separate from application accuracy.
