# Tea

Tea is a CUDA/C++ trainer for large randomized decision trees over precomputed feature datasets. It was built for the regime where the expensive part of the pipeline is not feature extraction at runtime, but searching large banks of candidate features and thresholds across very large sample sets.

Tea was used for large-scale training in the BMVC 2015 paper [*Rule Of Thumb: Deep derotation for improved fingertip detection*](https://arxiv.org/abs/1507.05726) by Aaron Wetzler, Ron Slossberg, and Ron Kimmel.

## What Tea Is Good At

Tea is a good fit when you have:

- a large offline dataset that is already stored as feature columns
- many candidate features that must be evaluated efficiently
- one or more NVIDIA GPUs on a single machine
- a workflow where shallow-to-moderate depth trees are useful and training speed matters

Tea is not a general modern machine learning framework. It does not aim to replace end-to-end deep learning libraries, and it is not built around Python-first training loops. Its strength is focused GPU acceleration for a specific style of decision-tree training.

## Core Idea

Tea stores each feature in a separate file, keeps labels and threshold metadata alongside them, and repeatedly performs the same high-level loop:

1. Load a feature column into host memory.
2. Scatter it into sample order on the GPU.
3. Evaluate multiple thresholds for that feature.
4. Build class histograms for candidate splits.
5. Score split quality and keep the best split per node.
6. Repartition samples and continue to the next tree level.

The implementation is designed to overlap disk reads, host-side coordination, and GPU work so the system can handle datasets that are large in sample count, feature count, or both.

## Why The Design Is Useful

Tea exists for the case where a dataset is too large or too expensive to treat as a tiny in-memory experiment. The design is useful because it:

- makes feature evaluation scale with GPU throughput instead of pure CPU loops
- keeps the training format simple and explicit on disk
- supports multi-GPU training on a single machine
- provides a natural way to benchmark how cost changes with feature count, threshold count, depth, and sample count

## Maintained Path

The maintained path in this repository is:

- build with `make` or CMake
- verify the binary with shallow smoke tests
- use the Python dataset utilities for examples, benchmarking, and HandNet conversion

The repository does not claim full regression coverage for all historical deep-tree workloads. The supported automated verification surface currently focuses on shallow correctness and bounded benchmark runs.

## Where To Start

If you are new to Tea, read these pages in order:

1. [Getting Started](getting-started.md)
2. [Training Workflow](training-workflow.md)
3. [Dataset Format](dataset-format.md)
4. [CLI and Outputs](cli-and-outputs.md)

If you want to measure performance, go to [Benchmarking](benchmarking.md).

If you want to use restored HandNet data, go to [HandNet](handnet.md).
