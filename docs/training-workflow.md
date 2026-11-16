# Training Workflow

Tea trains a single randomized decision tree by evaluating many feature and threshold candidates at each depth and selecting the best split for every active node.

## High-Level Flow

At a high level, a run looks like this:

1. Load labels and threshold metadata from disk.
2. Initialize every sample at the root node.
3. Create one or more GPU workers.
4. Iterate over tree depth.
5. For each feature:
   - load the feature column
   - send data to a worker
   - test thresholds and accumulate histograms
   - compute split scores
6. Reduce feature results back on the host.
7. Choose the best split per node.
8. Reorder the sample index lists for the next depth.
9. Snapshot the tree and predicted labels.

## Why Feature Columns Matter

Tea assumes that features have already been computed offline and saved column-wise:

- `F_0000.feat`
- `F_0001.feat`
- ...

This is a deliberate tradeoff:

- it makes training simple and repeatable
- it avoids mixing feature extraction with split search
- it allows the trainer to benchmark raw split-evaluation throughput

The cost is that you must prepare the dataset in Tea's on-disk format before training.

## GPU Work Distribution

Tea uses a pool of GPU workers. Each worker:

- receives a feature column in host memory
- copies data to the device
- scatters it into the current sample order
- evaluates thresholds
- builds class histograms
- computes entropy-based split scores

The host then reduces those per-feature results and updates the tree state for the next level.

## Folding Depth

Tea distinguishes between:

- `maxDepth`: zero-based split level; `d` means `d+1` decisions (`19` for 20)
- `foldingDepth`: the depth at which histogram work is folded for device-side processing

The folding depth changes how histogram storage and passes over node subsets are organized. In practice, it is a performance parameter, not just a semantic tree-depth parameter.

Shallow examples can use `foldingDepth == maxDepth`; deeper runs can use a lower folding depth to bound histogram buffers. Both folded and unfolded paths have regression coverage. Require `0 <= foldingDepth <= maxDepth <= 20`.

## Work distribution

The implementation separates disk loading, host coordination and GPU split evaluation.
Feature columns can be prepared while workers process earlier columns. The performance
of that overlap depends on storage, feature counts, device memory and GPU configuration;
this description is not a measured speedup.

## Limits and deep training

One invocation trains one tree; the tree number selects its output filename and
is not an RNG seed. There is no native minimum-leaf-size or pruning option.
Three diverse trees need independent reproducible feature/sample banks.

The repaired native path has independent CPU/GPU regression coverage, including
the full 20-decision dense model. This does not establish application accuracy.
Use [deep training](deep-training.md) for bounded queueing, memory, atomic depth
checkpoints and validation selection. Use natural-distribution held-out data for
segmentation/localization metrics rather than training self-success.

The [completed forest report](handnet-forest-results.md) records the separate
held-out application results; it does not establish every historical or multi-GPU
configuration or reproduce the paper's detector metric.
