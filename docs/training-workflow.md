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

- the trainer reads a defined binary layout
- feature extraction is separate from split search
- synthetic benchmark runs can vary feature count independently of extraction cost

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

- `maxDepth`: the depth you want to train to
- `foldingDepth`: the depth at which histogram work is folded for device-side processing

The folding depth changes how histogram storage and passes over node subsets are organized. In practice, it is a performance parameter, not just a semantic tree-depth parameter.

Both generated training cases use `foldingDepth == maxDepth`; alternative folding configurations are outside their coverage.

## Work distribution

The implementation separates disk loading, host coordination and GPU split evaluation. Feature columns can be prepared while workers process earlier columns. The performance of that overlap depends on storage, feature counts, device memory and GPU configuration; this description is not a measured speedup.

## Known Limits

Tea is an older research codebase. The available small tests leave these areas unverified:

- very deep trees
- large historical benchmark configurations that were never part of modern CI
- application-level evaluation beyond the current smoke and converter tests

For supported verification, stay close to the testing and benchmarking flows documented in this site.
