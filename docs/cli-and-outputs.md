# CLI and Outputs

Tea is driven by a positional command-line interface from `TreeBuilder.cu`.

## CLI Parameters

The maintained command layout is:

```text
./Tea \
  <numFeatures> \
  <poolSize> \
  <maxDepth> \
  <foldingDepth> \
  <featureType> \
  <numSamples> \
  <numThresh> \
  <startDevice> \
  <endDevice> \
  <weightType> \
  <baseDir> \
  <treePrefix> \
  <treeNum> \
  <logLevel> \
  "<comment>"
```

Parameter meanings:

1. `numFeatures`: how many feature columns to evaluate
2. `poolSize`: how many features to preload at once
3. `maxDepth`: zero-based split level; `19` means 20 split decisions
4. `foldingDepth`: folding depth used for histogram work partitioning
5. `featureType`: `F_CHAR`, `F_SHORT`, `F_INT`, or `F_FLOAT`
6. `numSamples`: number of samples to read from the feature files
7. `numThresh`: number of thresholds to test per feature
8. `startDevice`: first CUDA device index
9. `endDevice`: last CUDA device index
10. `weightType`: `W_ONES`, `W_APRIORI`, or `W_FILE`
11. `baseDir`: dataset directory
12. `treePrefix`: prefix for the log filename
13. `treeNum`: numeric model/log suffix, not an RNG seed
14. `logLevel`: `LOG0` through `LOG4`
15. `comment`: free-form run comment

## Minimal Example

```bash
./Tea 4 4 0 0 F_FLOAT 16 4 0 0 W_ONES /tmp/tea-example-root/ example_root 0 LOG1 "manual root example"
```

## Depth-1 Example

```bash
./Tea 4 4 1 1 F_FLOAT 16 4 0 0 W_ONES /tmp/tea-example-depth1/ example_depth1 0 LOG1 "manual depth1 example"
```

## Output Files

A run writes:

- `Tree_0000.tree`
- `Tree_0000.depthDD.tree`
- `Tree_0000.tree.labels`
- `LogFile_<prefix>_0000.log`

### `Tree_0000.tree`

This is the serialized tree snapshot for the run. It contains:

- a tree header
- internal node feature and threshold selections
- compact leaf metadata
- leaf histogram bytes

Current model and training-label files publish atomically. Each completed depth
also writes an immutable `Tree_0000.depthDD.tree`. Existing depth files cause an
overwrite refusal; use a fresh run directory after diagnosing a failure. These
are inference checkpoints, not resumable training state.

### `Tree_0000.tree.labels`

This file stores the predicted label per sample for the current tree snapshot.

It is the easiest output to use in smoke tests because it can be compared directly to a known label list.

### `LogFile_<prefix>_0000.log`

This file contains:

- the resolved CLI parameters
- allocation summaries
- per-depth success summaries
- overall timing lines

The benchmark script and smoke tests both parse these logs.

## Log Levels

Tea supports `LOG0` through `LOG4`.

Practical guidance:

- `LOG0`: concise run output
- `LOG1`: a good default for debugging and smoke tests
- `LOG2+`: more detailed internal timing and worker chatter

## Weight types

`W_ONES` uses unweighted Shannon split entropy. `W_APRIORI` divides class counts
by their global training frequency; `W_FILE` reads positive integer class
frequency divisors from `Weights.txt` (`class_index divisor` per line). Weighted
entropy normalizes probabilities before the logarithm, including very small
deep-node masses. All modes keep raw leaf posteriors. The weighted modes are
covered by the independent CPU split regressions.

The accepted depth range is `0 <= foldingDepth <= maxDepth <= 20`. Invalid feature
or weight modes, truncated inputs and CUDA allocation failures terminate the run.
See [deep training](deep-training.md) for the 20-decision and forest contracts.
