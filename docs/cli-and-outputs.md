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
3. `maxDepth`: tree depth to train to
4. `foldingDepth`: folding depth used for histogram work partitioning
5. `featureType`: `F_CHAR`, `F_SHORT`, `F_INT`, or `F_FLOAT`
6. `numSamples`: number of samples to read from the feature files
7. `numThresh`: number of thresholds to test per feature
8. `startDevice`: first CUDA device index
9. `endDevice`: last CUDA device index
10. `weightType`: `W_ONES`, `W_APRIORI`, or `W_FILE`
11. `baseDir`: dataset directory
12. `treePrefix`: prefix for written outputs
13. `treeNum`: numeric suffix for the run
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
- `Tree_0000.tree.labels`
- `LogFile_<prefix>_0000.log`

### `Tree_0000.tree`

This is the serialized tree snapshot for the run. It contains:

- a tree header
- internal node feature and threshold selections
- compact leaf metadata
- leaf histogram bytes

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

## Weight Types

The maintained path uses `W_ONES`.

Other modes exist:

- `W_APRIORI`
- `W_FILE`

Those are still part of the original code, but they are not part of the current shallow maintained smoke-test path.
