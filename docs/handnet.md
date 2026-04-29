# HandNet

`HandNet/` is preserved as a legacy preprocessing artifact from one historical use case of Tea.

## What It Contains

The directory currently contains:

- `genData.m`
- `getDepthFeatures.cpp`
- `getDepthFeatures.mexa64`

## What It Was Used For

The HandNet code path was used to generate precomputed depth-image features for a hand/fingertip pipeline. That preprocessing stage is related to the BMVC 2015 work that used Tea for large-scale training.

In other words:

- HandNet prepares feature columns
- Tea consumes those feature columns for tree training

## Current Status

HandNet is not part of the maintained modern build and smoke-test path.

That means:

- it is not built by `make Tea`
- it is not built by the CMake path
- it is not validated by the current Python smoke tests

## Why It Is Still In The Repository

It remains in the repository because it documents one real historical data-generation workflow tied to the original research use case.

## Recommendation

Treat HandNet as archived research support code unless you specifically need to revive that exact preprocessing path.

If you do need it, document its dataset assumptions and build steps separately before relying on it for new work.
