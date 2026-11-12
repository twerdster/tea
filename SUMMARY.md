# tea context

## Purpose

CUDA/C++ randomized decision-tree trainer over precomputed binary feature columns, with Python generation, conversion, estimation and synthetic benchmark tools.

## Implementation evidence

Two F_FLOAT/W_ONES/GPU0 cases check exact synthetic training labels at depths 0/1. Separate converter test makes 1,024 samples/40 F_CHAR features, checks layout/labels and nonempty training output but not predicted-label correctness.

Key entry points: `TreeBuilder.cu`, `histograms.cu`, `entropies.cu`, `tests/run_smoke_tests.py`, `tests/run_handnet_converter_test.py`, `docs/`.

## Documentation state

The README explains operations, requirements, validation scope and limitations. Its pre-existing explanatory SVG is retained and labeled as a schematic rather than a runtime result. Source files and attribution are unchanged.

## Validation scope

Implementation/assertion inspection and static MkDocs rebuild; no new CUDA build, GPU test, training or benchmark run. Earlier guide’s depth-8 account lacks raw logs/device/output artifacts in this checkout.

## Limitations

Deep trees, multiple GPUs, alternate weights/types and held-out application accuracy are outside the synthetic cases. VRAM caps are estimates, and self-success is training-set agreement.

The Python HandNet converter is intended to follow the historical preprocessing; its outputs
have not been compared with the MATLAB/MEX implementation.
