# Tests

Trainer correctness checks use generated data and independent CPU expectations.
They are separate from held-out HandNet accuracy evaluation.

## Build and run

```bash
make -j2 Tea CUDA_ARCH=sm_86
make smoke CUDA_ARCH=sm_86
make handnet-smoke CUDA_ARCH=sm_86
make trainer-kernels CUDA_ARCH=sm_86
python tests/trainer_regression.py --tea ./Tea
python tests/trainer_depth20.py --tea ./Tea
python -m pytest tests/test_handnet_pipeline.py tests/test_training_guard.py
```

Use an architecture supported by the installed CUDA toolkit. Native CUDA checks
need a GPU; Python pipeline tests use generated/small fixtures. `trainer_depth20.py`
is a bounded nine-sample allocation/serialization test, not a real training run.

## Native regression coverage

| Test | Checks |
| --- | --- |
| `run_smoke_tests.py` | Shallow root and second-split training, known predicted labels and expected files |
| `trainer_kernels_test.cu` | Exhaustive paired partitions of sizes 0–12; odd 101-sample scatter/gather; CPU/GPU parent/child counts for fast 64/256 and regular histograms; all feature types, odd/empty boundaries and empty folded batches |
| `trainer_regression.py` | Independent CPU exhaustive split-cost checks, all serialized posterior/count/label fields, external tree traversal versus saved predictions, odd/deep/unbalanced/empty paths, folding, streaming pool 1, ignored legacy `DROP_CACHE`, inverse-prior weights and tiny weighted masses, sparse classes above 256 and atomic depth files |
| `trainer_depth20.py` | Exactly 20 decisions (CLI 19), 1,048,575 nodes, 2,097,151 valid posteriors, all 20 immutable depth checkpoints and inference on nine odd samples |

The old `5b791b7` binary fails the new odd-sample fixture at its vector-width
assertion. A separate original-entropy build selects a noisy split over an
available pure split under tiny weighted mass; the repaired criterion passes.
Repaired kernels and trainer pass the independent checks. Weighted
split entropy is also tested against normalized CPU probabilities; adding an
absolute epsilon to very small inverse-frequency counts is not accepted.

The native binary format remains a four-int32 header, 8-byte split nodes,
8-byte compact leaves and byte histograms. All completed histograms have a
posterior; empty branches inherit their nearest populated ancestor. Header/CLI
depth `d` means `d+1` decisions. Current files publish atomically and depth files
are immutable; tests must use a fresh temporary dataset/run directory.

CMake/CTest includes the kernel and serialization suites when `BUILD_TESTING`
is enabled. The full dense 20-decision test is invoked explicitly above.

## Shallow fixtures

```bash
python tests/run_smoke_tests.py --list
python tests/run_smoke_tests.py --tea ./Tea --case depth1-balanced
python tests/gen_smoke_dataset.py /tmp/tea-smoke-root --scenario root
python tests/gen_smoke_dataset.py /tmp/tea-smoke-depth1 --scenario depth1
```

The [HandNet converter smoke](run_handnet_converter_test.py) creates a synthetic
MAT archive, converts it, checks native outputs and runs a shallow tree. The
[Python pipeline tests](test_handnet_pipeline.py) cover feature arithmetic,
coordinate reuse, identities/splits, class-aware sampling, optional learned
background, native forest decoding and evaluation behavior. The
[supervisor tests](test_training_guard.py) check bounded queue/status handling.

## Accuracy limits

Synthetic classification success proves the checked code paths, not fingertip
accuracy or generalization. The earlier 10,000-frame modernization result was
majority-class self-success before native repairs. Use the
[deep-training workflow](../docs/deep-training.md) and
[HandNet held-out protocol](../docs/handnet.md) to evaluate real models with frozen
training/validation/test membership and actual inference outputs.
