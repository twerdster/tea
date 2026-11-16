# HandNet

The Python converter reads HandNet `Data_*.mat` frames and writes native Tea
feature columns. The [deep training workflow](deep-training.md) covers validated
20-decision training, three-tree ensembles and held-out evaluation. The maintained
Python feature contract has not been independently matched to the historical
MATLAB/MEX preprocessing. See `scripts/handnet_to_tea.py` and the generated-input
`tests/run_handnet_converter_test.py` for their separate implementation/check scope.

## Labels and sampling

The default seven modeled classes are raw HandNet labels `1..7` minus one:
thumb, index, middle, ring, pinky, palm and wrist. Raw label `0` is excluded.
Positive centre depth is required. Seven-class inference needs a foreground
mask or detector because it has no learned background class.

`--include-background` also samples positive-depth raw label `0`, mapping it to
modeled class `7` (background). An eight-class manifest reserves output label
`8` for invalid/excluded centres; learned background `7`, invalid centre `8` and
unknown/no-vote `255` are distinct. Zero-depth centres are never trained.

`--sampling uniform` samples eligible pixels uniformly. `--sampling balanced`
caps each visible class without replacement. With eight classes and
`--samples-per-frame 512`, each class has a per-frame cap of 64. Missing/small
classes contribute fewer samples; no duplicates are inserted to meet a quota.
`--class-samples` accepts explicit seven/eight comma-separated class caps.
Actual class histograms and sample counts are recorded in metadata.

`--seed` controls feature coordinates, `--frame-seed` selects frames without
replacement, and `--sampling-seed` selects pixels reproducibly by frame identity.
Keep frame/pixel seeds and manifests fixed when comparing feature scales.
`--sample-alignment 1` preserves every selected pixel with the repaired trainer;
the default alignment of four is retained for compatibility with older Tea.

## Feature definition and scale

Depths are millimetres; coordinates are pixel·millimetres. For a centre `(r,c,z)`,
each neighbour is obtained by **float32** evaluation of the complete sum
`r + offset_r/z` or `c + offset_c/z`, then truncation toward zero. Adding a
separately truncated offset gives different pixels. Do not zero raw depth
outside a foreground mask: masks restrict queried centres only.

For `F_CHAR`, valid depth differences use
`ceil(clamp((d1-d2)/250*127, -127, 127))`. A missing/OOB first neighbour gives
`+127`, a missing second neighbour gives `-127`, and both missing gives zero.
The native reference profile is `float32-complete-sum`.

The historical MATLAB generator used probe size **50**. The maintained converter
now defaults to **50000**, a spatial-scale hypothesis for raw millimetre depths,
not a recovered correction to the historical setting. Compare explicit probes
`50`, `5000`, `25000` and `50000` on validation data within the experiment budget.
At 450 mm depth their maximum radial offsets are approximately 0.056, 5.56,
27.78 and 55.56 pixels. Even subpixel negative offsets can reach a neighbouring
pixel because truncation follows the complete sum. The best scale has not been
established by the trainer correctness tests.

Feature banks generated before the float32 complete-sum repair must be
regenerated. Reuse a tree's exact `Coords.co` for all validation, test and deployed
inference; generating another bank with an assumed equivalent seed is insufficient.

## Bounded conversion

Install NumPy/SciPy with `python -m pip install -r requirements-tools.txt`.
RAR input also requires `bsdtar`; ZIP and extracted directories are supported.
For large RAR subsets, extract selected files once before conversion rather than
reopening the archive for each frame.

A bounded eight-class pilot conversion:

```bash
python scripts/handnet_to_tea.py \
  --archive /data/HandNet/TrainData --output /work/tea/pilot-25000 \
  --split train --features 1200 --probe-size 25000 --seed 101 \
  --frame-seed 20261115 --max-files 2000 \
  --sampling balanced --include-background --samples-per-frame 512 \
  --sampling-seed 101 --sample-alignment 1 --max-output-gib 4
```

The output limit is an upper bound, checked with free space before feature
writing. Default `--max-output-gib` is 100; a larger bank needs an explicit limit
and verified storage. Keep source archives read-only and use a new task-scoped
output directory. `--force` explicitly replaces converter outputs and is not a
run-recovery mechanism.

Outputs include:

| File | Meaning |
| --- | --- |
| `F_NNNN.feat` | One feature column, one value per selected pixel |
| `Labels.lbl` | Little-endian uint16 modeled class per sample |
| `Coords.co` | Little-endian float32 `(row1,col1,row2,col2)` records |
| `Threshholds.thr` | Three float32 threshold metadata values per feature |
| `Poses.pose` | uint32 frame-record index per sample |
| `Pixels.pix` | uint16 `(row,col)` per sample |
| `frames.json` | Exact source frame identities, sample ranges and split |
| `handnet_metadata.json` | Completion, sample/class counts, seeds, feature profile and coordinate checksum |

Use `metadata["samples"]` as Tea's sample count. Requested frames multiplied by
requested pixels is only an upper bound. A successful metadata file has
`complete:true`; interruption must not be mistaken for a completed dataset.

## Freeze splits and reuse coordinates

`--frame-manifest /path/to/frames.json` reuses the exact selected frames and checks
split/source identities. `--manifest-only` records membership without features.
`--coords /path/to/Coords.co` reuses the exact bank and infers its feature count.
For a paired scale comparison, reuse the same training frame manifest and pixel
sampling seed/quotas while changing the explicit probe size. For evaluation of
one bank, reuse its coordinates across held-out splits.

Official HandNet splits are randomized frame splits. Session, global index and
frame index do not establish independent participants. Report official held-out
frame evaluation honestly; subject-independent generalization needs verified
subject grouping. Pass every training `frames.json` to evaluation so known
matching global frame IDs are rejected.

## Train and evaluate

Tea CLI depth is zero based: `19` means **20 split decisions**. Tree indices only
select filenames; they are not seeds. See [deep training](deep-training.md) for
memory bounds, atomic depth checkpoints and independent tree diversity.

```bash
python scripts/tea_model.py inspect --manifest /models/forest.json
python scripts/tea_model.py evaluate \
  --manifest /models/forest.json --archive /data/HandNet/ValidationData \
  --split validation --output /results/validation.json \
  --max-frames 300 --frame-seed 420 --localize --foreground none \
  --training-manifest /work/tea/pilot-25000/frames.json
```

`--foreground none` queries all positive-depth centres and is appropriate for a
forest with learned background. The independent `near-component` depth heuristic
and labeled-hand `oracle` mode are separate evaluation conditions; oracle masks
must be named as such. Foreground masks never modify neighbour depth values.

Compare hand-class macro IoU, per-class recall/precision and fingertip localization
recall at 10/15/20 mm, counting missed present fingertips as failures. Also report
foreground false positives and localization coverage. Training self-success and
label-centroid agreement do not establish anatomical localization accuracy.
Select scales/depths/thresholds on validation; use untouched test data only for
final metrics after configuration is frozen.

## Tests and prior results

`make handnet-smoke` tests generated conversion data. Native routing, histogram
and serialization correctness are checked by the [trainer tests](https://github.com/twerdster/tea/blob/main/tests/README.md).
The earlier guide reported CLI depth 8 on 10,000 frames and 19,949,016 samples;
that historical account lacks its raw logs/device/output artifacts in the public
checkout, and its dominant-class self-success does not validate classifier quality.
It predates the native repairs. The later
[three-tree forest report](handnet-forest-results.md) provides the separate audited
20-decision held-out evaluation. Synthetic success alone is not an accuracy claim.
