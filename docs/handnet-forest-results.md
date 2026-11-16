# HandNet forest results — 2026-11-16

The completed three-tree forest scored **0.374948 mean fingertip F1 at 15 mm** on
the **10,000-frame official test split**, compared with **0.327988** for the
validation-selected single tree. That is a descriptive improvement of **4.70
percentage points**. Localization and foreground errors remain substantial; this
is a baseline from the repaired Tea pipeline, not a best-possible HandNet result.

The [aggregate JSON](evidence/handnet-forest-2026-11-16.json) records full-precision
metrics, denominators, model/coordinate hashes and settings. The earlier
[pilot report](handnet-pilots.md) remains a historical checkpoint.

## Training and selection

Eight bounded pilots selected eight classes, including background, and a probe
size of **25,000 pixel·mm**. Three full-size trees then trained sequentially with
**1,200 signed-int8 (`F_CHAR`) features** and **20 split decisions** each
(Tea CLI depth 19). Each seed independently selected a 90% training-frame bag,
coordinates and sampled pixels. Balanced sampling capped each class at 64 pixels
per frame: **512 is a cap**, so sparse classes leave the actual count lower.

| Tree | Frame/coordinate/pixel seed | Training frames | Actual sampled pixels |
| --- | ---: | ---: | ---: |
| 0 | 201 | 181,979 | 84,759,330 |
| 1 | 202 | 181,979 | 84,758,588 |
| 2 | 203 | 181,979 | 84,759,516 |

The forest averages the normalized per-tree histograms with equal weights.
The three bags contain 202,010 unique training-frame identities in total.

The forest and each individual tree were compared on the **same frozen 300
validation frames**, at 12/16/20 decisions and probability cutoffs 0.25/0.50:
**24 final validation configurations**. Both the forest and the best single tree
(seed 202) selected **20 decisions, cutoff 0.25, minimum component support 3 and
foreground `none`**. Selection maximized mean fingertip F1@15mm; ties favored fewer
decisions, then the higher cutoff, with seed order breaking ties between singles.

These choices and asset hashes were frozen before test extraction. The later
2,773-frame validation run is descriptive, includes the 300 selection frames and
did not retune the settings. Both frozen choices were then evaluated on all
10,000 test frames. Raw frame identities were checked against every training bag
and all validation frames; no test/training or test/validation overlap was found.

The training/evaluation source is
[`27ae728`](https://github.com/twerdster/tea/tree/27ae728e9cb012d2255337d9b3583bbe7f66b3cc).
Release packaging began from
[`651a95d`](https://github.com/twerdster/tea/tree/651a95dc7894e3e704bdf47158d4452cd547eae6);
later packaging and documentation commits do not change the training revision or
weights. See the [feature contract](handnet.md) and
[training workflow](deep-training.md) for reproducible commands.

## Held-out results

F1 values use a 0–1 scale. The selected single is seed 202 throughout.

| Cohort | Frames | Three-tree F1 @ 15 mm | Single-tree F1 @ 15 mm |
| --- | ---: | ---: | ---: |
| Validation used for selection | 300 | 0.370768 | 0.334966 |
| Full validation, descriptive | 2,773 | 0.383344 | 0.337060 |
| Test, frozen settings | 10,000 | **0.374948** | **0.327988** |

| Test measure | Three-tree forest | Selected single |
| --- | ---: | ---: |
| Mean fingertip recall within 15 mm | 38.78% | 33.92% |
| Detected/scorable fingertip errors | 46,767 | 46,768 |
| Median detected-finger error | 17.49 mm | 19.08 mm |
| Mean detected-finger error | 45.11 mm | 47.94 mm |
| 90th percentile detected-finger error | 72.06 mm | 88.10 mm |
| Seven hand-class mean IoU | 31.13% | 22.59% |
| Learned foreground precision | 72.15% | 64.26% |
| Learned foreground recall | 96.63% | 95.09% |
| Learned foreground IoU | 70.37% | 62.20% |

Test per-finger F1 and its ground-truth denominator are:

| Finger | Scorable presence | Forest F1 @ 15 mm | Single F1 @ 15 mm |
| --- | ---: | ---: | ---: |
| Thumb | 9,399 | 0.458009 | 0.423321 |
| Index | 9,691 | 0.367274 | 0.327155 |
| Middle | 9,448 | 0.362505 | 0.307589 |
| Ring | 9,084 | 0.327919 | 0.266925 |
| Pinky | 9,146 | 0.359031 | 0.314948 |

For each finger, the operational score is:

```text
F1@15mm = 2 × hits within 15 mm
          / (scorable ground-truth presence
             + predictions on scorable present fingers
             + predictions on absent fingers)
```

The reported score is the mean over the five fingers. Missing predictions fail;
wrong locations and predictions on absent fingers are penalized. The forest
still predicted absent fingers on **3,231 of 3,232 absent-finger occasions**.
Its apparently high detection coverage therefore does not establish reliable
presence detection. Error summaries cover detected/scorable fingers only;
use F1 and recall alongside them.

References are HandNet's adjusted sensor centres `pos[:3, :5]`, in camera
millimetres, ordered thumb through pinky. Predictions use the largest 4-connected
confident class component and its probability-weighted centroid. This is our
operational detector and metric, **not a reproduction of the paper's mean-shift
detector or score**.

Eight-class inference evaluates the natural distribution of **all positive-depth
pixels**, learning raw label 0 as background. Hand-class mean IoU averages classes
0–6; foreground metrics merge them into hand versus background. No oracle mask
is used for these results. Balanced training changes class priors, and the
histogram votes have not been established as calibrated probabilities.

Official HandNet splits randomize frames. These results do not establish subject
or session independence, statistical significance or temporal tracking accuracy.
The later [200m-sample/256-feature comparison](handnet-comparison.md) is
post-exposure exploratory engineering after these test outcomes; it is not
independent untouched confirmation. This original report and its models remain
unchanged.

## Verification and native integration

Independent checks matched all saved evaluation identities to raw records and
recomputed ground-truth presence, absent-finger false positives, localization
hits and error summaries. Classification ground-truth supports and confusion
count conservation were checked. This audit reused saved predictions; it did
not rerun the classifier or independently reconstruct its pixel predictions.

Both selected deployment models passed Python/native CPU parity on Linux and
macOS on the same **three training frames**, at the selected cutoff/support and
with both `none` and `near-component` foreground policies. Labels, posterior
bytes, reached heaps and ROI masks matched exactly. Geometry differed by at most
**3.5 × 10⁻¹³ mm**, and Linux/macOS native outputs agreed. Tracker-history checks
used synthetic **0.1-second timestamps**. These are numerical checks, not
held-out tracking accuracy or inference latency measurements. Deep Metal,
mobile and FPGA execution are not established by these CPU results.

See the [DepthTrack CPU integration](https://github.com/twerdster/depthtrack/blob/main/docs/tea-forest.md)
and [selected-model parity evidence](https://github.com/twerdster/depthtrack/blob/main/docs/tea-handnet-selected-model-validation.json).

## Model downloads and attribution

The [HandNet forest release](https://github.com/twerdster/tea/releases/tag/handnet-forest-2026-11-16)
contains relative-path manifests, exact per-tree coordinates and native model
assets. Preserve each model's coordinates and declared float32 feature arithmetic.
The manifests declare the selected postprocessing defaults.

| Artifact | SHA256 |
| --- | --- |
| [Three-tree ensemble ZIP](https://github.com/twerdster/tea/releases/download/handnet-forest-2026-11-16/tea-handnet-selected-ensemble.zip) | `e9cff62eead049b5c5d7dc568acab3e2483a228188ec8290c62170d5fe580cf1` |
| [Seed-202 single-tree ZIP](https://github.com/twerdster/tea/releases/download/handnet-forest-2026-11-16/tea-handnet-selected-single.zip) | `6fbab3bee21e95b13b20f95c1fe4b31c979b07e17361d9d7a859993c464e2641` |

Download [SHA256SUMS](https://github.com/twerdster/tea/releases/download/handnet-forest-2026-11-16/SHA256SUMS.txt)
and verify the ZIP before unpacking. Inspect the selected artifact with:

```bash
python scripts/tea_model.py inspect --manifest unpacked/forest.json
```

HandNet data is credited to Aaron Wetzler, Ron Slossberg and Ron Kimmel under the
dataset release's [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/).
See the [dataset release](https://huggingface.co/datasets/twerdster/HandNet) and
[BMVC 2015 paper](https://arxiv.org/abs/1507.05726). These models use the repaired
Tea trainer and new sampling/evaluation pipeline. Original data archives are not
included; the historical MATLAB code's academic-use notice remains separate.
