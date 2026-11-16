# HandNet pilot results — 2026-11-15

Eight bounded single-tree pilots and **48 validation configurations** completed.
The highest selection score was **0.236501 mean fingertip F1 at 15 mm**, using
**eight classes including background and probe size 25,000 pixel·mm**. These
modest pilot scores select a larger training configuration; they do not establish
full-size forest accuracy or an ensemble improvement.

The frozen training/evaluation source is
[`27ae728`](https://github.com/twerdster/tea/tree/27ae728e9cb012d2255337d9b3583bbe7f66b3cc).
The [safe aggregate JSON](evidence/handnet-pilots-2026-11-15.json) contains full-precision
scores, model/coordinate hashes and the comparison settings.

## Experiment

Each pilot used the same **2,000 training frames**, **1,200 `F_CHAR` features**,
coordinate/pixel seed 101 and frame-selection seed 20261115. Sampling capped each
visible class without replacement: up to 73 pixels per hand class for seven-class
training, or 64 per class including background for eight-class training. Actual
counts were 911,260 and 932,968 respectively. Frame and sampled-pixel identities were shared across the four scale comparisons
within each class mode; the coordinate seed stayed fixed while probe scale changed.

All eight models trained to **20 split decisions** (Tea CLI depth 19). Each was
then evaluated at 12, 16 and 20 decisions with probability cutoffs 0.25 and 0.50.
Every evaluation used the **same frozen 300-frame official validation subset**,
selected with seed 420. The table shows the highest score among those six settings
for each pilot. F1 values are on a 0–1 scale.

| Classes | Probe size (pixel·mm) | Training frames | Training pixels | Selected decisions | Probability cutoff | Mean F1 @ 15 mm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 hand classes | 50 | 2,000 | 911,260 | 12 | 0.25 | 0.008744 |
| 7 hand classes | 5,000 | 2,000 | 911,260 | 16 | 0.25 | 0.115623 |
| 7 hand classes | 25,000 | 2,000 | 911,260 | 20 | 0.25 | 0.227339 |
| 7 hand classes | 50,000 | 2,000 | 911,260 | 20 | 0.25 | 0.202858 |
| 7 hand + background | 50 | 2,000 | 932,968 | 16 | 0.25 | 0.002559 |
| 7 hand + background | 5,000 | 2,000 | 932,968 | 12 | 0.25 | 0.127600 |
| **7 hand + background** | **25,000** | **2,000** | **932,968** | **20** | **0.50** | **0.236501** |
| 7 hand + background | 50,000 | 2,000 | 932,968 | 20 | 0.25 | 0.209021 |

Probe coordinates have explicit pixel·millimetre units and are divided by centre
depth in millimetres. Scale 50 is the historical baseline, not evidence that its
units were originally intended to be larger. See the [feature and sampling
contract](handnet.md) and [deep-training workflow](deep-training.md).

## Metric and interpretation

For each fingertip, a hit requires a predicted point within 15 mm of the adjusted
sensor reference. The score is the mean of per-finger F1 values:

```text
F1 = 2 × hits / (scorable ground-truth presence
                + predictions on scorable present fingers
                + predictions on absent fingers)
```

Missing predictions remain failures; wrong locations and predictions on absent
fingers are penalized. Predictions use the largest 4-connected confident component
with at least three pixels and its probability-weighted centroid. This is the new pipeline's metric, not a
reproduction of the paper's detector or scoring procedure.

The foreground policy differs between class modes. Seven-class pilots use the
fixed depth-only `near-component` ROI. Eight-class pilots classify all positive
depth and learn raw label 0 as background. Their inference uses raw neighbour depths
throughout; an ROI only restricts queried centres. Pixel domains therefore differ,
and raw pixel accuracies are not interchangeable.

The selected eight-class pilot has **14.25% hand-class macro IoU** and learned
foreground precision/recall/IoU of **53.15% / 89.64% / 50.08%**. The seven-class
25,000-scale comparison has **22.17% hand-class macro IoU** and depth-ROI
precision/recall/IoU of **85.76% / 90.63% / 78.78%**. Selection follows fingertip
F1, so its winner does not win every segmentation measure. The substantial
foreground errors and modest localization scores remain limitations.

One additional CPU-only validation diagnostic applied the depth ROI to the chosen
pilot at the same depth, cutoff and minimum support. It reduced F1 from **0.236501
to 0.218240** (15 mm hits 344 → 317), despite improving the learned foreground
IoU from 50.08% to 77.41%. False positives on absent fingers stayed at 94. This
separate diagnostic is **outside the original 48-configuration grid** and left the
frozen `none` foreground choice unchanged; no training or test inputs were added.

## Verification and remaining work

An independent audit checked all eight saved models and 1,024 external
feature-column traversals per model. It independently recomputed the ground-truth
counts and F1 for all 48 configurations from the same 300 raw validation frames.
No training-frame identity overlap was found in that subset.

The selected pilot was also compared with native CPU inference on three **training**
frames using both foreground policies. Labels, reached heaps, masks and probability
bytes agreed exactly. Geometry/tracker numerical differences were below
3.5 × 10⁻¹³ mm on those inputs. The synthetic timestamp cadence used for that check
establishes numerical equivalence, not tracking accuracy, inference latency or
held-out recognition quality.

At this reporting checkpoint, preparation of the first full-size feature bank had
begun. The chosen **eight-class / 25,000-scale** setting carries forward into three
independent 90% training-frame bags with seeds 201, 202 and 203. Final depth and
probability cutoff will be selected again on validation after those trees exist,
with the three-tree forest compared against individual trees. No full-size tree,
ensemble or final-test score is reported here.

The official test data was not used for pilot selection. Final test evaluation
remains after the final model/depth/threshold selection is frozen. Official HandNet
splits randomize frames; these results do **not** establish participant or session
independence. The reused selection subset is not an unbiased final test estimate.

HandNet data is credited to Aaron Wetzler, Ron Slossberg and Ron Kimmel under the
release's [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/).
These results use the repaired Tea trainer and new sampling/evaluation pipeline.
See the [dataset release](https://huggingface.co/datasets/twerdster/HandNet) and
[BMVC 2015 paper](https://arxiv.org/abs/1507.05726). The historical MATLAB code's
academic-use notice remains separate from the data license.
