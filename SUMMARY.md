# Tea

Tea trains CUDA decision trees over columnar precomputed features. This revision
repairs deep-node routing, vector histogram tails, weighted split scoring,
histogram ancestry, empty-leaf posteriors, large-count comparisons and snapshot
publication. Native model layout is preserved. CLI depth is zero based:
`maxDepth=19` produces 20 split decisions. Each finished level has an immutable
`Tree_NNNN.depthDD.tree` checkpoint; these support inference, not build-state resume.

The HandNet Python converter supports explicit millimetre feature profiles,
reproducible frame/pixel/coordinate seeds, balanced sampling, exact coordinate
reuse, identity manifests and optional learned background (eight classes).
The original probe scale 50 remains a validation baseline; wider probes are
candidate settings, not a claimed reconstruction of the original experiment.

`scripts/tea_model.py` provides native forest inference, normalized per-tree
voting, held-out segmentation/localization evaluation and verified prefix export.
Inference must preserve float32 complete coordinate arithmetic, feature scale,
missing-depth rules, class names and unmodified neighbor depths. Deployment crops
trees to the selected depth rather than silently evaluating a deeper model.

Validation includes CPU/GPU kernel comparisons, eight independent CPU-reference
training cases and a bounded actual 20-decision run. Python tests cover conversion,
model/evaluation semantics and supervisor failure handling. The new DepthTrack CPU
adapter executes on Mac/Linux; historical fixed-depth Metal/FPGA models are separate.
See `docs/deep-training.md` and `docs/handnet.md` for exact commands and limits.

## Pilot checkpoint — 2026-11-15

Eight single-tree HandNet pilots and all 48 configurations on the same frozen 300
validation frames completed. Independent counts/F1 checks agree. The highest pilot
mean fingertip F1@15mm was 0.23650137167567786: eight classes including background,
probe 25000 pixel·mm, 20 decisions and cutoff 0.5. This selects class mode/scale for three
full-size 90 percent frame bags; final depth/cutoff and forest-versus-single comparison
remain for later validation. The first full-size feature bank entered preparation.
`docs/handnet-pilots.md` and its safe aggregate JSON record the modest pilot scores,
foreground/domain differences and numerical CPU parity evidence. No full-size,
ensemble or final-test accuracy is established. Frozen training source remains
27ae728; these documentation updates do not change it. Test data is excluded from
selection; official random-frame splits do not establish subject independence.
Training, data staging and publication remain distinct completion stages.

## Completed forest evaluation — 2026-11-16

Three full-size trees completed with seeds 201/202/203, 181,979 training frames
per tree, about 84.76 million sampled pixels per tree and 1,200 int8 features.
All three have 20 split decisions. The same frozen 300 validation frames selected
20 decisions/cutoff 0.25/minimum support 3/foreground none for the forest and best
single (seed 202); 24 final validation configurations were checked. Selection was
frozen before test extraction and remained unchanged.

The complete 10,000-frame test comparison gives mean fingertip F1@15mm
0.37494755678731473 for the forest versus 0.32798765233655836 for the selected
single. Full 2,773-frame validation scores are 0.3833437004586795/0.3370599042841775;
that descriptive cohort includes the 300 selection frames. Independent raw-reference
count/hit/error audits passed, with no test overlap with any training bag or
validation frame. Classification truth supports and count conservation were checked;
predictions were not rerun during the audit. Official random frame splits do not
establish subject/session independence.

`docs/handnet-forest-results.md` and
`docs/evidence/handnet-forest-2026-11-16.json` contain safe results, denominators,
selected assets and numerical Linux/macOS CPU parity evidence. These checks use
three training frames and synthetic timestamps; they do not establish tracking
accuracy, latency or deep Metal/FPGA execution. The selected release is
`handnet-forest-2026-11-16`; publication is a distinct stage. Training source stays
27ae728; packaging began from 651a95d and later documentation does not change weights.
The historical pilot checkpoint above remains unchanged. The separate 200m/256
comparison uses an explicitly authorized additional attempt; its live phase belongs
in the private run ledgers. Comparisons after original test disclosure are explicitly
post-exposure exploratory engineering, not untouched confirmation.

`scripts/prepare_forest_release.py` prepares a local publication candidate from a
selected deployment manifest. It reconstructs known inference fields, copies only
model/coordinate assets, replaces runtime provenance with reviewed public metadata,
and verifies exact synthetic labels, probabilities, heaps and declared observations.
The training source revision must be supplied explicitly; repository HEAD is never
used as a substitute. Private source hashes are retained in a separate verification
record. This utility does not upload artifacts or satisfy actual-frame/native
parity requirements before publication.

Task-local machine paths, active job identities, budgets, storage verification and
monitoring state belong in ignored `runs/current/HANDOFF.md`, `WATCHLIST.json`,
`STATE.json`, `CAMPAIGN.json` and `BUDGET.json`. Consult live evidence before resuming
work; source documentation is not proof that a job completed.

## Upstream documentation reconciliation — 2026-11-16

The incoming main documentation changes preserve the concise first-run setup,
CUDA/host operator chain, attribution and `docs/images/tea-training-pipeline.svg`.
The figure remains an unmeasured implementation schematic. Validation statements
now distinguish shallow smoke cases, independent deep/weighted regressions and
this campaign's real held-out results. The docs merge performs static/CPU checks;
it does not launch CUDA training or claim every historical or multi-GPU case.
The maintained Python feature contract has not been matched to historical
MATLAB/MEX output. Original credits and software-license notices are preserved.

The public 100-frame training label-semantics evidence is aggregate-only: pixel
counts, intrinsics and source-archive hash are retained, with the audited frame
count and a declared canonical ordered-identity-list hash. Exact identity records
are preserved privately; public evidence omits raw frame/session identifiers.
This inventory repair does not rewrite training history, change the frozen
27ae728 training code or alter models.
Its `handnet-label-semantics-aggregate-v1` schema explicitly records the original
raw audit schema separately.

## Bounded 200m/256 comparison workflow

`scripts/run_handnet_comparison.py` implements one additional seed204 comparison
with the original tree0 frame identities, 256 saved coordinate features and an
expected 200,216,884 balanced pixel samples. It preserves the original three
trees and their immutable budget. Explicit authorization must reconcile four
final attempts in total, with no new pilots, at most 16 active training hours
for the new run and the existing 48-hour aggregate ceiling. A separate residual
ledger records only new charges; waiting for resources is not training time.

Private runtime configuration supplies machine paths, identity and service
settings. Published source review and operational monitoring receipts bind the
exact worker and configuration bytes. Unknown or failed reservations consume
an attempt and cannot be silently retried. The worker validates the converted
bank and all checkpoints, evaluates the frozen six-setting validation grid and
exports the selected prefix. It does not open TestData, publish models or delete
feature columns. Independent metric audit, selected-model native parity, backup
and final reporting remain separate stages.

`scripts/collect_handnet_comparison.py` checks the separate job, original plus
residual plus unsettled live budget, process/service identity, logs, checkpoints,
finite evaluations and resources. The existing recurring project monitor must
cover these paths before launch and stay active through evaluation and recovery.
See `docs/handnet-comparison.md`; source documentation is not evidence of a live
or completed job.
