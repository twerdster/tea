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
