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

The current HandNet campaign prepares eight bounded pilot configurations and
three independently sampled final trees. Application accuracy and ensemble gains
remain unverified until held-out evaluation completes. Test data is excluded from
selection; official random-frame splits do not establish subject independence.
Training, data staging and publication are distinct completion stages.

Task-local machine paths, active job identities, budgets, storage verification and
monitoring state belong in ignored `runs/current/HANDOFF.md`, `WATCHLIST.json`,
`STATE.json`, `CAMPAIGN.json` and `BUDGET.json`. Consult live evidence before resuming
work; source documentation is not proof that a job completed.
