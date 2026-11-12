# tea maintenance workflow

Read `SUMMARY.md`, this file and `README.md` before editing.

## Working on the repository

README keeps requirements and the first run; use docs for advanced CLI/build/conversion/sweeps. Rebuild MkDocs after edits and distinguish a static documentation build from CUDA runtime validation. Do not start training or benchmarks for a prose-only change. Preserve author/source notices; no standalone license is added.

## Documentation checks

- Check claims against the named source files and recorded results. Separate source inspection,
  configuration generation, runtime execution and benchmark evidence.
- Keep README setup requirements available without following an advanced guide. Check Markdown
  headings, links and image references, including source files omitted by a sparse checkout.
- Inspect labels inside diagrams as well as captions. Never turn a schematic or generated input
  into an implied successful run.
- Keep original source, credits and license notices intact for documentation-only changes.
- Review and synchronize `SUMMARY.md` and `SKILL.md` before every commit. Record any unrun checks
  as limitations rather than implying that validation passed.

## Current evidence boundary

Implementation/assertion inspection and static MkDocs rebuild; no new CUDA build, GPU test, training or benchmark run. Earlier guide’s depth-8 account lacks raw logs/device/output artifacts in this checkout.

Deep trees, multiple GPUs, alternate weights/types and held-out application accuracy are outside the synthetic cases. VRAM caps are estimates, and self-success is training-set agreement.

The Python HandNet converter is intended to follow the historical preprocessing; its outputs
have not been compared with the MATLAB/MEX implementation.
