# Maintaining This Site

Tea uses MkDocs to build a multi-page static documentation site that can be served locally or deployed through GitHub Pages.

## Files

The docs site is driven by:

- `mkdocs.yml`
- `docs/`
- `requirements-docs.txt`
- `.github/workflows/docs.yml`

## Local Preview

Install dependencies:

```bash
python3 -m pip install -r requirements-docs.txt
```

Serve locally:

```bash
bash scripts/serve_docs.sh
```

Build the static output:

```bash
bash scripts/build_docs.sh
```

MkDocs writes the generated site to:

```text
site/
```

## Deployment

The repository includes a GitHub Pages workflow:

- `.github/workflows/docs.yml`

It builds the docs on pushes to `main` when docs-related files change and deploys the generated static site to GitHub Pages.

## Writing Guidance

When updating the docs:

- prefer task-oriented pages over long unstructured prose
- keep CLI examples copy-pasteable
- be explicit about what is maintained and what is legacy
- keep HandNet clearly separated from the main maintained path

## When To Update The README

The README should stay as a concise entry point. More detailed explanations should usually go in `docs/` and then be linked from the README rather than duplicated line-for-line.
