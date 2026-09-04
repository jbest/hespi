# Installing this fork in another project

This fork of [hespi](https://github.com/rbturnbull/hespi) (https://github.com/jbest/hespi) is not
published on PyPI, so it's installed directly from git. See the README's
["Why This Fork Exists"](README.rst) section for what's different from upstream and why.

Pin to a tag or commit rather than `main`, since `main` may move. The current stable point is the
tag `v0.1.0-fork` (commit `78d3f2c9ddd1c99db28a2a42b9126398a3db0579`).

## Install

**pip**
```bash
pip install "git+https://github.com/jbest/hespi.git@v0.1.0-fork"
```

**Poetry** (add to `pyproject.toml`)
```toml
hespi = {git = "https://github.com/jbest/hespi.git", tag = "v0.1.0-fork"}
```

**uv**
```bash
uv add "git+https://github.com/jbest/hespi.git@v0.1.0-fork"
```

To track the latest fork commit instead of a fixed tag, replace `v0.1.0-fork` with `main` (or omit
the ref entirely for pip, which defaults to the default branch).

## Prerequisites

- Python 3.10-3.13
- [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Home.html) installed as a **system**
  binary, not a Python package — e.g. `apt install tesseract-ocr` (Debian/Ubuntu) or
  `brew install tesseract` (macOS)

## After installing

- The `hespi` CLI becomes available. The first run downloads pretrained model weights
  automatically.
- If you'll use the LLM-correction step, set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the
  environment (or pass `--llm-api-key`), or disable it with `--llm none`.
- Run `hespi --help` to confirm the install and see all CLI options.

## Watch out for

This package depends on `fastai`, `torch`, `transformers`, `ultralytics`, and `langchain`, which
are heavy and version-sensitive. If the target project already pins any of these, resolve
dependencies explicitly (`pip install` / `poetry lock` / `uv lock`) and check the resolution
before assuming it's clean, rather than just trusting that the install command above succeeded.

## Updating the pin

To move to a newer fork commit later, either point the ref at a newer tag (once one exists) or a
specific commit hash from https://github.com/jbest/hespi/commits/main, then reinstall.
