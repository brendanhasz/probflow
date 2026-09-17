---
name: formatting
description: Ensure consistent code formatting using the uv package manager and pre-commit. Trigger this whenever formatting is requested.
domain: development.formatting
version: 1.0.0
---

# Python Formatting Skill via uv

This skill ensures that all Python code is consistently formatted and type-checked using the `uv` environment and `pre-commit`. You must never invoke formatting commands (`pre-commit`, `black`, `isort`, etc.) directly on the host system.  Always run the pre-commit pipeline, do **not** attempt to run formatting or type-checking commands manually (e.g. `ruff`, `mypy`, `black`, `isort`).

## Core Directives

- **Always run full formatting pipeline** via `uv run pre-commit run --all-files`.

## Execution Rules

### 1. Running the Full Formatting and Type-checking Pipeline
After applying any changes and ensuring the tests pass, always run the full formatting pipeline to ensure consistent code style across the project:

```bash
# To run the full formatting and type-checking pipeline on all files
uv run pre-commit run --all-files
```

### 2. Address Formatting and Type-checking Issues
If the formatting and type-checking pipeline fails, first try running a second time to check if the issues were able to be auto-fixed.

If the formatting and type-checking pipeline fails a second time, you must manually address the issues before proceeding.

Iterate this process until all formatting and type-checking issues are resolved and the pre-commit pipeline passes successfully.
