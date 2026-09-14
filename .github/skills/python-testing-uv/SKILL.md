---
name: python-testing-uv
description: Run Python test suites, verify code quality gates, and handle test execution strictly using the uv package manager. Trigger this whenever testing or quality checks are requested.
domain: development.testing
version: 1.0.0
---

# Python Testing Skill via uv

This skill ensures that all Python test suites, quality gates, and ad-hoc scripts are strictly executed inside the `uv` environment. You must never invoke testing framework commands (`pytest`) directly on the host system.

## Core Directives

- **Always prepend test executions** with `uv run`.
- **Never bypass the virtual environment** or use system-level python binaries directly for project tasks.
- **Maintain local isolation** by ensuring `uv.lock` and project dependencies remain active during verification.

## Execution Rules

### 1. Running the Test Suite
When executing tests, always target the root workspace or specified test directory using `uv run pytest`.

```bash
# Correct execution pattern
uv run pytest

# Verbose mode with terminal color forced off for clean agent logs
uv run pytest -v --color=no
```

### 2. Quality Gates & Pre-Commit Checks
A task involving code changes is only considered "done" when the following test and linting checklist passes successfully via `uv`:

- `uv run pre-commit run --all-files` leaves the code unchanged and reports zero errors.
- `uv run pytest` passes cleanly.

### 3. Ad-hoc/Isolated Script Checks
If you need to execute temporary scripts or evaluate Python object behaviors to diagnose a failing test, always spin them up using the project's pinned virtual environment context:

```bash
uv run python path/to/temporary_script.py
```
