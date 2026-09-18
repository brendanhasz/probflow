---
name: running-tests
description: Run Python unit test suites strictly using the uv package manager and pytest. Trigger this whenever testing is requested.
domain: development.testing
version: 1.0.0
---

# Python Testing Skill via uv

This skill ensures that all Python test suites and ad-hoc scripts are strictly executed inside the `uv` environment. You must never invoke testing framework commands (`pytest`) directly on the host system.

## Core Directives

- **Always prepend test executions** with `uv run`.
- **Never bypass the virtual environment** or use system-level python binaries directly for project tasks.
- **Maintain local isolation** by ensuring `uv.lock` and project dependencies remain active during verification.

## Execution Rules

### 1. Running the Unit Test Suite
When executing tests, always target the root workspace or specified test file using `uv run pytest path/to/test.py -v --color=no`.

```bash
# To run the entire shared unit test suite
uv run pytest tests/unit/shared -v --color=no

# To run unit tests within a specific directory
uv run pytest path/to/test_directory -v --color=no

# To run a specific test file
uv run pytest path/to/test_file.py -v --color=no

# To run a specific test within a specific file
uv run pytest path/to/test_file.py::name_of_specific_test -v --color=no
```

### 2. Ad-hoc/Isolated Script Checks
If you need to execute temporary scripts or evaluate Python object behaviors to diagnose a failing test, always spin them up using the project's pinned virtual environment context:

```bash
uv run python path/to/temporary_script.py
```

### 3. Backend Selection and Setup

If the desired test being run requires using a specific "backend" (i.e., Tensorflow, PyTorch, or JAX), ensure that the appropriate dependencies are installed and activated within the `uv` environment before executing the tests.

```bash
# Example: Running a test with Tensorflow backend
uv sync --extra tensorflow
uv run pytest path/to/test_file.py -v --color=no

# Example: Running a test with PyTorch backend
uv sync --extra pytorch
uv run pytest path/to/test_file.py -v --color=no

# Example: Running a test with JAX backend
uv sync --extra jax
uv run pytest path/to/test_file.py -v --color=no
```

If no backend is required or specified, assume one has already been installed, and simply run the tests as usual within the current `uv` environment.

If it turns out backend dependencies are not installed, and a specific backend is required,
use the TensorFlow backend by default (`uv sync --extra tensorflow`), and then proceed to run the tests within the `uv` environment as usual.
