
.PHONY: install-with-tensorflow install-with-pytorch install test-tensorflow test-pytorch test format docs bump-minor bump-patch package push-package clean

# Install with default backend of tensorflow (explicit)
install-with-tensorflow:
	uv sync --extra tensorflow --extra docs

# Install with default backend of pytorch
install-with-pytorch:
	uv sync --extra pytorch --extra docs

# Install with default backend of tensorflow
install: install-with-tensorflow

# Run tests with tensorflow backend
test-tensorflow: install-with-tensorflow
	uv run pytest tests/unit/tensorflow

# Run tests with pytorch backend
test-pytorch: install-with-pytorch
	uv run pytest tests/unit/pytorch

# Run tests with default backend of tensorflow
test: test-tensorflow

# Format code
format:
	uv run ruff check --fix src/probflow tests
	uv run ruff format src/probflow tests

# Build documentation
docs:
	uv sync --extra docs
	uv run sphinx-build -b html docs docs/_html

# Bump minor version number
bump-minor:
	uv version --bump minor

# Bump patch version number
bump-patch:
	uv version --bump patch

# Build and check the package
package:
	uv build
	uvx twine check dist/*

# Push the package to pypi
push-package:
	uv publish

# Clean up build artifacts and caches
clean:
	rm -rf .pytest_cache docs/_html build dist src/probflow.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} \+

