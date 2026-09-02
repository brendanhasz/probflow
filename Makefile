
.PHONY: install-with-tensorflow install-with-pytorch test-tensorflow test-pytorch test-stats-tensorflow test-stats-pytorch test format lint type-check docs bump-minor bump-patch package push-package clean

# Install with tensorflow backend
install-with-tensorflow:
	uv sync --extra tensorflow --extra docs

# Install with pytorch backend
install-with-pytorch:
	uv sync --extra pytorch --extra docs

# Run unit tests with tensorflow backend
test-unit-tensorflow: install-with-tensorflow
	uv run pytest tests/unit/tensorflow

# Run unit tests with pytorch backend
test-unit-pytorch: install-with-pytorch
	uv run pytest tests/unit/pytorch

# Run statistical checks using tensorflow backend
test-stats-tensorflow: install-with-tensorflow
	uv run pytest tests/stats --backend=tensorflow

# Run statistical checks using pytorch backend
test-stats-pytorch: install-with-pytorch
	uv run pytest tests/stats --backend=pytorch

# Run all tests, including statistical checks
test: test-unit-tensorflow test-stats-tensorflow test-unit-pytorch test-stats-pytorch

# Format, lint, and type check code
format:
	uv run ruff check --fix src/probflow tests
	uv run ruff format src/probflow tests
	uv run mypy src/probflow

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

