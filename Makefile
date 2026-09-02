
.PHONY: install-with-tensorflow install-with-pytorch test-tensorflow test-pytorch test-stats-tensorflow test-stats-pytorch test format lint type-check docs bump-minor bump-patch package push-package clean

BACKEND ?= tensorflow
AVAILABLE_BACKENDS := tensorflow pytorch

# Install probflow package and requirements
install:
	uv sync --extra $(BACKEND) --extra docs

# Run unit tests
test-unit: install
	uv run pytest tests/unit/$(BACKEND)

# Run statistical checks
test-stats: install
	uv run pytest tests/stats --backend=$(BACKEND)

# Run all tests, including statistical checks, for all backends
test:
	@for backend in $(AVAILABLE_BACKENDS); do \
		$(MAKE) test-unit BACKEND=$$backend; \
		$(MAKE) test-stats BACKEND=$$backend; \
	done

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
