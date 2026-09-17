
.PHONY: install test-unit test-stats test format benchmark-linear-regression benchmark docs bump-minor bump-patch clean

BACKEND ?= tensorflow
AVAILABLE_BACKENDS := tensorflow pytorch jax

# Install probflow package and requirements
install:
	uv sync --extra $(BACKEND)

# Run unit tests
test-unit: install
	uv run pytest tests/unit/shared tests/unit/$(BACKEND)

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
	uv run pre-commit run --all-files

# Benchmark fitting a linear regression
benchmark-linear-regression: install
	uv run scripts/benchmark_linear_regression.py

# Run benchmarking for all backends and write docs file
benchmark:
	@for backend in $(AVAILABLE_BACKENDS); do \
		$(MAKE) benchmark-linear-regression BACKEND=$$backend; \
	done
	uv run scripts/write_benchmark_results.py
	
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

# Clean up build artifacts and caches
clean:
	rm -rf .pytest_cache docs/_html build dist src/probflow.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} \+
