
.PHONY: init-tensorflow init-pytorch test-tensorflow test-pytorch format docs bump-minor bump-patch package push-package clean

init-tensorflow:
	uv sync --extra tensorflow

init-pytorch:
	uv sync --extra pytorch

test-tensorflow:
	uv run pytest tests/unit/tensorflow

test-pytorch:
	uv run pytest tests/unit/pytorch

format:
	uv run ruff check --fix src/probflow tests
	uv run ruff format src/probflow tests

docs:
	uv sync --extra docs
	uv run sphinx-build -b html docs docs/_html

bump-minor:
	uv version --bump minor

bump-patch:
	uv version --bump patch

package:
	uv build
	uvx twine check dist/*

push-package:
	uv publish

clean:
	rm -rf .pytest_cache docs/_html build dist src/probflow.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} \+

