dev-env:
	source venv/bin/activate
	pip install -r requirements.txt
	pip install -e .[tests]

test:
	pytest tests/unit

format:
	black src/probflow tests
	flake8 src/probflow tests
