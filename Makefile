.PHONY: install doctor test lint typecheck audit package check release-check demo-gpu demo-cpu

install:
	python -m pip install -e '.[dev]'

doctor:
	agentic-autoresearch doctor

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src/agentic_al

audit:
	PYTHONWARNINGS=default bandit -r src/agentic_al
	PYTHONWARNINGS=default pip-audit --local --skip-editable

package:
	python -m build
	twine check dist/*

check: lint typecheck test

release-check: check audit package

demo-gpu:
	agentic-autoresearch demo --output-dir outputs/demo-gpu

demo-cpu:
	agentic-autoresearch demo --cpu-baseline --output-dir outputs/demo-cpu
