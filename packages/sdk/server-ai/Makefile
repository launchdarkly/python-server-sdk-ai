PYTEST_FLAGS=-W error::SyntaxWarning

.PHONY: help
help: #! Show this help message
	@echo 'Usage: make [target] ... '
	@echo ''
	@echo 'Targets:'
	@grep -h -F '#!' $(MAKEFILE_LIST) | grep -v grep | sed 's/:.*#!/:/' | column -t -s":"

.PHONY: install
install: #! Install package dependencies
	uv sync --all-groups

.PHONY: test
test: #! Run unit tests
test: install
	uv run pytest $(PYTEST_FLAGS)

.PHONY: lint
lint: #! Run type analysis and linting checks
lint: install
	uv run mypy src/ldai
	uv run isort --check --atomic src/ldai
	uv run pycodestyle src/ldai

.PHONY: build
build: #! Build distribution files
build: install
	uv build --out-dir dist

.PHONY: docs
docs: #! Generate sphinx-based documentation
	uv sync --group docs
	uv run sphinx-build -W --keep-going -M html docs docs/build
