PYTEST_FLAGS=-W error::SyntaxWarning
SPHINXOPTS    = -W --keep-going
SPHINXBUILD   = sphinx-build

# Configurable docs paths (can be overridden from root Makefile)
DOCS_DIR ?= docs
DOCS_BUILD_DIR ?= docs/build

.PHONY: help
help: #! Show this help message
	@echo 'Usage: make [target] ... '
	@echo ''
	@echo 'Targets:'
	@grep -h -F '#!' $(MAKEFILE_LIST) | grep -v grep | sed 's/:.*#!/:/' | column -t -s":"

.PHONY: install
install: #! Install package dependencies
	uv sync --group dev

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
	uv build

.PHONY: docs
docs: #! Generate sphinx-based documentation
	uv sync --group docs
	uv run $(SPHINXBUILD) -M html "$(DOCS_DIR)" "$(DOCS_BUILD_DIR)" $(SPHINXOPTS)
