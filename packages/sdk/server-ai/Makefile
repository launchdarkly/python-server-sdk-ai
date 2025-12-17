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
	poetry install

.PHONY: test
test: #! Run unit tests
test: install
	poetry run pytest $(PYTEST_FLAGS)

.PHONY: lint
lint: #! Run type analysis and linting checks
lint: install
	poetry run mypy src/ldai
	poetry run isort --check --atomic src/ldai
	poetry run pycodestyle src/ldai

.PHONY: build
build: #! Build distribution files
build: install
	poetry build

.PHONY: docs
docs: #! Generate sphinx-based documentation
	poetry install --with docs
	poetry run $(SPHINXBUILD) -M html "$(DOCS_DIR)" "$(DOCS_BUILD_DIR)" $(SPHINXOPTS)
