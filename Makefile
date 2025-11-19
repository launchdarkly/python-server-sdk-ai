PYTEST_FLAGS=-W error::SyntaxWarning

SPHINXOPTS    = -W --keep-going
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = launchdarkly-server-sdk
SOURCEDIR     = docs
BUILDDIR      = $(SOURCEDIR)/build

.PHONY: help
help: #! Show this help message
	@echo 'Usage: make [target] ... '
	@echo ''
	@echo 'Targets:'
	@grep -h -F '#!' $(MAKEFILE_LIST) | grep -v grep | sed 's/:.*#!/:/' | column -t -s":"

.PHONY: install
install: #! Install all packages
	@echo "Installing core package..."
	@cd packages/core && poetry install
	@echo "Installing langchain package..."
	@cd packages/langchain && poetry install

.PHONY: install-core
install-core: #! Install core package only
	@cd packages/core && poetry install

.PHONY: install-langchain
install-langchain: #! Install langchain package only
	@cd packages/langchain && poetry install

#
# Quality control checks
#

.PHONY: test
test: #! Run unit tests for all packages
	@echo "Testing core package..."
	@cd packages/core && poetry run pytest $(PYTEST_FLAGS)
	@echo "Testing langchain package..."
	@cd packages/langchain && poetry run pytest $(PYTEST_FLAGS)

.PHONY: test-core
test-core: #! Run unit tests for core package
	@cd packages/core && poetry run pytest $(PYTEST_FLAGS)

.PHONY: test-langchain
test-langchain: #! Run unit tests for langchain package
	@cd packages/langchain && poetry run pytest $(PYTEST_FLAGS)

.PHONY: lint
lint: #! Run type analysis and linting checks
	@echo "Linting core package..."
	@cd packages/core && poetry run mypy ldai
	@cd packages/core && poetry run isort --check --atomic ldai
	@cd packages/core && poetry run pycodestyle ldai

.PHONY: build
build: #! Build all packages
	@echo "Building core package..."
	@cd packages/core && poetry build
	@echo "Building langchain package..."
	@cd packages/langchain && poetry build

.PHONY: build-core
build-core: #! Build core package
	@cd packages/core && poetry build

.PHONY: build-langchain
build-langchain: #! Build langchain package
	@cd packages/langchain && poetry build

#
# Documentation generation
#

.PHONY: docs
docs: #! Generate sphinx-based documentation
	@cd packages/core && poetry install --with docs
	@cd docs
	@cd packages/core && poetry run $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
