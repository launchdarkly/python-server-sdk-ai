PYTEST_FLAGS=-W error::SyntaxWarning

SPHINXOPTS    = -W --keep-going
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = launchdarkly-server-sdk
SOURCEDIR     = docs
BUILDDIR      = $(SOURCEDIR)/build

# Package paths
SERVER_AI_PKG = packages/sdk/server-ai
LANGCHAIN_PKG = packages/ai-providers/server-ai-langchain

.PHONY: help
help: #! Show this help message
	@echo 'Usage: make [target] ... '
	@echo ''
	@echo 'Targets:'
	@grep -h -F '#!' $(MAKEFILE_LIST) | grep -v grep | sed 's/:.*#!/:/' | column -t -s":"

.PHONY: install
install: #! Install all packages
	@cd $(SERVER_AI_PKG) && poetry install
	@cd $(LANGCHAIN_PKG) && poetry install

.PHONY: install-server-ai
install-server-ai: #! Install server-ai package
	@cd $(SERVER_AI_PKG) && poetry install

.PHONY: install-langchain
install-langchain: #! Install langchain provider package
	@cd $(LANGCHAIN_PKG) && poetry install

#
# Quality control checks
#

.PHONY: test
test: #! Run unit tests for all packages
test: test-server-ai

.PHONY: test-server-ai
test-server-ai: #! Run unit tests for server-ai package
test-server-ai: install-server-ai
	@cd $(SERVER_AI_PKG) && poetry run pytest $(PYTEST_FLAGS)

.PHONY: test-langchain
test-langchain: #! Run unit tests for langchain provider package
test-langchain: install-langchain
	@cd $(LANGCHAIN_PKG) && poetry run pytest $(PYTEST_FLAGS)

.PHONY: lint
lint: #! Run type analysis and linting checks for all packages
lint: lint-server-ai

.PHONY: lint-server-ai
lint-server-ai: #! Run type analysis and linting checks for server-ai package
lint-server-ai: install-server-ai
	@cd $(SERVER_AI_PKG) && poetry run mypy src/ldai
	@cd $(SERVER_AI_PKG) && poetry run isort --check --atomic src/ldai
	@cd $(SERVER_AI_PKG) && poetry run pycodestyle src/ldai

.PHONY: lint-langchain
lint-langchain: #! Run type analysis and linting checks for langchain provider package
lint-langchain: install-langchain
	@cd $(LANGCHAIN_PKG) && poetry run mypy src/ldai_langchain
	@cd $(LANGCHAIN_PKG) && poetry run pycodestyle src/ldai_langchain

#
# Documentation generation
#

.PHONY: docs
docs: #! Generate sphinx-based documentation
	@cd $(SERVER_AI_PKG) && poetry install --with docs
	@cd docs
	@cd $(SERVER_AI_PKG) && poetry run $(SPHINXBUILD) -M html "../../../$(SOURCEDIR)" "../../../$(BUILDDIR)" $(SPHINXOPTS) $(O)
