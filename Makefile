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

#
# Installation targets
#

.PHONY: install
install: #! Install all packages
	$(MAKE) -C $(SERVER_AI_PKG) install
	$(MAKE) -C $(LANGCHAIN_PKG) install

.PHONY: install-server-ai
install-server-ai: #! Install server-ai package
	$(MAKE) -C $(SERVER_AI_PKG) install

.PHONY: install-langchain
install-langchain: #! Install langchain provider package
	$(MAKE) -C $(LANGCHAIN_PKG) install

#
# Quality control checks
#

.PHONY: test
test: #! Run unit tests for all packages
	$(MAKE) -C $(SERVER_AI_PKG) test

.PHONY: test-server-ai
test-server-ai: #! Run unit tests for server-ai package
	$(MAKE) -C $(SERVER_AI_PKG) test

.PHONY: test-langchain
test-langchain: #! Run unit tests for langchain provider package
	$(MAKE) -C $(LANGCHAIN_PKG) test

.PHONY: lint
lint: #! Run type analysis and linting checks for all packages
	$(MAKE) -C $(SERVER_AI_PKG) lint

.PHONY: lint-server-ai
lint-server-ai: #! Run type analysis and linting checks for server-ai package
	$(MAKE) -C $(SERVER_AI_PKG) lint

.PHONY: lint-langchain
lint-langchain: #! Run type analysis and linting checks for langchain provider package
	$(MAKE) -C $(LANGCHAIN_PKG) lint

#
# Build targets
#

.PHONY: build
build: #! Build all packages
	$(MAKE) -C $(SERVER_AI_PKG) build

.PHONY: build-server-ai
build-server-ai: #! Build server-ai package
	$(MAKE) -C $(SERVER_AI_PKG) build

.PHONY: build-langchain
build-langchain: #! Build langchain provider package
	$(MAKE) -C $(LANGCHAIN_PKG) build

#
# Documentation generation
#

.PHONY: docs
docs: #! Generate sphinx-based documentation
	$(MAKE) -C $(SERVER_AI_PKG) docs DOCS_DIR=../../../$(SOURCEDIR) DOCS_BUILD_DIR=../../../$(BUILDDIR)
