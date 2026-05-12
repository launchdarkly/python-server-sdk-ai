PYTEST_FLAGS=-W error::SyntaxWarning

# Package paths
SERVER_AI_PKG = packages/sdk/server-ai
LANGCHAIN_PKG = packages/ai-providers/server-ai-langchain
OPENAI_PKG = packages/ai-providers/server-ai-openai

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
install: #! Install all packages and dev dependencies into the shared workspace environment
	uv sync --all-groups

.PHONY: install-server-ai
install-server-ai: #! Install server-ai package
	$(MAKE) -C $(SERVER_AI_PKG) install

.PHONY: install-langchain
install-langchain: #! Install langchain provider package
	$(MAKE) -C $(LANGCHAIN_PKG) install

.PHONY: install-openai
install-openai: #! Install openai provider package
	$(MAKE) -C $(OPENAI_PKG) install

#
# Quality control checks
#

.PHONY: test
test: #! Run unit tests for all packages
	$(MAKE) test-server-ai
	$(MAKE) test-langchain
	$(MAKE) test-openai

.PHONY: test-server-ai
test-server-ai: #! Run unit tests for server-ai package
	$(MAKE) -C $(SERVER_AI_PKG) test

.PHONY: test-langchain
test-langchain: #! Run unit tests for langchain provider package
	$(MAKE) -C $(LANGCHAIN_PKG) test

.PHONY: test-openai
test-openai: #! Run unit tests for openai provider package
	$(MAKE) -C $(OPENAI_PKG) test

.PHONY: lint
lint: #! Run type analysis and linting checks for all packages
	$(MAKE) lint-server-ai
	$(MAKE) lint-langchain
	$(MAKE) lint-openai

.PHONY: lint-server-ai
lint-server-ai: #! Run type analysis and linting checks for server-ai package
	$(MAKE) -C $(SERVER_AI_PKG) lint

.PHONY: lint-langchain
lint-langchain: #! Run type analysis and linting checks for langchain provider package
	$(MAKE) -C $(LANGCHAIN_PKG) lint

.PHONY: lint-openai
lint-openai: #! Run type analysis and linting checks for openai provider package
	$(MAKE) -C $(OPENAI_PKG) lint

#
# Build targets
#

.PHONY: build
build: #! Build all packages
	$(MAKE) build-server-ai
	$(MAKE) build-langchain
	$(MAKE) build-openai

.PHONY: build-server-ai
build-server-ai: #! Build server-ai package
	$(MAKE) -C $(SERVER_AI_PKG) build

.PHONY: build-langchain
build-langchain: #! Build langchain provider package
	$(MAKE) -C $(LANGCHAIN_PKG) build

.PHONY: build-openai
build-openai: #! Build openai provider package
	$(MAKE) -C $(OPENAI_PKG) build

#
# Documentation generation
#

.PHONY: docs
docs: docs-server-ai docs-openai #! Generate sphinx docs for all documented packages

.PHONY: docs-server-ai
docs-server-ai: #! Generate sphinx docs for server-ai package
	$(MAKE) -C $(SERVER_AI_PKG) docs

.PHONY: docs-openai
docs-openai: #! Generate sphinx docs for openai provider package
	$(MAKE) -C $(OPENAI_PKG) docs
