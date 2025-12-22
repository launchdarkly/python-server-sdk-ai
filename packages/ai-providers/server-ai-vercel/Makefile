PYTEST_FLAGS=-W error::SyntaxWarning

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
	poetry run mypy src/ldai_vercel
	poetry run isort --check --atomic src/ldai_vercel
	poetry run pycodestyle src/ldai_vercel

.PHONY: build
build: #! Build distribution files
build: install
	poetry build

