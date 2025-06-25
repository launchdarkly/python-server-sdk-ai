# Contributing to the LaunchDarkly Server-side AI library for Python

LaunchDarkly has published an [SDK contributor's guide](https://docs.launchdarkly.com/sdk/concepts/contributors-guide) that provides a detailed explanation of how our SDKs work. See below for additional information on how to contribute to this SDK.

## Submitting bug reports and feature requests

The LaunchDarkly SDK team monitors the [issue tracker](https://github.com/launchdarkly/python-server-sdk-ai/issues) in the SDK repository. Bug reports and feature requests specific to this library should be filed in this issue tracker. The SDK team will respond to all newly filed issues within two business days.

## Submitting pull requests

We encourage pull requests and other contributions from the community. Before submitting pull requests, ensure that all temporary or unintended code is removed. Don't worry about adding reviewers to the pull request; the LaunchDarkly SDK team will add themselves. The SDK team will acknowledge all pull requests within two business days.

## Build instructions

### Setup

This project is built using [poetry](https://python-poetry.org/). To learn more about the basics of working with this tool, read [Poetry's basic usage guide](https://python-poetry.org/docs/basic-usage/).

To begin development, ensure your dependencies are installed and (optionally) activate the virtualenv.

```
poetry install
eval $(poetry env activate)
```

### Testing

To run all unit tests:

```shell
make test
```

It is preferable to run tests against all supported minor versions of Python (as described in `README.md` under Requirements), or at least the lowest and highest versions, prior to submitting a pull request. However, LaunchDarkly's CI tests will run automatically against all supported versions.

### Building documentation

See "Documenting types and methods" below. To build the documentation locally, so you can see the effects of any changes before a release:

```shell
make docs
```

The output will appear in `docs/build/html`. Its formatting will be somewhat different since it does not have the same stylesheets used on readthedocs.io.

### Running the linter

The `mypy` tool is used in CI to verify type hints and warn of potential code problems. To run it locally:

```shell
make lint
```

## Code organization

The library's module structure is as follows:

### Type hints

Python does not require the use of type hints, but they can be extremely helpful for spotting mistakes and for improving the IDE experience, so we should always use them in the library. Every method in the public API is expected to have type hints for all non-`self` parameters, and for its return value if any.

It's also desirable to use type hints for private attributes, to catch possible mistakes in their use. Until all versions of Python that we support allow the PEP 526 syntax for doing this, we must do it via a comment in the format that `mypy` understands, for instance:

```python
    self._some_attribute = None  # type: Optional[int]
```

## Documenting types and methods

Please try to make the style and terminology in documentation comments consistent with other documentation comments in the library. Also, if a class or method is being added that has an equivalent in other libraries, and if we have described it in a consistent away in those other libraries, please reuse the text whenever possible (with adjustments for anything language-specific) rather than writing new text.
