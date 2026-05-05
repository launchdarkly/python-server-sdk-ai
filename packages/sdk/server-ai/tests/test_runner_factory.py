"""Tests for RunnerFactory provider loading and error messages."""

from unittest.mock import patch

import pytest

from ldai.providers.runner_factory import RunnerFactory, _PYPI_PACKAGE_NAMES


class TestPkgExists:
    """_pkg_exists raises ImportError with the PyPI package name when a module is missing."""

    def test_raises_import_error_with_pypi_name_for_openai(self):
        with patch('ldai.providers.runner_factory.util') as mock_util:
            mock_util.find_spec.return_value = None
            with pytest.raises(ImportError) as exc_info:
                RunnerFactory._pkg_exists('ldai_openai')
        assert 'launchdarkly-server-sdk-ai-openai' in str(exc_info.value)
        assert 'pip install launchdarkly-server-sdk-ai-openai' in str(exc_info.value)

    def test_raises_import_error_with_pypi_name_for_langchain(self):
        with patch('ldai.providers.runner_factory.util') as mock_util:
            mock_util.find_spec.return_value = None
            with pytest.raises(ImportError) as exc_info:
                RunnerFactory._pkg_exists('ldai_langchain')
        assert 'launchdarkly-server-sdk-ai-langchain' in str(exc_info.value)
        assert 'pip install launchdarkly-server-sdk-ai-langchain' in str(exc_info.value)

    def test_raises_import_error_with_module_name_when_no_mapping(self):
        """Unknown module names fall back to the module name itself."""
        with patch('ldai.providers.runner_factory.util') as mock_util:
            mock_util.find_spec.return_value = None
            with pytest.raises(ImportError) as exc_info:
                RunnerFactory._pkg_exists('some_unknown_module')
        assert 'some_unknown_module' in str(exc_info.value)

    def test_does_not_raise_when_package_is_found(self):
        with patch('ldai.providers.runner_factory.util') as mock_util:
            mock_util.find_spec.return_value = object()  # non-None means found
            # Should not raise
            RunnerFactory._pkg_exists('ldai_openai')


class TestPypiPackageNameMapping:
    """The _PYPI_PACKAGE_NAMES mapping covers all supported providers."""

    def test_openai_module_maps_to_pypi_name(self):
        assert _PYPI_PACKAGE_NAMES['ldai_openai'] == 'launchdarkly-server-sdk-ai-openai'

    def test_langchain_module_maps_to_pypi_name(self):
        assert _PYPI_PACKAGE_NAMES['ldai_langchain'] == 'launchdarkly-server-sdk-ai-langchain'


class TestGetProviderFactory:
    """_get_provider_factory logs the PyPI package name in its warning when a package is missing."""

    def test_warning_includes_pypi_name_for_openai(self):
        with patch('ldai.providers.runner_factory.util') as mock_util, \
             patch('ldai.providers.runner_factory.log') as mock_log:
            mock_util.find_spec.return_value = None
            result = RunnerFactory._get_provider_factory('openai')
        assert result is None
        warning_text = mock_log.warning.call_args[0][0]
        assert 'launchdarkly-server-sdk-ai-openai' in warning_text
        assert 'ldai_openai' not in warning_text

    def test_warning_includes_pypi_name_for_langchain(self):
        with patch('ldai.providers.runner_factory.util') as mock_util, \
             patch('ldai.providers.runner_factory.log') as mock_log:
            mock_util.find_spec.return_value = None
            result = RunnerFactory._get_provider_factory('langchain')
        assert result is None
        warning_text = mock_log.warning.call_args[0][0]
        assert 'launchdarkly-server-sdk-ai-langchain' in warning_text
        assert 'ldai_langchain' not in warning_text

    def test_warning_does_not_contain_make_sure_text(self):
        """The old boilerplate text should be replaced by actionable pip install instructions."""
        with patch('ldai.providers.runner_factory.util') as mock_util, \
             patch('ldai.providers.runner_factory.log') as mock_log:
            mock_util.find_spec.return_value = None
            RunnerFactory._get_provider_factory('openai')
        warning_text = mock_log.warning.call_args[0][0]
        assert 'Make sure the corresponding package is installed' not in warning_text
