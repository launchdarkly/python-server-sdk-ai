"""Tests for RunnerFactory provider loading and error messages."""

from unittest.mock import MagicMock, patch

import pytest

from ldai.providers.ai_provider import AIProvider
from ldai.providers.runner_factory import RunnerFactory, _PYPI_PACKAGE_NAMES


class TestPkgExists:
    """_pkg_exists raises ImportError with the PyPI package name when a module is missing."""

    def test_raises_import_error_with_pypi_name_for_openai(self):
        with patch('ldai.providers.runner_factory.util') as mock_util:
            mock_util.find_spec.return_value = None
            with pytest.raises(ImportError) as exc_info:
                RunnerFactory._pkg_exists('ldai_openai')
        assert 'launchdarkly-server-sdk-ai-openai' in str(exc_info.value)
        assert 'pip install' not in str(exc_info.value)

    def test_raises_import_error_with_pypi_name_for_langchain(self):
        with patch('ldai.providers.runner_factory.util') as mock_util:
            mock_util.find_spec.return_value = None
            with pytest.raises(ImportError) as exc_info:
                RunnerFactory._pkg_exists('ldai_langchain')
        assert 'launchdarkly-server-sdk-ai-langchain' in str(exc_info.value)
        assert 'pip install' not in str(exc_info.value)

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

    def test_warning_does_not_reference_pip(self):
        """Warning should be package-manager agnostic — no pip install command."""
        with patch('ldai.providers.runner_factory.util') as mock_util, \
             patch('ldai.providers.runner_factory.log') as mock_log:
            mock_util.find_spec.return_value = None
            RunnerFactory._get_provider_factory('openai')
        warning_text = mock_log.warning.call_args[0][0]
        assert 'pip install' not in warning_text


class TestCreateModelMultiTurn:
    """create_model forwards the multi_turn flag to the provider factory."""

    def _make_config(self, provider_name='openai'):
        config = MagicMock()
        config.provider = MagicMock()
        config.provider.name = provider_name
        return config

    def test_forwards_multi_turn_false_to_provider(self):
        sentinel_runner = object()
        provider_factory = MagicMock(spec=AIProvider)
        provider_factory.create_model.return_value = sentinel_runner

        with patch.object(
            RunnerFactory, '_get_provider_factory', return_value=provider_factory
        ):
            result = RunnerFactory.create_model(
                self._make_config(), default_ai_provider='openai', multi_turn=False
            )

        assert result is sentinel_runner
        provider_factory.create_model.assert_called_once()
        _, kwargs = provider_factory.create_model.call_args
        assert kwargs.get('multi_turn') is False

    def test_defaults_multi_turn_to_true(self):
        sentinel_runner = object()
        provider_factory = MagicMock(spec=AIProvider)
        provider_factory.create_model.return_value = sentinel_runner

        with patch.object(
            RunnerFactory, '_get_provider_factory', return_value=provider_factory
        ):
            RunnerFactory.create_model(self._make_config(), default_ai_provider='openai')

        _, kwargs = provider_factory.create_model.call_args
        assert kwargs.get('multi_turn') is True

    def test_constructed_runner_has_multi_turn_false_attribute(self):
        """End-to-end: when multi_turn=False is passed, the constructed runner has _multi_turn == False."""
        # Use a stub AIProvider that constructs an OpenAIModelRunner-shaped object.
        class _StubProvider(AIProvider):
            def __init__(self):
                self.last_kwargs = None

            def create_model(self, config, multi_turn: bool = True):
                self.last_kwargs = {'multi_turn': multi_turn}
                runner = MagicMock()
                runner._multi_turn = multi_turn
                return runner

        stub = _StubProvider()
        with patch.object(RunnerFactory, '_get_provider_factory', return_value=stub):
            runner = RunnerFactory.create_model(
                self._make_config(), default_ai_provider='openai', multi_turn=False
            )

        assert runner is not None
        assert runner._multi_turn is False
