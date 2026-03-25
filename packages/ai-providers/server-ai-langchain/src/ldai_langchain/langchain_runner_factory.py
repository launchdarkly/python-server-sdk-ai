from ldai.models import AIConfigKind
from ldai.providers import AIProvider
from ldai_langchain.langchain_helper import LangChainHelper
from ldai_langchain.langchain_model_runner import LangChainModelRunner


class LangChainRunnerFactory(AIProvider):
    """
    LangChain connector for the LaunchDarkly AI SDK.

    Acts as a per-provider factory. Instantiate with no arguments, then call
    ``create_model(config)`` to obtain a configured ``LangChainModelRunner``.
    """

    def create_model(self, config: AIConfigKind) -> LangChainModelRunner:
        """
        Create a configured LangChainModelRunner for the given AI config.

        :param config: The LaunchDarkly AI configuration
        :return: LangChainModelRunner ready to invoke the model
        """
        llm = LangChainHelper.create_langchain_model(config)
        return LangChainModelRunner(llm)
