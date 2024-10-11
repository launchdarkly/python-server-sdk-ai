import json
from typing import Any, Dict, Optional
from ldclient import Context
#from config import LDAIConfig, LDAIConfigTracker
from ldclient.client import LDClient
import chevron

from ldai.tracker import LDAIConfigTracker

class LDAIClient:
    """The LaunchDarkly AI SDK client object."""

    def __init__(self, client: LDClient):
        self.client = client

    def model_config(self, key: str, context: Context, default_value: str, variables: Optional[Dict[str, Any]] = None) -> Any:
        """Get the value of a model configuration asynchronously.

        Args:
            key: The key of the model configuration.
            context: The context to evaluate the model configuration in.
            default_value: The default value of the model configuration.
            variables: Additional variables for the model configuration.

        Returns:
            The value of the model configuration.
        """
        variation = self.client.variation(key, context, default_value)

        all_variables = {'ldctx': context}
        if variables:
            all_variables.update(variables)

        print(variation)
        variation['prompt'] = [
            {
                **entry,
                'content': self.interpolate_template(entry['content'], all_variables)
            }
            for entry in variation['prompt']
        ]

        #return detail.value,

        return {
            'config': variation,
            'tracker': LDAIConfigTracker(self.client, variation['_ldMeta']['variationId'], key, context)
        }

    def interpolate_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Interpolate the template with the given variables.

        Args:
            template: The template string.
            variables: The variables to interpolate into the template.

        Returns:
            The interpolated string.
        """
        return chevron.render(template, variables)