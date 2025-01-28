# llm_manager.py
"""
A manager class for configuring and interacting with
Language Learning Models (LLMs) through a unified interface.
Handles provider configuration, initialization, and message passing using environment variables.
"""
import os
import importlib
from typing import Any, Dict, List, Optional, Tuple
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from .llm_providers import LLMProviders  # type: ignore


class LLMManager:
    """Internal manager class for LLM operations"""

    @staticmethod
    def configure_llm_backend(provider: str,
    model: str,
    **kwargs
    ) -> None:
        """
        Call this function before using LLM integrations. Configures the LLM backend by setting environment variables.

        Parameters
        -----------
            provider
                The LLM provider name. Run :func:`LLMProviders.get_providers` to view list of supported providers.
            model
                The LLM model name.
            **kwargs
                Additional configuration parameters passed to :func:`LLMManager.configure_llm_backend`

        Examples
        -----------
        **General** (for most providers)

        .. code-block:: python

            configure_llm_backend('your-provider-name',
                'your-provider-model-name',
                api_key='your-provider-api-key'
            )

        **OpenAI**

        .. code-block:: python

            configure_llm_backend('openai',
                'gpt-3.5-turbo',
                api_key='your-openai-api-key'
            )

        **Anthropic**

        .. code-block:: python

            configure_llm_backend('anthropic',
                'claude-3-5-sonnet-20240620',
                api_key='your-anthropic-api-key'
            )

        **Google**

        .. code-block:: python

            configure_llm_backend('google',
                'gemini-1.5-pro',
                api_key='your-google_ai-api-key'
            )

        **Bedrock**

        .. code-block:: python

            configure_llm_backend('bedrock',
                'anthropic.claude-v2',
                region_name='us-west-2',
                aws_access_key_id='your-access-key-id',
                aws_secret_access_key='your-secret-access-key'
            )

        **AzureML Endpoint**

        .. code-block:: python

            configure_llm_backend('azureml_endpoint',
                'llama-2',
                endpoint_name='your-endpoint-name',
                region='your-region',
                api_key='your-api-key'
            )

        """
        providers = LLMProviders.get_providers()
        if provider.lower() not in providers:
            raise ValueError(f"Unsupported provider: {provider}")

        # Clean up old LLM_ environment variables
        for key in list(os.environ.keys()):
            if key.startswith("LLM_"):
                del os.environ[key]

        os.environ["LLM_PROVIDER"] = provider.lower()
        os.environ["LLM_MODEL"] = model

        for key, value in kwargs.items():
            os.environ[f"LLM_{key.upper()}"] = str(value)

    @staticmethod
    def get_llm_config() -> Dict[str, Any]:
        """
        Retrieves the LLM configuration from environment variables.

        This function loads the LLM provider and model from environment variables,
        validates the provider against available providers, and constructs a
        configuration dictionary. It also includes any additional environment
        variables prefixed with ``'LLM_'`` in the configuration.

        Returns
        --------
            A dictionary containing LLM configuration with the following keys:

            provider : :class:`str`
                The LLM provider name from ``LLM_PROVIDER`` env var

            model : :class:`str`
                The model identifier from ``LLM_MODEL`` env var

            class : :class:`str`
                The provider's class name for instantiation

            module : :class:`str`
                The provider module's path

            Additional keys are included from any environment variables
            prefixed with ``'LLM_'``, excluding ``LLM_PROVIDER`` and ``LLM_MODEL``

        Raises
        -------
        ValueError
            If the specified provider is not in the list of supported providers

        Examples
        --------
        .. code-block:: python

            configure_llm_backend('openai',
                'gpt-3.5-turbo',
                api_key='your-openai-api-key'
            )

            config = get_llm_config()
            print(config['provider'])
            > 'openai'
            print(config['model'])
            > 'gpt-3.5-turbo'

        """
        provider = os.getenv("LLM_PROVIDER")
        model = os.getenv("LLM_MODEL")
        providers = LLMProviders.get_providers()

        if provider not in providers:
            raise ValueError(f"Unsupported provider: {provider}")

        provider_config = providers[provider]
        config = {
            "provider": provider,
            "model": model,
            "class": provider_config.class_name,
            "module": provider_config.module_path,
        }

        # Add all LLM_ prefixed environment variables to the config
        for key, value in os.environ.items():
            if key.startswith("LLM_") and key not in ["LLM_PROVIDER", "LLM_MODEL"]:
                config[key[4:].lower()] = value

        return config

    @staticmethod
    def get_llm(
        instance: Optional[Any], config: Optional[Dict[str, Any]], **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """Dynamically retrieves the appropriate LLM based on the configuration."""
        current_config = LLMManager.get_llm_config()

        # Check if instance exists and config hasn't changed
        if instance is not None and config == current_config:
            return instance, config

        try:
            module = importlib.import_module(current_config["module"])
            llm_class = getattr(module, current_config["class"])

            # Remove 'class' and 'module' from config before passing to constructor
            constructor_args = {
                k: v
                for k, v in current_config.items()
                if k not in ["class", "module", "provider"]
            }

            # Get provider config and initialize
            providers = LLMProviders.get_providers()
            provider_config = providers[current_config["provider"]]
            initializer = provider_config.init_class(provider_config)

            constructor_args, kwargs = initializer.initialize(
                constructor_args, **kwargs
            )

            instance = llm_class(**constructor_args)
            return instance, current_config

        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Error initializing provider {current_config['provider']}: {str(e)}"
            ) from e

    @staticmethod
    def call_llm(messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Call the configured LLM model with given messages and parameters.

        Parameters
        -----------
        messages
            List of message dictionaries, where each dictionary contains:
                'role' : :class:`str`
                    The role of the message sender (``'system'``, ``'user'``, or ``'assistant'``)
                'content' : :class:`str`
                    The content of the message
        **kwargs
            Additional keyword arguments passed to the LLM provider.

            Common parameters include:
                supports_system_messages : :class:`bool`, optional
                    Whether the model supports system messages (default: ``True``)

        Returns
        --------
            The stripped content of the LLM's response.

        Notes
        ------
        The function performs the following steps:
        1. Gets LLM configuration and initializes the provider
        2. Converts messages to appropriate ``LangChain`` message types
        3. Calls the LLM with the processed messages
        4. Writes the response to a file specified by ``RESPONSE_PATH`` env variable
        
        The response is written to ``'./response.txt'`` by default if ``RESPONSE_PATH`` is not set.

        See Also
        --------
        LLMManager : Class handling LLM configuration and use.
        :func:`LLMProviders.get_providers` : To see supported providers.
        """
        config = LLMManager.get_llm_config()
        llm, _ = LLMManager.get_llm(None, None, **kwargs)

        # Get provider config and initialize
        providers = LLMProviders.get_providers()
        provider_config = providers[config["provider"]]
        initializer = provider_config.init_class(provider_config)

        # Get additional kwargs from initializer
        _, kwargs = initializer.initialize({}, **kwargs)

        # Check if this model doesn't support system messages
        supports_system_messages = kwargs.pop("supports_system_messages", True)

        message_types = {
            "system": (
                SystemMessage if supports_system_messages is not False else HumanMessage
            ),
            "user": HumanMessage,
            "assistant": AIMessage,
        }

        langchain_messages = [
            message_types.get(msg["role"], HumanMessage)(content=msg["content"])
            for msg in messages
        ]

        response = llm(langchain_messages, **kwargs)

        # Write the response to a file
        with open(
            os.getenv("RESPONSE_PATH", "response.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(f"{response}\n")

        return response.content.strip()

    @staticmethod
    def retry_call_llm(
       messages: List[Dict[str, str]],
       process_response,
       failure_handler,
       max_attempts: int = 5,
       call_llm_kwargs: Optional[Dict[str, Any]] = None,
       process_response_kwargs: Optional[Dict[str, Any]] = None,
       failure_handler_kwargs: Optional[Dict[str, Any]] = None,
   ) -> Any:
        """
        A retry wrapper for call_llm that handles response processing and failures.

        Parameters
        -----------
        messages
            List of message dictionaries, where each dictionary contains:
            - 'role' : :class:`str`
                The role of the message sender (``'system'``, ``'user'``, or ``'assistant'``)
            - 'content' : :class:`str`
                The content of the message
        process_response
            Function to process the LLM response. Should accept response string as first argument.
            If processing fails, triggers retry logic.
        failure_handler
            Function called after all retry attempts are exhausted.
            Should handle the complete failure case.
        max_attempts
            Maximum number of retry attempts (default: ``5``)
        call_llm_kwargs
            Keyword arguments passed to :func:`call_llm` (default: ``None``)
            If contains ``'temperature'``, it's adjusted on retries:
            - Attempts 1-2: temperature = 0
            - Attempts 3+: temperature = (attempt - 2) * 0.025
        process_response_kwargs
            Keyword arguments passed to process_response function (default: ``None``)
        failure_handler_kwargs
            Keyword arguments passed to failure_handler function (default: ``None``)

        Returns
        --------
        The processed result if successful, or the failure handler's return value.

        See Also
        --------
        :func:`call_llm` : The wrapped LLM call function.
        """
        call_llm_kwargs = call_llm_kwargs or {}
        process_response_kwargs = process_response_kwargs or {}
        failure_handler_kwargs = failure_handler_kwargs or {}

        for attempt in range(1, max_attempts + 1):
            # Adjust temperature if it's in call_llm_kwargs
            if "temperature" in call_llm_kwargs:
                call_llm_kwargs["temperature"] = (
                    0 if attempt <= 2 else (attempt - 2) * 0.025
                )

            # Call the LLM
            response = LLMManager.call_llm(messages=messages, **call_llm_kwargs)

            # Attempt to process the response
            try:
                processed_result = process_response(response, **process_response_kwargs)
                return processed_result
            except Exception as e:
                print(f"Attempt {attempt} failed: {str(e)}. Retrying...")
                print(f"Response from failed attempt:\n{response}")

        # If we've exhausted all attempts, call the failure handler
        print(f"All {max_attempts} attempts failed. Calling failure handler.")
        return failure_handler(**failure_handler_kwargs)
