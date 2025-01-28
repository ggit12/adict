# llm_providers.py
"""
Manages configurations for different LLM providers supported by LangChain, including model paths,
class names, and initialization methods.
"""
from dataclasses import dataclass
from typing import Dict
from .llm_provider_base import DefaultLLMInitializer, LLMProviderConfig
from .llm_provider_implementations import (  # type: ignore
    BedrockLLMInitializer,
    AzureMLLLMInitializer,
    GoogleGenAILLMInitializer,
)


@dataclass
class LLMProviders:
    """Collection of LLM provider configurations"""

    @classmethod
    def get_providers(cls) -> Dict[str, LLMProviderConfig]:
        """Returns a dictionary mapping provider names to their LLM configurations.

        Provider configurations include the LangChain chat model class name, module path,
        and initializer class used to instantiate the model.

        Returns:
            Dict[str, LLMProviderConfig]: Map of provider names (e.g. 'openai', 'anthropic')
                to their corresponding LLMProviderConfig objects containing:
                - class_name (str): Name of LangChain chat model class
                - module_path (str): Import path for the chat model module
                - init_class (Type): Class used to initialize the chat model
        """
        return {
            "openai": LLMProviderConfig(
                class_name="ChatOpenAI",
                module_path="langchain_openai.chat_models",
                init_class=DefaultLLMInitializer,
            ),
            "anthropic": LLMProviderConfig(
                class_name="ChatAnthropic",
                module_path="langchain_anthropic.chat_models",
                init_class=DefaultLLMInitializer,
            ),
            "bedrock": LLMProviderConfig(
                class_name="ChatBedrockConverse",
                module_path="langchain_aws.chat_models.bedrock_converse",
                init_class=BedrockLLMInitializer,
            ),
            "google": LLMProviderConfig(
                class_name="ChatGoogleGenerativeAI",
                module_path="langchain_google_genai.chat_models",
                init_class=GoogleGenAILLMInitializer,
            ),
            "azureml_endpoint": LLMProviderConfig(
                class_name="AzureMLChatOnlineEndpoint",
                module_path="langchain_community.chat_models.azureml_endpoint",
                init_class=AzureMLLLMInitializer,
            ),
            "azure_openai": LLMProviderConfig(
                class_name="AzureChatOpenAI",
                module_path="langchain_community.chat_models.azure_openai",
                init_class=DefaultLLMInitializer,
            ),
            "cohere": LLMProviderConfig(
                class_name="ChatCohere",
                module_path="langchain_community.chat_models.cohere",
                init_class=DefaultLLMInitializer,
            ),
            "huggingface": LLMProviderConfig(
                class_name="ChatHuggingFace",
                module_path="langchain_community.chat_models.huggingface",
                init_class=DefaultLLMInitializer,
            ),
            "vertexai": LLMProviderConfig(
                class_name="ChatVertexAI",
                module_path="langchain_community.chat_models.vertexai",
                init_class=DefaultLLMInitializer,
            ),
            "ollama": LLMProviderConfig(
                class_name="ChatOllama",
                module_path="langchain_community.chat_models.ollama",
                init_class=DefaultLLMInitializer,
            ),
        }
