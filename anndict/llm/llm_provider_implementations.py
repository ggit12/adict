# llm_provider_implementations.py
"""
Implements LLM provider initialization for AWS Bedrock, Azure ML, and Google Generative AI.

Classes:
    BedrockLLMInitializer: AWS Bedrock initialization and configuration
    AzureMLLLMInitializer: Azure ML endpoint setup and configuration 
    GoogleGenAILLMInitializer: Google Generative AI initialization

Required env vars for Bedrock:
    LLM_REGION_NAME
    LLM_AWS_ACCESS_KEY_ID  
    LLM_AWS_SECRET_ACCESS_KEY
    LLM_MODEL
"""
import os
from typing import Any, Dict, Tuple
import boto3
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    LlamaChatContentFormatter,
)
from .llm_provider_base import BaseLLMInitializer


class BedrockLLMInitializer(BaseLLMInitializer):
    """Initialization logic for AWS Bedrock"""

    MODELS_WITHOUT_SYSTEM_MESSAGES = {
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
        "ai21.j2-ultra-v1",
    }

    def initialize(
        self, constructor_args: Dict[str, Any], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        region_name = os.environ.get("LLM_REGION_NAME")
        aws_access_key_id = os.environ.get("LLM_AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("LLM_AWS_SECRET_ACCESS_KEY")
        model_id = os.environ.get("LLM_MODEL")

        if not region_name:
            raise ValueError(
                "Bedrock requires LLM_REGION_NAME to be set in environment variables."
            )
        if not aws_access_key_id:
            raise ValueError(
                "Bedrock requires LLM_AWS_ACCESS_KEY_ID to be set in environment variables."
            )
        if not aws_secret_access_key:
            raise ValueError(
                "Bedrock requires LLM_AWS_SECRET_ACCESS_KEY to be set in environment variables."
            )
        if not model_id:
            raise ValueError("model_id (LLM_MODEL) is required for BedrockChat")

        kwargs["supports_system_messages"] = (
            model_id not in self.MODELS_WITHOUT_SYSTEM_MESSAGES
        )

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

        bedrock_client = session.client("bedrock-runtime")
        allowed_params = ["model", "client", "streaming", "callbacks"]
        filtered_args = {
            k: v for k, v in constructor_args.items() if k in allowed_params
        }
        filtered_args["client"] = bedrock_client

        rate_limiter = self.create_rate_limiter(constructor_args)
        filtered_args["rate_limiter"] = rate_limiter

        return filtered_args, kwargs


class AzureMLLLMInitializer(BaseLLMInitializer):
    """Initialization logic for Azure ML endpoints"""

    def initialize(
        self, constructor_args: Dict[str, Any], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        endpoint_name = constructor_args.pop("endpoint_name", None)
        region = constructor_args.pop("region", None)
        api_key = constructor_args.pop("api_key", None)

        if not all([endpoint_name, region, api_key]):
            raise ValueError(
                "AzureML requires endpoint_name, region, and api_key to be set."
            )

        constructor_args["endpoint_url"] = (
            f"https://{endpoint_name}.{region}.inference.ai.azure.com/v1/chat/completions"
        )
        constructor_args["endpoint_api_type"] = AzureMLEndpointApiType.serverless
        constructor_args["endpoint_api_key"] = api_key
        constructor_args["content_formatter"] = LlamaChatContentFormatter()

        return constructor_args, kwargs


class GoogleGenAILLMInitializer(BaseLLMInitializer):
    """Initialization logic for Google Generative AI"""

    def initialize(
        self, constructor_args: Dict[str, Any], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if "max_tokens" in kwargs:
            constructor_args["max_output_tokens"] = kwargs.pop("max_tokens")
        if "temperature" in kwargs:
            constructor_args["temperature"] = kwargs.pop("temperature")

        kwargs["supports_system_messages"] = False
        os.environ["GRPC_VERBOSITY"] = "ERROR"

        rate_limiter = self.create_rate_limiter(constructor_args)
        constructor_args["rate_limiter"] = rate_limiter

        return constructor_args, {}
