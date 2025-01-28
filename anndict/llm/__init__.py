"""
LLM Configuration Module

This module handles the configuration and initialization of different LLM providers through 
a unified interface. It manages provider configurations, initialization strategies, and 
rate limiting for each supported LLM provider.

The module supports dynamic configuration of various LLM backends including:
- OpenAI
- Anthropic
- AWS Bedrock
- Google AI
- Azure OpenAI
- Azure ML endpoints
- Cohere
- HuggingFace
- Vertex AI
- Ollama

Each provider can be configured with custom initialization parameters, rate limits,
and provider-specific settings. The module handles environment variable management,
provider-specific initializations, and maintains consistent interfaces across providers.

Key Components:
- Provider configuration using dataclasses
- Abstract base classes for provider initialization
- Rate limiting configuration
- Environment variable management
- Provider-specific initialization strategies

The module is used internally by the anndict package and shouldn't be imported directly
by end users. Instead, use the main package interface:
    import anndict as adt
    adt.configure_llm_backend(...)
"""

from .llm_call import (  # type: ignore
    configure_llm_backend,
    get_llm_config,
    call_llm,
    retry_call_llm,

)

from .parse_llm_response import (
    extract_dictionary_from_ai_string,
    extract_list_from_ai_string,
    process_llm_category_mapping,

)

from .llm_providers import LLMProviders

__all__ = [

    #llm_call.py
    "configure_llm_backend",
    "get_llm_config",
    "call_llm",
    "retry_call_llm",

    #parse_llm_response.py
    "extract_dictionary_from_ai_string",
    "extract_list_from_ai_string",
    "process_llm_category_mapping",

    #llm_providers.py
    "LLMProviders",
]
