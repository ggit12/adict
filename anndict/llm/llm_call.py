# llm_call
"""
A module for managing interactions with Large Language Models (LLMs) across providers.
Provides configuration, initialization, and retry mechanisms for making LLM calls
through a unified interface that works with OpenAI, Google, Anthropic and others.
"""
from functools import wraps

from typing import Any, Dict, List, Optional
from .llm_manager import LLMManager  # type: ignore

# Global instance management
_llm_instance: Optional[Any] = None
_llm_config: Optional[Dict[str, Any]] = None

@wraps(LLMManager.configure_llm_backend)
def configure_llm_backend(
    provider: str,
    model: str,
    **kwargs
) -> None:
    """Wrapper for LLMManager.configure_llm_backend"""
    LLMManager.configure_llm_backend(provider, model, **kwargs)
    global _llm_instance, _llm_config
    _llm_instance = None
    _llm_config = None


@wraps(LLMManager.get_llm_config)
def get_llm_config() -> Dict[str, Any]:
    """Wrapper for LLMManager.get_llm_config"""
    return LLMManager.get_llm_config()


@wraps(LLMManager.call_llm)
def call_llm(messages: List[Dict[str, str]], **kwargs) -> str:
    """Wrapper for LLMManager.call_llm"""
    return LLMManager.call_llm(messages, **kwargs)


@wraps(LLMManager.retry_call_llm)
def retry_call_llm(
    messages: List[Dict[str, str]],
    process_response,
    failure_handler,
    max_attempts: int = 5,
    call_llm_kwargs: Optional[Dict[str, Any]] = None,
    process_response_kwargs: Optional[Dict[str, Any]] = None,
    failure_handler_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Wrapper for LLMManager.retry_call_llm"""
    return LLMManager.retry_call_llm(
        messages=messages,
        process_response=process_response,
        failure_handler=failure_handler,
        max_attempts=max_attempts,
        call_llm_kwargs=call_llm_kwargs,
        process_response_kwargs=process_response_kwargs,
        failure_handler_kwargs=failure_handler_kwargs,
    )
