#AI integration for cell typing, interpretation of gene lists, and other labelling tasks
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import anndata as ad
import os
import re
import pandas as pd
import random
import itertools
from IPython.display import HTML, display

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import subprocess

import base64
import matplotlib.pyplot as plt
from io import BytesIO

import importlib
from typing import List, Dict, Any
from langchain.llms.base import BaseLLM
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
import boto3
import json

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from difflib import get_close_matches
import ast

from .utils import normalize_string, normalize_label
# import time
# import csv
# import threading

#LLM configuration
def bedrock_init(constructor_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Initialization function for Bedrock"""
    # Retrieve values from environment variables
    region_name = os.environ.get('LLM_REGION_NAME')
    aws_access_key_id = os.environ.get('LLM_AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('LLM_AWS_SECRET_ACCESS_KEY')
    model_id = os.environ.get('LLM_MODEL')  # This comes from the 'model' parameter in configure_llm_backend
    
    if not region_name:
        raise ValueError("Bedrock requires LLM_REGION_NAME to be set in environment variables.")
    
    if not model_id:
        raise ValueError("model_id (LLM_MODEL) is required for BedrockChat")
    
    #Define models that do not support system messages
    models_not_supporting_system_messages = {
        'amazon.titan-text-express-v1',
        'amazon.titan-text-lite-v1',
        'ai21.j2-ultra-v1'
    }
    
    #Determine if the specific model does not support system messages
    supports_system_messages = model_id not in models_not_supporting_system_messages
    
    #Add system message support flag to kwargs
    kwargs['supports_system_messages'] = supports_system_messages
    
    #Create a boto3 session with explicit credentials if provided
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    
    bedrock_client = session.client('bedrock-runtime')
    
    #Create a new dict with only the allowed parameters
    allowed_params = ['model', 'client', 'streaming', 'callbacks']
    filtered_args = {k: v for k, v in constructor_args.items() if k in allowed_params}
    
    # filtered_args['model_id'] = model_id
    filtered_args['client'] = bedrock_client

    # Extract rate limiter parameters from kwargs, or use defaults
    requests_per_minute = float(constructor_args.pop('requests_per_minute', 40))
    requests_per_second = requests_per_minute / 60  # Convert to requests per second
    check_every_n_seconds = float(constructor_args.pop('check_every_n_seconds', 0.1))
    max_bucket_size = float(constructor_args.pop('max_bucket_size', requests_per_minute))

    # Add rate limiter specific to Bedrock
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=check_every_n_seconds,
        max_bucket_size=max_bucket_size
    )
    
    # Add rate limiter to the constructor arguments
    filtered_args['rate_limiter'] = rate_limiter

    #Add a debug print to see what's being passed to BedrockChat
    # print(f"BedrockChat constructor args: {json.dumps(filtered_args, default=str, indent=2)}")
    
    return filtered_args, kwargs


def azureml_init(constructor_args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Initialization function for AzureML Endpoint"""
    from langchain_community.chat_models.azureml_endpoint import (
        AzureMLEndpointApiType,
        LlamaChatContentFormatter,
    )
    
    endpoint_name = constructor_args.pop('endpoint_name', None)
    region = constructor_args.pop('region', None)
    api_key = constructor_args.pop('api_key', None)
    
    if not all([endpoint_name, region, api_key]):
        raise ValueError("AzureML requires endpoint_name, region, and api_key to be set.")
    
    constructor_args['endpoint_url'] = f"https://{endpoint_name}.{region}.inference.ai.azure.com/v1/chat/completions"
    constructor_args['endpoint_api_type'] = AzureMLEndpointApiType.serverless
    constructor_args['endpoint_api_key'] = api_key
    constructor_args['content_formatter'] = LlamaChatContentFormatter()
    
    return constructor_args, kwargs

def google_genai_init(constructor_args, **kwargs):
    """Initialization function for Google (Gemini) API. Among other things, sets GRPC_VERBOSITY env variable to ERROR."""
    if 'max_tokens' in kwargs:
        constructor_args['max_output_tokens'] = kwargs.pop('max_tokens')
    if 'temperature' in kwargs:
        constructor_args['temperature'] = kwargs.pop('temperature')

    # constructor_args['convert_system_message_to_human'] = True
    kwargs['supports_system_messages'] = False

    #Google API will send ignorable warnings if you are on mac, so supress them by setting this env var
    os.environ['GRPC_VERBOSITY'] = 'ERROR'

    # Extract rate limiter parameters from kwargs, or use defaults
    requests_per_minute = float(constructor_args.pop('requests_per_minute', 40))
    requests_per_second = requests_per_minute / 60  # Convert to requests per second
    check_every_n_seconds = float(constructor_args.pop('check_every_n_seconds', 0.1))
    max_bucket_size = float(constructor_args.pop('max_bucket_size', requests_per_minute))

    # Add a custom rate limiter for Google Gemini API
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_second, 
        check_every_n_seconds=check_every_n_seconds,
        max_bucket_size=max_bucket_size
    )
    
    # Add rate limiter to constructor arguments
    constructor_args['rate_limiter'] = rate_limiter

    # For Google, we've handled these in the constructor, so we return empty kwargs
    return constructor_args, {}

def default_init(constructor_args, **kwargs):
    """Default initialization function that sets a rate limiter"""

    # Extract rate limiter parameters from kwargs, or use defaults
    requests_per_minute = float(constructor_args.pop('requests_per_minute', 40))
    requests_per_second = requests_per_minute / 60  # Convert to requests per second
    check_every_n_seconds = float(constructor_args.pop('check_every_n_seconds', 0.1))
    max_bucket_size = float(constructor_args.pop('max_bucket_size', requests_per_minute))

    # Add a default rate limiter
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=check_every_n_seconds,
        max_bucket_size=max_bucket_size
    )

    # Add rate limiter to constructor arguments
    constructor_args['rate_limiter'] = rate_limiter

    return constructor_args, kwargs

PROVIDER_MAPPING = {
    'openai': {
        'class': 'ChatOpenAI',
        'module': 'langchain_openai.chat_models',
        'init_func': default_init
    },
    'anthropic': {
        'class': 'ChatAnthropic',
        'module': 'langchain_anthropic.chat_models',
        'init_func': default_init
    },
    'azure_openai': {
        'class': 'AzureChatOpenAI',
        'module': 'langchain_community.chat_models.azure_openai',
        'init_func': default_init
    },
    'azureml_endpoint': {
        'class': 'AzureMLChatOnlineEndpoint',
        'module': 'langchain_community.chat_models.azureml_endpoint',
        'init_func': azureml_init
    },
    'google': {
        'class': 'ChatGoogleGenerativeAI',
        'module': 'langchain_google_genai.chat_models',
        'init_func': google_genai_init
    },
    'google_palm': {
        'class': 'ChatGooglePalm',
        'module': 'langchain_community.chat_models.google_palm',
        'init_func': default_init
    },
    'bedrock': {
        'class': 'ChatBedrockConverse',
        'module': 'langchain_aws.chat_models.bedrock_converse',
        'init_func': bedrock_init
    },
    'cohere': {
        'class': 'ChatCohere',
        'module': 'langchain_community.chat_models.cohere',
        'init_func': default_init
    },
    'huggingface': {
        'class': 'ChatHuggingFace',
        'module': 'langchain_community.chat_models.huggingface',
        'init_func': default_init
    },
    'vertexai': {
        'class': 'ChatVertexAI',
        'module': 'langchain_community.chat_models.vertexai',
        'init_func': default_init
    },
    'ollama': {
        'class': 'ChatOllama',
        'module': 'langchain_community.chat_models.ollama',
        'init_func': default_init
    }
}

#list of models and providers for reference (PROVIDERS and PROVIDER_MODELS are not used in the code):
PROVIDERS = [
    'openai',
    'anthropic',
    'google',
    'mistral',
    'cohere',
    'ai21',
    'huggingface',
    'nvidia_bionemo',
    'ibm_watson',
    'azureml_endpoint',
    'bedrock'
]

PROVIDER_MODELS = {
    'openai': [
        'gpt-4o',  
        'gpt-4o-mini', 
        'gpt-4-0125-preview',  
        'gpt-4-1106-preview',
        'gpt-4-vision-preview',
        'gpt-4',
        'gpt-4-32k',
        'gpt-3.5-turbo-0125',
        'gpt-3.5-turbo-1106',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-16k',
        'text-davinci-003',
        'text-davinci-002',
        'text-curie-001',
        'text-babbage-001',
        'text-ada-001'
    ],
    'anthropic': [
        'claude-3-opus-20240229',
        'claude-3-5-sonnet-20240620',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307',
        'claude-2.1',
        'claude-2.0',
        'claude-instant-1.2'
    ],
    'azureml_endpoint': [
        'Meta-Llama-3.1-405B-Instruct-adt',
        'Meta-Llama-3.1-70B-Instruct-adt',
        'Meta-Llama-3.1-8B-Instruct-adt'
    ],
    'google': [
        'gemini-1.0-pro',
        'gemini-1.5-pro', 
        'gemini-1.5-flash'
    ],
    'bedrock' :[
        'ai21.jamba-instruct-v1:0',
        'ai21.j2-mid-v1',
        'ai21.j2-ultra-v1',
        'amazon.titan-text-express-v1',
        'amazon.titan-text-lite-v1',
        'amazon.titan-text-premier-v1:0',
        'amazon.titan-embed-text-v1',
        'amazon.titan-embed-text-v2:0',
        'amazon.titan-embed-image-v1',
        'amazon.titan-image-generator-v1',
        'amazon.titan-image-generator-v2:0',
        'anthropic.claude-v2',
        'anthropic.claude-v2:1',
        'anthropic.claude-3-sonnet-20240229-v1:0',
        'anthropic.claude-3-5-sonnet-20240620-v1:0',
        'anthropic.claude-3-haiku-20240307-v1:0',
        'anthropic.claude-3-opus-20240229-v1:0',
        'anthropic.claude-instant-v1',
        'cohere.command-text-v14',
        'cohere.command-light-text-v14',
        'cohere.command-r-v1:0',
        'cohere.command-r-plus-v1:0',
        'cohere.embed-english-v3',
        'cohere.embed-multilingual-v3',
        'meta.llama2-13b-chat-v1',
        'meta.llama2-70b-chat-v1',
        'meta.llama3-8b-instruct-v1:0',
        'meta.llama3-70b-instruct-v1:0',
        'meta.llama3-1-8b-instruct-v1:0',
        'meta.llama3-1-70b-instruct-v1:0',
        'meta.llama3-1-405b-instruct-v1:0',
        'mistral.mistral-7b-instruct-v0:2',
        'mistral.mixtral-8x7b-instruct-v0:1',
        'mistral.mistral-large-2402-v1:0',
        'mistral.mistral-large-2407-v1:0',
        'mistral.mistral-small-2402-v1:0',
        'stability.stable-diffusion-xl-v0',
        'stability.stable-diffusion-xl-v1'
    ],
    'huggingface': [
        'falcon-40b',
        'falcon-7b',
        'bloom',
        'gpt-neox-20b',
        'gpt2-xl',
        'gpt2-large',
        'gpt2-medium',
        'gpt2',
        'roberta-large',
        'roberta-base'
    ],
    'nvidia_bionemo': [
        'bionemo-dna-v1',
        'bionemo-protein-v1',
        'bionemo-clinical-v1'
    ],
    'ibm_watson': [
        'watson-assistant',
        'watson-discovery',
        'watson-natural-language-understanding',
        'watson-speech-to-text',
        'watson-text-to-speech'
    ]
}

def configure_llm_backend(provider, model, **kwargs):
    """
    Configures the LLM backend by setting environment variables.
    
    See keys of PROVIDER_MODELS for valid providers.
    
    See values of PROVIDER_MODELS for valid models from each provider.

    To view PROVIDER_MODELS:

    import anndict as adt

    adt.PROVIDER_MODELS

    
    api_key is a provider-specific API key that you will have to obtain from your specified provider

    
    Examples:

        # General (for most providers)

        configure_llm_backend('your-provider-name',
        'your-provider-model-name',
        api_key='your-provider-api-key')

        # For general example (OpenAI), works the same for providers google and anthropic.

        configure_llm_backend('openai', 'gpt-3.5-turbo', api_key='your-openai-api-key')
        configure_llm_backend('anthropic', 'claude-3-5-sonnet-20240620', api_key='your-anthropic-api-key')

        # For AzureML Endpoint

        configure_llm_backend('azureml_endpoint', 'llama-2', endpoint_name='your-endpoint-name', region='your-region', api_key='your-api-key')

        # For Bedrock

        configure_llm_backend('bedrock', 'anthropic.claude-v2', region_name='us-west-2', aws_access_key_id='your-access-key-id', aws_secret_access_key='your-secret-access-key')
    """
    global _llm_instance
    provider_info = PROVIDER_MAPPING.get(provider.lower())
    if not provider_info:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Clean up old LLM_ environment variables
    for key in list(os.environ.keys()):
        if key.startswith('LLM_'):
            del os.environ[key]
    
    os.environ['LLM_PROVIDER'] = provider.lower()
    os.environ['LLM_MODEL'] = model
    
    for key, value in kwargs.items():
        os.environ[f'LLM_{key.upper()}'] = str(value)

    _llm_instance = None


def get_llm_config():
    """Retrieves the LLM configuration from environment variables."""
    provider = os.getenv('LLM_PROVIDER')
    model = os.getenv('LLM_MODEL')
    provider_info = PROVIDER_MAPPING.get(provider)
    
    if not provider_info:
        raise ValueError(f"Unsupported provider: {provider}")
    
    config = {'provider': provider, 'model': model, 'class': provider_info['class'], 'module': provider_info['module']}
    
    # Add all LLM_ prefixed environment variables to the config
    for key, value in os.environ.items():
        if key.startswith('LLM_') and key not in ['LLM_PROVIDER', 'LLM_MODEL']:
            config[key[4:].lower()] = value
    
    return config

_llm_instance = None
_llm_config = None

def get_llm(**kwargs):
    """Dynamically retrieves the appropriate LLM based on the configuration."""
    global _llm_instance, _llm_config
    
    #Retrieve the current configuration
    config = get_llm_config()

    # Check if the instance already exists and the configuration hasn't changed
    if _llm_instance is not None and _llm_config == config:
        return _llm_instance
    
    try:
        module = importlib.import_module(config['module'])
        llm_class = getattr(module, config['class'])
        
        # Remove 'class' and 'module' from config before passing to the constructor
        constructor_args = {k: v for k, v in config.items() if k not in ['class', 'module', 'provider']}
        
        # Run provider-specific initialization
        init_func = PROVIDER_MAPPING[config['provider']]['init_func']
        constructor_args, _ = init_func(constructor_args, **kwargs)
        # print(constructor_args)

        _llm_instance = llm_class(**constructor_args)

        # Cache the config to detect changes
        _llm_config = config  
        
        return _llm_instance
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Error initializing provider {config['provider']}: {str(e)}")

# Define a global thread-safe lock
# csv_lock = threading.Lock()

def call_llm(messages, **kwargs):
    """Calls the configured LLM provider with the given parameters."""
    config = get_llm_config()
    llm = get_llm(**kwargs)

    # Get the provider-specific parameter handler
    _, kwargs = PROVIDER_MAPPING[config['provider']]['init_func']({}, **kwargs)

    # Check if this model doesn't support system messages
    supports_system_messages = kwargs.pop('supports_system_messages', True)
    
    message_types = {
        'system': SystemMessage if supports_system_messages is not False else HumanMessage,
        'user': HumanMessage,
        'assistant': AIMessage
    }

    langchain_messages = [
        message_types.get(msg['role'], HumanMessage)(content=msg['content'])
        for msg in messages
    ]

    # Log timestamp for when the request is sent
    # request_timestamp = time.time()

    # Call the LLM with the processed parameters
    response = llm(langchain_messages, **kwargs)

    # Log timestamp for when the response is received
    # response_timestamp = time.time()

    # Ensure thread-safe writing to CSV
    # csv_file = os.getenv("CSV_PATH", "responses_log.csv")
    # with csv_lock:
    #     # Open the CSV file in append mode
    #     file_exists = os.path.isfile(csv_file)
    #     with open(csv_file, mode='a', newline='') as f:
    #         csv_writer = csv.writer(f)
            
    #         # Write the header only if the file does not already exist
    #         if not file_exists:
    #             csv_writer.writerow(["request_made_time", "response_received_time", "elapsed_time", "response_content"])
            
    #         # Write the timestamps and response content
    #         csv_writer.writerow([
    #             time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(request_timestamp)),
    #             time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(response_timestamp)),
    #             f"{response_timestamp - request_timestamp:.2f} seconds",
    #             response.content.strip()
    #         ])

    # Write the response to a file instead of printing it
    with open(os.getenv("RESPONSE_PATH", "response.txt"), "a") as f:
        f.write(f"{response}\n")

    return response.content.strip()

def retry_llm_call(messages, process_response, failure_handler, max_attempts=5, call_llm_kwargs=None, process_response_kwargs=None, failure_handler_kwargs=None):
    """
    A generic wrapper for LLM calls that implements retry logic with custom processing and failure handling.
    
    Args:
    messages (list): The messages or prompt to send to the LLM.
    process_response (callable): A function that takes the LLM output and attempts to process it into the desired result.
    failure_handler (callable): A function to call if max_attempts is reached without successful processing.
    max_attempts (int): Maximum number of attempts before calling the failure_handler function.
    call_llm_kwargs (dict): Keyword arguments to pass to the LLM call function.
    process_response_kwargs (dict): Keyword arguments to pass to the process_response function.
    failure_handler_kwargs (dict): Keyword arguments to pass to the failure_handler function.
    
    Returns:
    The result of process_response if successful, or the result of failure_handler if not.
    """
    call_llm_kwargs = call_llm_kwargs or {}
    process_response_kwargs = process_response_kwargs or {}
    failure_handler_kwargs = failure_handler_kwargs or {}

    for attempt in range(1, max_attempts + 1):
        # Adjust temperature if it's in call_llm_kwargs
        if 'temperature' in call_llm_kwargs:
            call_llm_kwargs['temperature'] = 0 if attempt <= 2 else (attempt - 2) * 0.025
        
        # Call the LLM
        response = call_llm(messages=messages, **call_llm_kwargs)
        
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

def enforce_semantic_list(lst):
    error_message = "input list appears to contain any of: NaN, numeric, or numeric cast as string. Please ensure you are passing semantic labels (i.e. gene symbols or cell types) and not integer labels for AI interpretation. Make sure adata.var.index and adata.obs.index are not integers or integers cast as strings."
    
    def get_context(lst, index):
        before = lst[index - 1] if index > 0 else None
        after = lst[index + 1] if index < len(lst) - 1 else None
        return before, after

    # Check if all items are strings
    for index, item in enumerate(lst):
        if not isinstance(item, str):
            before, after = get_context(lst, index)
            raise ValueError(f"{error_message} Item at index {index} is not a string: {item}. Context: Before: {before}, After: {after}")

    # Check if any item can be converted to float
    for index, item in enumerate(lst):
        try:
            float(item)
        except ValueError:
            pass
        else:
            before, after = get_context(lst, index)
            raise ValueError(f"{error_message} Item at index {index} can be cast to a number: {item}. Context: Before: {before}, After: {after}")

    return True
    

def extract_dictionary_from_ai_string(ai_string):
    """
    Cleans a generated string by removing everything before the first '{'
    and everything after the first '}'.

    Args:
    generated_string (str): The string generated by retry_call_llm which includes unwanted characters or code.

    Returns:
    str: A cleaned string that can be evaluated as a dictionary.
    """
    # Find the positions of the first '{' and the first '}'
    start = ai_string.find('{')
    end = ai_string.find('}')
    # Extract the first substring that starts with '{' and ends with '}'
    cleaned_string = ai_string[start:end+1]
    return cleaned_string

def extract_list_from_ai_string(ai_string):
    """
    Cleans a generated string by removing everything before the first '{'
    and everything after the first '}'.

    Args:
    generated_string (str): The string generated by retry_call_llm which includes unwanted characters or code.

    Returns:
    str: A cleaned string that can be evaluated as a dictionary.
    """
    # Find the positions of the first '{' and the first '}'
    start = ai_string.find('[')
    end = ai_string.find(']')
    # Extract the first substring that starts with '{' and ends with '}'
    cleaned_string = ai_string[start:end+1]
    return cleaned_string


def attempt_ai_integration(ai_func, fallback_func, *args, **kwargs):
    """
    Attempts to run the AI integration function with the provided arguments. If an exception occurs,
    runs the fallback function instead.

    Args:
        ai_func (callable): The AI integration function to run.
        fallback_func (callable): The fallback function to run if the AI function fails.
        args: Variable length argument list for the AI function.
        kwargs: Arbitrary keyword arguments for the AI function.

    Returns:
        The result of the AI function if it succeeds, otherwise the result of the fallback function.
    """
    try:
        return ai_func(*args, **kwargs)
    except Exception as e:
        print(f"AI integration failed: {e}")
        return fallback_func()
    

def generate_file_key(file_path):
    """
    Generates a concise, descriptive name for a file based on its file path using the configured AI model. Deprecated.

    Args:
        file_path (str): The path to the file for which to generate a name.

    Returns:
        str: A generated name for the file that can be used as a dictionary key.
    """
    # Prepare the messages for the LLM
    messages = [
        {"role": "system", "content": "You are a python dictionary key generator. Examples: /Users/john/data/single_cell_liver_data.h5ad -> liver, /Users/john/data/single_cell_heart_normalized.h5ad -> heart_normalized"},
        {"role": "user", "content": f"{file_path} -> "}
    ]

    # Call the LLM using the call_llm function
    generated_name = call_llm(
        messages=messages,
        max_tokens=20,
        temperature=0.1
    )

    return generated_name


def process_llm_category_mapping(original_categories, llm_dict):
    """
    Map original categories to simplified categories using LLM output.

    Normalizes strings, finds exact or close matches between original categories
    and LLM-provided keys, and creates a mapping. If no match is found, the 
    original category is used.

    Args:
    original_categories (list): List of original category strings.
    llm_dict (dict): Dictionary of LLM-provided category mappings.

    Returns:
    dict: Mapping of original categories to simplified categories.
    """
    
    # Normalize original categories and LLM keys
    normalized_llm_keys = [normalize_string(key) for key in llm_dict.keys()]
    
    # Create mapping
    final_mapping = {}
    for original in original_categories:
        normalized = normalize_string(original)
        if normalized in normalized_llm_keys:
            # Direct match found
            index = normalized_llm_keys.index(normalized)
            final_mapping[original] = llm_dict[list(llm_dict.keys())[index]]
        else:
            # Find closest match
            matches = get_close_matches(normalized, normalized_llm_keys, n=1, cutoff=0.6)
            if matches:
                index = normalized_llm_keys.index(matches[0])
                final_mapping[original] = llm_dict[list(llm_dict.keys())[index]]
            else:
                # No match found, use original category
                final_mapping[original] = original
    
    return final_mapping

#Label simplification functions
def map_cell_type_labels_to_simplified_set(labels, simplification_level='', batch_size=50):
    """
    Maps a list of labels to a smaller set of labels using the AI, processing in batches.
    Args:
    labels (list of str): The list of labels to be mapped.
    simplification_level (str): A qualitative description of how much you want the labels to be simplified. Or a direction about how to simplify the labels. Could be anything, like 'extremely', 'barely', 'compartment-level', 'remove-typos'
    batch_size (int): The number of labels to process in each batch.
    Returns:
    dict: A dictionary mapping the original labels to the smaller set of labels.
    """
    #todo, could allow passing custom examples
    #enforce that labels are semantic
    enforce_semantic_list(labels)

    # Prepare the initial prompt
    initial_labels_str = "    ".join(labels)
    
    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": f"You are a python dictionary mapping generator that takes a list of categories and provides a mapping to a {simplification_level} simplified set as a dictionary. Generate only a dictionary. Example: Fibroblast.    Fibroblasts.    CD8-positive T Cells.    CD4-positive T Cells. -> {{'Fibroblast.':'Fibroblast','Fibroblasts.':'Fibroblast','CD8-positive T Cells.':'T Cell','CD4-positive T Cells.':'T Cell'}}"},
        {"role": "user", "content": f"Here is the full list of labels to be simplified: {initial_labels_str}. Acknowledge that you've seen all labels. Do not provide the mapping yet."}
    ]

    # Get initial acknowledgment
    initial_response = retry_llm_call(
        messages=messages,
        process_response=lambda x: x,
        failure_handler=lambda: "Failed to process initial prompt",
        call_llm_kwargs={'max_tokens': 30, 'temperature': 0},
        max_attempts=1
    )
    messages.append({"role": "assistant", "content": initial_response})

    def process_batch(batch_labels):
        batch_str = "    ".join(batch_labels)
        messages.append({"role": "user", "content": f"Provide a mapping for this batch of labels. Generate only a dictionary: {batch_str} -> "})
        
        def process_response(response):
            cleaned_mapping = extract_dictionary_from_ai_string(response)
            return eval(cleaned_mapping)

        def failure_handler(labels):
            print(f"Simplification failed for labels: {labels}")
            return {label: label for label in labels}

        call_llm_kwargs = {
            'max_tokens': min(300 + 25*len(batch_labels), 4000),
            'temperature': 0
        }
        failure_handler_kwargs = {'labels': batch_labels}

        batch_mapping = retry_llm_call(
            messages=messages,
            process_response=process_response,
            failure_handler=failure_handler,
            call_llm_kwargs=call_llm_kwargs,
            failure_handler_kwargs=failure_handler_kwargs
        )
        messages.append({"role": "assistant", "content": str(batch_mapping)})
        return batch_mapping

    # Process all labels in batches
    full_mapping = {}
    for i in range(0, len(labels), batch_size):
        batch = labels[i:i+batch_size]
        batch_mapping = process_batch(batch)
        full_mapping.update(batch_mapping)

    # Final pass to ensure consistency
    final_mapping = process_llm_category_mapping(labels, full_mapping)
    
    return final_mapping


def map_gene_labels_to_simplified_set(labels, simplification_level='', batch_size=50):
    """
    Maps a list of genes to a smaller set of labels using AI, processing in batches.
    Args:
    labels (list of str): The list of labels to be mapped.
    simplification_level (str): A qualitative description of how much you want the labels to be simplified.
    batch_size (int): The number of labels to process in each batch.
    Returns:
    dict: A dictionary mapping the original labels to the smaller set of labels.
    """
    # Enforce that labels are semantic
    enforce_semantic_list(labels)

    # Prepare the initial prompt
    initial_labels_str = "    ".join(labels)
    
    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": f"You are a python dictionary mapping generator that takes a list of genes and provides a mapping to a {simplification_level} simplified set as a dictionary. Example: HSP90AA1    HSPA1A    HSPA1B    CLOCK    ARNTL    PER1    IL1A    IL6 -> {{'HSP90AA1':'Heat Shock Proteins','HSPA1A':'Heat Shock Proteins','HSPA1B':'Heat Shock Proteins','CLOCK':'Circadian Rhythm','ARNTL':'Circadian Rhythm','PER1':'Circadian Rhythm','IL1A':'Interleukins','IL6':'Interleukins'}}"},
        {"role": "user", "content": f"Here is the full list of gene labels to be simplified: {initial_labels_str}. Acknowledge that you've seen all labels. Do not provide the mapping yet."}
    ]

    # Get initial acknowledgment
    initial_response = retry_llm_call(
        messages=messages,
        process_response=lambda x: x,
        failure_handler=lambda: "Failed to process initial prompt",
        call_llm_kwargs={'max_tokens': 30, 'temperature': 0},
        max_attempts=1
    )
    messages.append({"role": "assistant", "content": initial_response})

    def process_batch(batch_labels):
        batch_str = "    ".join(batch_labels)
        messages.append({"role": "user", "content": f"Provide a mapping for this batch of gene labels. Generate only a dictionary: {batch_str} -> "})
        
        def process_response(response):
            cleaned_mapping = extract_dictionary_from_ai_string(response)
            return eval(cleaned_mapping)

        def failure_handler(labels):
            print(f"Simplification failed for gene labels: {labels}")
            return {label: label for label in labels}

        call_llm_kwargs = {
            'max_tokens': min(300 + 25*len(batch_labels), 4000),
            'temperature': 0
        }
        failure_handler_kwargs = {'labels': batch_labels}

        batch_mapping = retry_llm_call(
            messages=messages,
            process_response=process_response,
            failure_handler=failure_handler,
            call_llm_kwargs=call_llm_kwargs,
            failure_handler_kwargs=failure_handler_kwargs
        )
        messages.append({"role": "assistant", "content": str(batch_mapping)})
        return batch_mapping

    # Process all labels in batches
    full_mapping = {}
    for i in range(0, len(labels), batch_size):
        batch = labels[i:i+batch_size]
        batch_mapping = process_batch(batch)
        full_mapping.update(batch_mapping)

    # Final pass to ensure consistency
    final_mapping = process_llm_category_mapping(labels, full_mapping)
    
    return final_mapping


#Biological inference functions
def ai_biological_process(gene_list):
    """
    Describes the most prominent biological process represented by a list of genes using the AI.

    Args:
        gene_list (list of str): The list of genes to be described.

    Returns:
        dict: A dictionary containing the description of the biological process.
    """
    #enforce that labels are semantic
    enforce_semantic_list(gene_list)

    # Prepare the prompt
    if len(gene_list) == 1:
        base_prompt = f"In a few words and without restating any part of the question, describe the single most prominent biological process represented by the gene: {gene_list[0]}"
    else:
        genes_str = "    ".join(gene_list[:-1])
        base_prompt = f"In a few words and without restating any part of the question, describe the single most prominent biological process represented by the genes: {genes_str}, and {gene_list[-1]}"

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": "You are a terse molecular biologist."},
        {"role": "user", "content": base_prompt}
    ]

    # Call the LLM using the call_llm function
    annotation = call_llm(
        messages=messages,
        max_tokens=200,
        temperature=0
    )

    return annotation

def ai_gene_list(cell_type, species, list_length=None):
    """
    Returns a list of specific marker genes for the input cell_type.

    Args:
        cell_type (str): The cell type to get marker genes for.
        species (str): The species to consider.
        list_length (str, optional): if not None, provides a {list_length} list of genes (i.e. very long, short, much shorter, etc.)

    Returns:
        list: A list of marker genes.
    """
    # Enforce that cell_type is a semantic string
    enforce_semantic_list([cell_type])
    
    # Initialize the conversation with the system prompt
    messages = [
        {"role": "system", "content": "You are a terse molecular biologist."}
    ]
    
    # Step 1: Ask about canonical marker genes
    step1_prompt = (
        f"Discuss canonical {cell_type} marker genes in {species}. "
        f"Then, narrow your discussion to highly specific marker genes of this cell type."
    )
    messages.append({"role": "user", "content": step1_prompt})
    
    # Get the response from the assistant
    response1 = call_llm(
        messages=messages,
        max_tokens=500,
        temperature=0
    )
    messages.append({"role": "assistant", "content": response1})
    
    # Step 2: If extensive_list is True, ask for a longer list
    if list_length:
        step2_prompt = f"Provide a {list_length} list of genes."
        messages.append({"role": "user", "content": step2_prompt})
        
        # Get the response from the assistant
        response2 = call_llm(
            messages=messages,
            max_tokens=750,
            temperature=0
        )
        messages.append({"role": "assistant", "content": response2})
    
    # Step 3: Ask for the genes as a Python list
    step3_prompt = "Provide these as a python list as they would be present in scRNA-seq data."
    messages.append({"role": "user", "content": step3_prompt})

    def process_response(response):
        gene_list = extract_list_from_ai_string(response)
        return eval(gene_list)

    def failure_handler(cell_type):
        print(f"Failed to generate list for: {cell_type}")
        return []

    call_llm_kwargs = {
        'max_tokens': 1000,
        'temperature': 0
    }

    failure_handler_kwargs = {'cell_type': cell_type}

    gene_list = retry_llm_call(
        messages=messages,
        process_response=process_response,
        failure_handler=failure_handler,
        call_llm_kwargs=call_llm_kwargs,
        failure_handler_kwargs=failure_handler_kwargs
    )
    
    return gene_list


def ai_cell_type(gene_list, tissue=None):
    """
    Returns the cell type based on a list of marker genes as determined by AI.
    Args:
    gene_list (list of str): The list of genes to be described.
    tissue (str, optional): The tissue of origin to provide context for the AI.
    Returns:
    str: The cell type label generated by AI
    """
    #enforce that labels are semantic
    enforce_semantic_list(gene_list)

    # Prepare the prompt
    if len(gene_list) == 1:
        base_prompt = f"In a few words and without restating any part of the question, describe the single most likely cell type represented by the marker gene: {gene_list[0]}"
    else:
        genes_str = "    ".join(gene_list)
        base_prompt = f"In a few words and without restating any part of the question, describe the single most likely cell type represented by the marker genes: {genes_str}"

    # Add tissue information if provided
    if tissue:
        base_prompt += f" Consider that these cells are from {tissue} tissue."

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": "You are a terse molecular biologist."},
        {"role": "user", "content": base_prompt}
    ]

    # Call the LLM using the call_llm function
    annotation = call_llm(
        messages=messages,
        max_tokens=100,
        temperature=0
    )

    return annotation


def ai_cell_types_by_comparison(gene_lists, cell_types=None, tissues=None, subtype=False):
    """
    Returns cell type labels for multiple lists of marker genes as determined by AI.
    Args:
    gene_lists (list of lists): A list containing multiple lists of genes to be described.
    cell_type (str, optional): The cell type to provide context for the AI.
    tissue (str, optional): The tissue of origin to provide context for the AI.
    Returns:
    list of str: The cell type labels generated by AI for each gene list.
    """
    # Enforce semantic_list for each gene list
    for gene_list in gene_lists:
        enforce_semantic_list(gene_list)

    # Prepare the system prompt
    system_prompt = (
        "You are a terse molecular biologist. You respond in a few words and without restating any part of the question. "
        f"Compare and contrast gene sets to identify the most likely cell {'sub' if subtype else ''}type based on marker genes."
    )

    # Prepare the initial user prompt for contrasting all gene lists
    # initial_prompt = f"Tissue: {tissues}, " if tissue else ""
    # initial_prompt += f"Cell Type: {cell_type}, " if cell_type else ""
    initial_prompt = "Briefly compare and contrast the following gene sets:\n"
    for i, gene_list in enumerate(gene_lists, 1):
        tissue_str = " " + ', '.join(tissues[i]) if tissues and tissues[i] else ""
        cell_type_str = " " + ', '.join(cell_types[i]) if cell_types and cell_types[i] else ""

        initial_prompt += f"{i}){tissue_str}{cell_type_str} {('    '.join(gene_list))}\n"

    # Initialize the conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_prompt}
    ]

    # Get the initial contrast response
    contrast_response = retry_llm_call(
        messages=messages,
        process_response=lambda x: x,
        failure_handler=lambda: "Failed to contrast gene sets",
        call_llm_kwargs={'max_tokens': 300, 'temperature': 0},
        max_attempts=1
    )

    # Append the contrast response to the conversation
    messages.append({"role": "assistant", "content": contrast_response})

    messages.append({"role": "user", "content": "Provide only the new label. "})

    # Process each gene list
    cell_subtype_labels = []
    for i, gene_list in enumerate(gene_lists, 1):
        tissue_str = " " + ', '.join(tissues[i]) if tissues and tissues[i] else ""
        cell_type_str = " " + ', '.join(cell_types[i]) if cell_types and cell_types[i] else ""

        gene_set_prompt = f"What is the cell{tissue_str}{cell_type_str} {'sub' if subtype else ''}type label for the gene set: {('    '.join(gene_list))}?"
        messages.append({"role": "user", "content": gene_set_prompt})

        # Get the subtype label
        subtype_label = retry_llm_call(
            messages=messages,
            process_response=lambda x: x.strip(),
            failure_handler=lambda: cell_type_str if cell_types and cell_types[i] else "Unknown",
            call_llm_kwargs={'max_tokens': 50, 'temperature': 0},
            max_attempts=1
        )

        cell_subtype_labels.append(subtype_label)
        messages.append({"role": "assistant", "content": subtype_label})

    # print(f"{messages}")
    return cell_subtype_labels


def ai_compare_cell_types_binary(label1, label2):
    """
    Compares two cell type labels using AI.

    Args:
        item1 (str): The first item to compare.
        item2 (str): The second item to compare.

    Returns:
        str: The comparison result generated by AI.
    """

    # Check if normalized labels are the same--Shortcut for exact match
    if normalize_string(label1) == normalize_string(label2):
        return 'yes'
    
    # Prepare the prompt
    gpt_prompt = f"1) {label1} 2) {label2} -> "

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": "You are a terse molecular biologist who determines if two labels refer to the same cell type. Respond only with 'yes' or 'no'. Examples: 1) CD8-positive T cell 2) T cell -> yes; 1) epithelial cell 2) intrahepatic cholangiocyte -> yes; 1) Macrophage 2) Endothelial Cell -> no; 1) B cell 2) Plasma Cell -> yes"},
        {"role": "user", "content": gpt_prompt}
    ]

    # Call the LLM using the call_llm function
    comparison_result = call_llm(
        messages=messages,
        max_tokens=20,
        temperature=0
    )

    return comparison_result


def ai_compare_cell_types_categorical(label1, label2):
    """
    Compares two cell type labels using AI.

    Args:
        item1 (str): The first item to compare.
        item2 (str): The second item to compare.

    Returns:
        str: The comparison result generated by AI.
    """
    # Check if normalized labels are the same--Shortcut for exact match
    if normalize_string(label1) == normalize_string(label2):
        return 'perfect match'
    
    # Prepare the prompt
    gpt_prompt = f"1) {label1} 2) {label2} -> "

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": "You are a terse molecular biologist who assesses the degree to which two labels refer to the same cell type. Respond only with 'perfect match', 'partial match', or 'no match'. Examples: 1) CD8-positive T cell 2) T cell -> partial match; 1) epithelial cell 2) intrahepatic cholangiocyte -> partial match; 1) Macrophage 2) Endothelial Cell -> no match; 1) macrophage 2) Macrophage. -> perfect match; 1) B cell 2) Plasma Cell -> partial match"},
        {"role": "user", "content": gpt_prompt}
    ]

    # Call the LLM using the call_llm function
    comparison_result = call_llm(
        messages=messages,
        max_tokens=30,
        temperature=0
    )

    return comparison_result


#image understanding functions
def encode_plot_for_openai(plot):
    buffer = BytesIO()
    plot()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return encoded_image


def ai_resolution_interpretation(plot):
    """
    Determines the clustering resolution adjustment based on an image of a plot using AI.

    Args:
        plot (function): A function that generates a matplotlib plot.

    Returns:
        str: The resolution adjustment suggestion generated by AI. One of "decreased", "increased", or "unchanged".
    """
    #convert plot to base64
    encoded_image = encode_plot_for_openai(plot)

    # Prepare the messages for the Chat Completions API

    # Example images encoded as base64 (replace with actual images as needed)
    oversplit_example = "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"
    well_split_example = "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"
    undersplit_example = "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"

    #Prepare messages
    messages = [
        {"role": "system", "content": "You are a nit-picking data scientist who decides if the clustering resolution should be 1) decreased 2) increased 3) unchanged. You only respond with one of these three words."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{oversplit_example}"}}
        ]},
        {"role": "user", "content": "decreased"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{undersplit_example}"}}
        ]},
        {"role": "user", "content": "increased"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{well_split_example}"}}
        ]},
        {"role": "user", "content": "unchanged"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        ]}
    ]

            # {"role": "system", "content": "A true cluster is an island of points. A subsplit cluster is when a true cluster has multiple labels assigned to it; this calls for resolution to be decreased. An undersplit cluster is when multiple true clusters have been assigned to the same label; this calls for resolution to be increased. Based on an image of a plot, decide if the clustering resolution should be 1) decreased 2) increased 3) unchanged. You only respond with one of these three words."},

    # Call the LLM using the call_llm function
    annotation = call_llm(
        messages=messages,
        max_tokens=30,
        temperature=0
    )

    return annotation


def determine_sign_of_resolution_change(annotation):
    """
    Returns -1 if "decreased", 1 if "increased", and 0 if "unchanged" is in the annotation string.

    Args: annotation (str): Annotation indicating resolution change.

    Returns: int: -1 for "decreased", 1 for "increased", 0 for "unchanged" or none.
    """
    if "decreased" in annotation:
        return -1
    elif "increased" in annotation:
        return 1
    elif "unchanged" in annotation:
        return 0
    else:
        return 0


#cell type confirmation functions

def filter_gene_list(adata, gene_list):
    """
    Filter and update a list of gene names based on their presence in the index of an AnnData object.

    Parameters:
        adata : AnnData
            AnnData object containing gene information in `adata.var.index`.
        gene_list : list of str
            List of gene names to be filtered and updated with possible unique suffixes.

    Returns:
        list of str
            Updated list of genes found in `adata.var.index`, including suffix variations.
    """
    enforce_semantic_list(adata.var.index)
    updated_gene_list = []
    import re
    for gene in gene_list:
        # Create a regex pattern to match the gene name and its possible unique suffixes, case-insensitive
        pattern = re.compile(r'^' + re.escape(gene) + r'(-\d+)?$', re.IGNORECASE)
        
        # Find all matching genes in adata.var.index
        matching_genes = [g for g in adata.var.index if pattern.match(g)]
        
        if matching_genes:
            updated_gene_list.extend(matching_genes)
        # else:
            # print(f"Gene '{gene}' not found in adata.var.index after making unique.")
            
    # Remove any duplicates in the updated marker list
    updated_gene_list = list(set(updated_gene_list))
    return updated_gene_list


def cell_type_marker_gene_score(adata, cell_type_col=None, cell_types=None, species='Human', list_length=None, score_name='_score', adt_key=None, **kwargs):
    """
    Compute marker gene scores for specified cell types. Must provide either a list of cell types, or a column that contains cell_type labels.
    
    Parameters:
        adata (AnnData): Annotated data matrix.
        cell_type_col (str, optional): Column name in adata.obs containing cell type annotations.
        cell_types (list of str, optional): List of cell types for which to compute the marker gene scores.
        species (str, optional): Species for gene list generation. Defaults to 'Human'.
        list_length (str, optional): Qualitative length of the marker gene list (i.e. "longer" if you are having trouble getting valid genes present in your dataset.)
        score_name (str, optional): Suffix for the computed score names. Defaults to '_score'.
        **kwargs: Optional keyword args passed to sc.tl.score_genes().
    
    Modifies:
        adata.var: Adds boolean columns indicating genes used in the scores.
        adata.obs: Adds new columns with the computed scores for each observation.
    
    """

    score_name_suffix = score_name

    # Check for conflicting parameters
    if cell_types is not None and cell_type_col is not None:
        raise ValueError("Provide either 'cell_type_col' or 'cell_types', not both.")
    
    if cell_types is None:
        if cell_type_col is not None:
            cell_types = adata.obs[cell_type_col].unique().tolist()
        else:
            raise ValueError("Either 'cell_type_col' or 'cell_types' must be provided.")
    else:
        # Ensure cell_types is a list
        if isinstance(cell_types, str):
            cell_types = [cell_types]
    
    for cell_type in cell_types:
        cell_type = str(cell_type)  # Ensure cell_type is a string
        # Set the score_name per cell type
        score_name = f"{cell_type}{score_name_suffix}"
        
        # Generate gene list using ai_gene_list function
        gene_list = ai_gene_list(cell_type, species, list_length=list_length)
        
        # Filter the gene list based on genes present in adata
        gene_list = filter_gene_list(adata, gene_list)
        
        # Mark genes included in this score in adata.var
        adata.var[score_name] = adata.var.index.isin(gene_list)

        #calculate score if any valid genes, otherwise print warning and assign score value as NaN.
        if gene_list:
            # Compute the gene score and store it in adata.obs[score_name]
            sc.tl.score_genes(adata, gene_list=gene_list, score_name=score_name, **kwargs)
        else:
            # Assign NaN to adata.obs[score_name] for all observations
            adata.obs[score_name] = np.nan
            print(f"No valid genes for {cell_type} in {adt_key if adt_key else ''}. Assigning score value as NaN")


def module_score_barplot(adata, group_cols, score_cols, adt_key=None, figsize=(10,8)):
    """
    Create a bar plot of mean module scores grouped by specified columns.

    Parameters:
    adata : AnnData
        The AnnData object containing the data.
    group_cols : str or list of str
        The column(s) in adata.obs to group by.
    score_cols : str or list of str
        The column(s) in adata.obs that contain the module scores.

    Returns:
    fig, ax : matplotlib Figure and Axes
        The figure and axes objects of the plot.
    """
    #print adt_key if provided
    if adt_key:
        print(adt_key)

    # Ensure group_cols and score_cols are lists
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(score_cols, str):
        score_cols = [score_cols]

    # Select module score columns and the group columns
    module_scores = adata.obs[score_cols + group_cols]

    # Group by the group_cols and compute mean module scores
    mean_scores = module_scores.groupby(group_cols, observed=False).mean()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the mean module scores as a grouped bar plot
    mean_scores.plot(kind='bar', ax=ax)

    # Set labels, title, and legend location
    ax.set_ylabel('Mean Module Score')
    ax.legend(title=None, loc=6, bbox_to_anchor=(1,0.5))

    plt.xticks(rotation=90)
    plt.tight_layout()
    

    return fig, ax


def module_score_umap(adata, score_cols, adt_key=None, **kwargs):
    """
    Generates UMAP plots for specified module scores in a single figure.

    Parameters:
        adata (AnnData): Annotated data matrix containing UMAP coordinates and module scores.
        score_cols (list of str): List of column names in `adata` containing module scores to plot.
        **kwargs: Additional keyword arguments passed to `sc.pl.umap`, including 'vmax' for color scaling 
                  (default is 'p99').

    Returns:
        matplotlib.figure.Figure: Figure containing the UMAP plots.
    """
    # Print adt_key if provided
    if adt_key:
        print(adt_key)

    # Extract vmax from kwargs or default to 'p99'
    vmax = kwargs.pop('vmax', 'p99')
    
    n_plots = len(score_cols)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    # Ensure axes is a flat array of axes
    axes = np.atleast_1d(axes).ravel()
    
    for i, (score_col, ax) in enumerate(zip(score_cols, axes)):
        # Process title
        title = ' '.join(word.capitalize() for word in score_col.replace('_', ' ').split())
        # Plot the UMAP
        sc.pl.umap(adata, color=score_col, title=title, vmax=vmax, ax=ax, show=False, **kwargs)
        ax.set_title(title)
            
    # Turn off any unused axes
    for ax in axes[n_plots:]:
        ax.axis('off')
            
    plt.tight_layout()
    return fig