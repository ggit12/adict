#AI integration for cell typing, interpretation of gene lists, and other labelling tasks

import os
import importlib
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import boto3


#LLM configuration
def bedrock_init(constructor_args, **kwargs):
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


def azureml_init(constructor_args, **kwargs):
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

    #Clear the existing LLM instance
    global _llm_instance
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





#delete this later
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