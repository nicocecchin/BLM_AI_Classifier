import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, AuthenticationError, Timeout, APIConnectionError, OpenAIError
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from azure.core.pipeline.policies import RetryPolicy
from azure.core.exceptions import HttpResponseError, ServiceRequestError

from mistralai import Mistral, UserMessage as mUserMessage, SystemMessage as mSystemMessage

# List of models to try sequentially
models = [
    "gpt-4o", 
    "gpt-4o-mini", 
    "gpt-4.1",
    "gpt-4.1-mini", 
    "Meta-Llama-3.1-405B-Instruct", 
    "Mistral-Large-2411",
    "DeepSeek-V3-0324",
    "grok-3",
]

# Load environment variables once
def _load_config():
    load_dotenv()
    return {
        "token": os.environ.get('GITHUB_TOKEN_CECCHIN'),
        "endpoint": "https://models.inference.ai.azure.com",
        "temperature": 1.0,
        "max_tokens": 4000,
        "top_p": 1.0
    }

PARAMS = _load_config()


def get_llm_response(user_message: str, system_message: str = "You are a helpful assistant.") -> str:
    for model in models:
        timeout_count = 0
        print(f"Trying model: {model}...")
        while True:
            try:
                if model in ("gpt-4o", "gpt-4.1", "gpt-4o-mini", "gpt-4.1-mini"):  # OpenAI-compatible
                    return _call_openai(model, system_message, user_message)
                else:
                    return _call_azure(model, system_message, user_message)

            except (RateLimitError, OpenAIError) as e:
                print(f"OpenAI error ({e}), skipping to next model.")
                break

            except HttpResponseError as e:
                if e.status_code == 429:
                    print("Azure rate limit, skipping model.")
                else:
                    print(f"Azure HTTP error ({e.status_code}), skipping model.")
                break

            except ServiceRequestError as e:
                print(f"Azure service request error ({e}), skipping model.")
                break

            except Exception as e:
                print(f"Unexpected error ({e}), skipping model.")
                break

    return "All models failed or rate-limited."


def _call_openai(model: str, system_msg: str, user_msg: str) -> str:
    """Invoke an OpenAI or Azure OpenAI model via the openai-python client."""
    client = OpenAI(base_url=PARAMS["endpoint"], api_key=PARAMS["token"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=PARAMS["temperature"],
        top_p=PARAMS["top_p"],
        max_tokens=PARAMS["max_tokens"]
    )
    return response.choices[0].message.content


def _call_azure(model: str, system_msg: str, user_msg: str) -> str:
    """Invoke an Azure-hosted model via ChatCompletionsClient."""
    retry = RetryPolicy(retry_total=3, timeout=10)
    client = ChatCompletionsClient(
        endpoint=PARAMS["endpoint"],
        credential=AzureKeyCredential(PARAMS["token"]),
        retry_policy=retry
    )
    response = client.complete(
        model=model,
        messages=[SystemMessage(content=system_msg), UserMessage(content=user_msg)],
        temperature=PARAMS["temperature"],
        top_p=PARAMS["top_p"],
        max_tokens=PARAMS["max_tokens"]
    )
    return response.choices[0].message.content
