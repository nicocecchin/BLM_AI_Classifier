import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, AuthenticationError, Timeout, APIConnectionError, OpenAIError
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from mistralai import Mistral, UserMessage as mUserMessage, SystemMessage as mSystemMessage

from azure.core.pipeline.policies import RetryPolicy
from azure.core.exceptions import HttpResponseError, ServiceRequestError

models = ["gpt-4o", "Meta-Llama-3.1-405B-Instruct", "Phi-3-medium-4k-instruct", "Mistral-large-2407"]

# Function to get explanation from different LLM models
def get_llm_explanation(model, message):
    # Load environment variables from .env file (e.g., for API keys)
    load_dotenv()

    # Set up the parameters to be used across the API calls
    params = {
        "token": os.environ.get('GITHUB_TOKEN'),  # GitHub token for authentication
        "endpoint": "https://models.inference.ai.azure.com",  # Endpoint for the Azure AI service
        "temperature": 1.0,  # Controls randomness in the output
        "max_tokens": 4000,  # Max token limit for the response
        "top_p": 1.0,  # Controls diversity via nucleus sampling
    }

    # Depending on the selected model, call the appropriate function to get the response
    # return "no text"
    if model == "gpt-4o":
        response = get_GPT4o_response(params, message)
    elif model == "Meta-Llama-3.1-405B-Instruct":
        response = get_LLAMA3_1_response(params, message)
    elif model == "Phi-3-medium-4k-instruct":
        response = get_Phi3_medium_response(params, message)
    elif model == "Mistral-large-2407":
        response = get_Mistral_large_response(params, message)
    else:
        response = "Model not found"  # Return a default message if the model is not recognized

    return response

# Function to get a response from GPT-4o (OpenAI API)
def get_GPT4o_response(params, message):
    # Initialize the OpenAI client
    client = OpenAI(
        base_url=params["endpoint"],  # Set the base URL for the API
        api_key=params["token"],  # Provide the API key
    )

    try:
        # Send the chat completion request to the API
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ],
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
            model="gpt-4o",  # Specify the model to use
        )
        return response.choices[0].message.content  # Return the content of the first choice

    except RateLimitError:
        return "Rate limit exceeded. Check your usage or wait for the limit to reset."
    except AuthenticationError:
        return "Authentication error: Check your API key."
    except Timeout:
        return "Request timed out: The API is not responding."
    except APIConnectionError:
        return "Connection error: Unable to reach the API."
    except OpenAIError as e:
        return f"An OpenAI error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Function to get a response from Meta's Llama-3.1 model (Azure OpenAI API)
def get_LLAMA3_1_response(params, message):
    # Create a retry policy for the request
    retry_policy = RetryPolicy(retry_total=3, timeout=10)

    # Initialize the Azure OpenAI client
    client = ChatCompletionsClient(
        endpoint=params["endpoint"], 
        credential=AzureKeyCredential(params["token"]),
        retry_policy=retry_policy,  # Set the retry policy
    )

    try:
        # Send the chat completion request to the Azure API
        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=message)
            ],
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
            model="Meta-Llama-3.1-405B-Instruct"
        )
        return response.choices[0].message.content  # Return the content of the first choice

    except HttpResponseError as e:
        # Handle different HTTP errors
        if e.status_code == 429:  # Too many requests
            return "Rate limit exceeded. Check your Azure OpenAI quota."
        elif e.status_code == 403:  # Authentication issues
            return "Access forbidden. Check your API key or permissions."
        else:
            return f"HTTP Error: {e.message} (Status Code: {e.status_code})"
    except ServiceRequestError as e:
        return f"Service Request Error: {e.message}"
    except Exception as e:
        return f"Unexpected Error: {e}"

# Function to get a response from Phi-3 medium (Azure OpenAI API)
def get_Phi3_medium_response(params, message):
    # Same retry policy as LLAMA3.1
    retry_policy = RetryPolicy(retry_total=3, timeout=10)

    # Initialize the Azure OpenAI client
    client = ChatCompletionsClient(
        endpoint=params["endpoint"],
        credential=AzureKeyCredential(params["token"]),
        retry_policy=retry_policy,  # Set the retry policy
    )

    try:
        # Send the chat completion request to the Azure API
        response = client.complete(
            messages=[UserMessage(content=message)],
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=3000,  # Limitation for Phi-3 model
            model="Phi-3-medium-4k-instruct"
        )
        return response.choices[0].message.content  # Return the content of the first choice

    except HttpResponseError as e:
        if e.status_code == 429:  # Too many requests
            return "Rate limit exceeded. Check your Azure OpenAI quota."
        elif e.status_code == 403:  # Authentication issues
            return "Access forbidden. Check your API key or permissions."
        else:
            return f"HTTP Error: {e.message} (Status Code: {e.status_code})"
    except ServiceRequestError as e:
        return f"Service Request Error: {e.message}"
    except Exception as e:
        return f"Unexpected Error: {e}"

# Function to get a response from Mistral's large model (Mistral AI)
def get_Mistral_large_response(params, message):
    # Initialize the Mistral client
    client = Mistral(api_key=params["token"], server_url=params["endpoint"])

    try:
        # Send the chat completion request to the Mistral API
        response = client.chat.complete(
            model="Mistral-large-2407",  # Specify the model to use
            messages=[
                mSystemMessage(content="You are a helpful assistant."),
                mUserMessage(content=message)
            ],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            top_p=params["top_p"],
        )
    except Exception as e:
        return f"Unexpected Error: {e}"

    return response.choices[0].message.content  # Return the content of the first choice


# if __name__ == '__main__':
#     message = "what is the capital of france?"
#     model = "gpt-4o"
    
#     response = get_llm_explanation(model, message)

#     print(response)
