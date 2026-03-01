import logging

logger = logging.getLogger(__name__)

import requests


def call_LLM(url: str, key: str, model: str, prompt, temp: float=0.0, max_tokens: int=1000, timeout=30.0) -> dict | None:
    """Call the openAI/Ollama API to generate a response based on the provided model and prompt.
    
    Args:
        url (str): The API endpoint URL.
        key (str): The API key for authentication.
        model (str): The model to use for generation.
        prompt (str): The input prompt for the model.
        temp (float, optional): The temperature for generation. Defaults to 0.0.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1000.
        timeout (float, optional): The timeout for the API request in seconds. Defaults to 30.0.
    Returns:
        dict | None: The JSON response from the API if successful, otherwise None.
    Raises:
        KeyboardInterrupt: If the operation is interrupted by the user
    """
    # Detect API type based on URL
    is_ollama = 'generate' in url.lower()
    
    if is_ollama:
        # Ollama API format
        data = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': temp,
                'num_predict': max_tokens,
            },
        }
    else:
        # OpenAI-compatible API format
        data = {
            'model': model,
            'messages': [
                { 'role': 'user', 'content': prompt }
            ],
            'temperature': temp,
            'max_tokens': max_tokens,
            'stream': False,
        }

    try:
        response = requests.post(
            url, 
            json=data, 
            headers = { 'content-type': 'application/json', 'Authorization': f'Bearer {key}' },
            timeout=timeout
        )
        if response.status_code != 200:
            logger.error(f'Error call LLM: {response.text}')
            return None
        return response.json()

    except requests.exceptions.Timeout:
        logger.error(f'Timeout error while executing prompt')
    except requests.exceptions.ConnectionError:
        logger.error(f'Connection error while executing prompt')
    except requests.exceptions.RequestException as e:
        logger.error(f'Request exception in function "call_LLM": {e}')
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f'Exception in function "call_LLM": {e}')

    return None