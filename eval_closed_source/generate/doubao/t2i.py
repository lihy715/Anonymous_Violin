import requests
import json
import os
from urllib.parse import urlparse

def download_image(json_response, output_path):
    """
    Downloads an image from the URL provided in the API response.

    Args:
        json_response (str): The raw JSON response string from the image generation API.
        output_path (str): The local file path where the image will be saved.
    """
    try:
        data = json.loads(json_response)
        # Check if the expected data structure exists
        if 'data' not in data or not data['data']:
            print(f"Error: Invalid API response structure. Message: {data.get('error')}")
            return

        image_url = data['data'][0]['url']
        
        # Download the image using a stream to handle large files efficiently
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
    except Exception as e:
        print(f"Failed to download image: {e}")

def call_image_generation_api(prompt, save_path, api_key=None):
    """
    Calls the remote LLM-based image generation service and saves the output.

    Args:
        prompt (str): The textual description for image generation.
        save_path (str): The target filesystem path for the output image.
        api_key (str): API authorization key. Defaults to environment variable.
    """
    # Use environment variables or arguments
    api_key = api_key or os.getenv("GENERATIVE_API_KEY", "YOUR_PLACEHOLDER_KEY")
    api_url = "https://api.bltcy.ai/v1/images/generations"

    payload = {
        "model": "doubao-seedream-5-0-260128",
        "prompt": prompt,
        "n": 1,
        "response_format": "url",
        "size": "2K",
        "aspect_ratio": "1:1",
        "watermark": False
    }
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        download_image(response.text, save_path)
        print(f"Success: Image saved to {save_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
