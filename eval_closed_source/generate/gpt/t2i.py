import requests
import json
import os
from urllib.parse import urlparse
import base64
import re

def save_b64_images(json_response, output_path):
    """
    Parses a JSON response, sanitizes Base64 data, and saves it to local files.
    Includes robust handling for truncated data and incorrect padding.

    Args:
        json_response (str): The raw JSON response string.
        output_path (str): The base local file path for saving images.
    """
    try:
        # Step 1: Ensure we have a valid JSON object
        data = json.loads(json_response)
        image_list = data.get("data", [])

        if not image_list:
            print(f"Error: No image data. API Message: {data.get('error')}")
            return

        for index, item in enumerate(image_list):
            b64_data = item.get("b64_json")
            if not b64_data:
                continue

            # Step 2: Sanitization - Remove all whitespace and potential data-URL prefixes
            # Sometimes APIs return 'data:image/png;base64,iVBOR...'
            if "," in b64_data:
                b64_data = b64_data.split(",")[-1]
            
            # Remove any characters that are not valid Base64 (A-Z, a-z, 0-9, +, /, =)
            b64_data = re.sub(r'[^A-Za-z0-9+/=]', '', b64_data)

            # Step 3: Critical Length Validation
            # Base64 string length % 4 cannot be 1. If it is, the data is physically corrupted.
            length = len(b64_data)
            if length % 4 == 1:
                print(f"Error: Base64 data at index {index} is truncated or corrupted (Length: {length}).")
                continue

            # Step 4: Fix Padding
            missing_padding = length % 4
            if missing_padding:
                b64_data += '=' * (4 - missing_padding)

            # Step 5: Decode and Verify Image Header
            try:
                img_bytes = base64.b64decode(b64_data)
                
                # Simple check: PNG header starts with \x89PNG
                if not img_bytes.startswith(b'\x89PNG') and not img_bytes.startswith(b'\xff\xd8'):
                    print(f"Warning: Decoded data at index {index} does not look like a valid PNG/JPG header.")

                # Determine file path
                if len(image_list) == 1:
                    current_path = output_path
                else:
                    base, ext = os.path.splitext(output_path)
                    current_path = f"{base}_{index}{ext if ext else '.png'}"

                with open(current_path, 'wb') as f:
                    f.write(img_bytes)
                print(f"Successfully saved: {current_path} ({len(img_bytes)} bytes)")

            except Exception as decode_err:
                print(f"Decode error at index {index}: {decode_err}")

    except json.JSONDecodeError:
        print("Fatal Error: The API response is not valid JSON. The download was likely interrupted.")
    except Exception as e:
        print(f"Unexpected error: {e}")


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

    payload = json.dumps({
        "model": "gpt-image-2",
        "prompt": prompt,
        "aspect_ratio": "1:1",
        "response_format":"b64_json",
    })
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", api_url, headers=headers, data=payload)
        response.raise_for_status()
        
        # Move success logic inside the helper or check return value
        save_b64_images(response.text, save_path)
        
    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
