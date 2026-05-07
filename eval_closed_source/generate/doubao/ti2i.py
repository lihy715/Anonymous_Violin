import requests
import json
import os
from urllib.parse import urlparse

def download_result(json_response, output_path):
    """
    Parses the API response and downloads the edited image.
    """
    try:
        data = json.loads(json_response)
        if 'data' not in data or not data['data']:
            print(f"Error: Invalid response. Details: {data.get('error')}")
            return

        image_url = data['data'][0]['url']
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Download failed: {e}")

def call_image_edit_api(prompt, base_img_path, mask_img_path, save_path, api_key=None):
    """
    Calls the Image Editing API to perform mask-based synthesis.
    
    Args:
        prompt (str): Instructions for the editing task.
        base_img_path (str): Path to the source image.
        mask_img_path (str): Path to the binary mask image.
        save_path (str): Local path to save the result.
        api_key (str): API key for authentication.
    """
    # Anonymized: Load key from environment and use relative base paths
    api_key = api_key or os.getenv("GENERATIVE_API_KEY", "PLACEHOLDER_KEY")
    url = "https://api.bltcy.ai/v1/images/edits"
    
    # Standardize data paths to be relative to the project root
    current_script_path = os.path.abspath(__file__) 
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

    default_data_root = os.path.join(root_dir, "benchmark", "data")
    data_root = os.getenv("VIOLIN_DATA_ROOT", default_data_root)

    full_img_path = os.path.join(data_root, base_img_path)
    full_mask_path = os.path.join(data_root, mask_img_path)

    payload = {
        "model": "doubao-seedream-5-0-260128",
        "prompt": prompt,
        "n": 1,
        "response_format": "url",
        "size": "2K",
        "aspect_ratio": "1:1",
        "watermark": False
    }
    
    headers = {'Authorization': f'Bearer {api_key}'}

    try:
        # Use 'with' statements for safe file handling during multipart upload
        with open(full_img_path, 'rb') as img_file, open(full_mask_path, 'rb') as mask_file:
            files = [
                ('image', (os.path.basename(full_img_path), img_file, 'image/jpeg')),
                ('mask', (os.path.basename(full_mask_path), mask_file, 'image/png'))
            ]
            
            response = requests.post(url, headers=headers, data=payload, files=files, timeout=60)
            response.raise_for_status()
            
            download_result(response.text, save_path)
            print(f"Success: Edited image saved to {save_path}")

    except FileNotFoundError:
        print(f"Error: Resource not found at {data_root}. Please check your data paths.")
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
