import pandas as pd
import os
import argparse
import sys
from tqdm import tqdm

# --- Environment Setup ---
# Ensure the script can locate internal modules in the current directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

try:
    # Import specific LLM calling functions for each model
    from gpt.ti2i import call_image_edit_api as gpt_call
    from nano_banana.ti2i import call_image_edit_api as nano_call
    from doubao.ti2i import call_image_edit_api as doubao_call
except ImportError as e:
    print(f"Error: Module import failed. Check directory structure or PYTHONPATH. \nDetail: {e}")
    sys.exit(1)

def get_eval_config(model_type, save_root):
    """
    Configures the generation function and output directory based on model type.
    """
    model_map = {
        'gpt': gpt_call,
        'nano_banana': nano_call,
        'doubao': doubao_call
    }

    if model_type not in model_map:
        raise ValueError(f"Model '{model_type}' not supported. Options: {list(model_map.keys())}")

    # Define subfolder for specific task
    save_dir = os.path.join(save_root, f"{model_type}_image_mask_inpainting")
    os.makedirs(save_dir, exist_ok=True)
    
    return model_map[model_type], save_dir

def run_evaluation(model_type):
    """
    Main logic for VIOLIN Image Mask Inpainting automation testing.
    """
    # Locate project root relative to script location
    project_root = os.path.abspath(os.path.join(current_script_dir, "../../"))
    
    # Default paths if not provided
    save_root = os.path.join(project_root, "closed_source_results")
    
    metadata_path = os.path.join(project_root, "benchmark/metadata/Task_Image_Mask_inpainting_metadata.csv")

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata not found at {metadata_path}")
        return

    # Load evaluation data
    try:
        df = pd.read_csv(metadata_path, usecols=['id', 'prompt', 'image1_path', 'image2_path'])
    except Exception as e:
        print(f"Error: Failed to load CSV: {e}")
        return

    # Map model to its function and output path
    gen_func, output_dir = get_eval_config(model_type, save_root)

    print(f"--- VIOLIN Benchmark Evaluation (Image Mask Inpainting) ---")
    print(f"Model: {model_type}")
    print(f"Saving results to: {output_dir}")

    # Iterate through prompts with a progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Inpaint"):
        current_id = row['id']
        current_prompt = row['prompt']
        
        # Path cleaning: ensure Unix-style paths for cross-platform compatibility
        img_path1 = row['image1_path'].replace("\\", "/")
        img_path2 = row['image2_path'].replace("\\", "/")
        
        # File naming with padding (e.g., id_000001.png) for correct sorting
        save_path = os.path.join(output_dir, f"id_{str(current_id).zfill(6)}.png")

        # Resume capability: skip if file already exists
        if os.path.exists(save_path):
            continue

        try:
            # Call the TI2I (Image-to-Image / Inpaint) generation function
            gen_func(current_prompt, img_path1, img_path2, save_path)
        except Exception as e:
            # Log error but continue with the next sample
            print(f"\n[!] Error at ID {current_id}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Image Mask Inpainting Tasks for Closed-Source Models.")
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['gpt', 'nano_banana', 'doubao'],
        help="Target model for evaluation"
    )
    
    args = parser.parse_args()
    run_evaluation(args.model)