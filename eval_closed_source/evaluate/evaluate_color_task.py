import os
import argparse
import sys
from typing import Tuple, List, Callable, Dict

current_script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

# Import specialized color metrics from the local module
from violin_metrics.color_metric import Color_metrics_from_img_list, Color_metrics_from_img_list_no_equal

def get_evaluation_configs(
    model_type: str, 
    var_id: int
) -> Tuple[Callable, List[str], List[str]]:
    """
    Dynamically configures evaluation paths relative to the project root.
    """
    # 1. Get the absolute path of the current script
    current_script = os.path.abspath(__file__)
    
    # 2. Trace back to the project root directory
    project_root = os.path.dirname(os.path.dirname((os.path.dirname(current_script))))

    # 3. Define paths relative to the project root
    test_results_root = os.path.join(project_root, "closed_source_results")
    benchmark_root = os.path.join(project_root, "benchmark", "data")
    
    # --- Remaining logic stays the same ---
    gt_imgs_dir = os.path.join(benchmark_root, f"Task_Color_Var{var_id}")
    test_imgs_dir = os.path.join(test_results_root, f"{model_type}_color_var{var_id}")

    # Standard validation and path collection
    if not os.path.exists(test_imgs_dir):
        raise FileNotFoundError(f"Test directory not found: {test_imgs_dir}")
    if not os.path.exists(gt_imgs_dir):
        raise FileNotFoundError(f"Ground truth directory not found: {gt_imgs_dir}")

    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    test_imgs_path = [
        os.path.join(test_imgs_dir, f) for f in sorted(os.listdir(test_imgs_dir)) 
        if f.lower().endswith(valid_exts)
    ]
    gt_imgs_path = [
        os.path.join(gt_imgs_dir, f) for f in sorted(os.listdir(gt_imgs_dir)) 
        if f.lower().endswith(valid_exts)
    ]

    metric_func = Color_metrics_from_img_list_no_equal

    return metric_func, test_imgs_path, gt_imgs_path



def run_evaluation(model_type: str, var_id: int, return_each_sample: bool = False):
    """
    Executes the color metric evaluation pipeline and logs the results.

    Args:
        model_type (str): The specific model architecture or version.
        var_id (int): Variation index determining the complexity (e.g., multi-block).
        return_each_sample (bool): If True, returns metrics for every individual image.
    """
    # Logical mapping for multi-block variations
    if var_id == 1:
        is_multi_block = False
    elif var_id == 2:
        is_multi_block = True
    else:
        print(f"Error: Unsupported Variation ID ({var_id}). Only 1 and 2 are implemented.")
        return 

    try:
        # Prepare paths and function
        metric_func, test_paths, gt_paths = get_evaluation_configs(model_type, var_id)
        
        # Execute metric calculation
        results = metric_func(
            test_paths, 
            gt_paths, 
            is_multi_block=is_multi_block, 
            return_each_sample=return_each_sample
        )
        
        # Format results for reporting (ensure all values are serializable floats)
        formatted_res = {k: float(v) for k, v in results.items()}

        # Generate Console Report
        report_width = 50
        print("\n" + "=" * report_width)
        print(f" COLOR METRIC ANALYSIS REPORT ".center(report_width, "|"))
        print(f" Model: {model_type.upper()} | Variation: {var_id} ".center(report_width, "-"))
        print("-" * report_width)

        for k, v in formatted_res.items():
            # Highlight the 'mean' score for better visibility
            if k.lower() == 'mean':
                print(f" >> {k:<12} : {v:.4f}")
            else:
                print(f"    {k:<12} : {v:.4f}")

        print("=" * report_width + "\n")

    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Pipeline for Color Accuracy Metrics.")
    parser.add_argument(
        '--model_type', 
        type=str, 
        required=True,
        help="Model Type"
    )
    parser.add_argument(
        '--var_id',
        type=int,
        required=True,
        help="Variation ID (1 for single block, 2 for multi-block)."
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help="Flag to return metrics for every individual sample."
    )
    
    args = parser.parse_args()
    run_evaluation(args.model_type, args.var_id, return_each_sample=args.detailed)