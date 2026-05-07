import os
import argparse
import sys
from typing import Tuple, List, Callable, Dict

current_script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

# Import specialized shape metrics
from violin_metrics.shape_metric import Shape_metrics_from_img_list

def get_evaluation_configs(
    model_type: str, 
    square: bool = True
) -> Tuple[Callable, List[str], List[str]]:
    """
    Dynamically configures evaluation paths and selects the appropriate shape metric function.

    Args:
        model_type (str): The identifier for the model under test.
        square (bool): Toggle between square and non-square metric logic.

    Returns:
        Tuple[Callable, List[str], List[str]]: A tuple containing the metric function, 
            validated test image paths, and ground truth image paths.
    """

    current_script = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname((os.path.dirname(current_script))))

    # Define standardized root directories relative to the project root
    test_results_root = os.path.join(project_root, "closed_source_results")
    benchmark_root = os.path.join(project_root, "benchmark", "data", "Task_Geometric")
    
    # Model-specific test directory
    test_imgs_dir = os.path.join(test_results_root, f"{model_type}_geometric")

    # Guard clauses to ensure directories exist
    if not os.path.exists(test_imgs_dir):
        raise FileNotFoundError(f"Test results directory not found: {test_imgs_dir}")
    if not os.path.exists(benchmark_root):
        raise FileNotFoundError(f"Benchmark ground truth directory not found: {benchmark_root}")

    # Standardized image extension filter
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')

    # Collect and sort paths to ensure consistent alignment between datasets
    test_imgs_path = [
        os.path.join(test_imgs_dir, f) for f in sorted(os.listdir(test_imgs_dir)) 
        if f.lower().endswith(valid_exts)
    ]
    gt_imgs_path = [
        os.path.join(benchmark_root, f) for f in sorted(os.listdir(benchmark_root)) 
        if f.lower().endswith(valid_exts)
    ]

    # Select the appropriate metric function based on geometric constraints
    metric_func = Shape_metrics_from_img_list

    return metric_func, test_imgs_path, gt_imgs_path

def run_shape_evaluation(
    model_type: str, 
    square: bool, 
    return_each_sample: bool = False
):
    """
    Executes the shape geometry evaluation pipeline and logs the results.

    Args:
        model_type (str): The specific model architecture or version.
        square (bool): Whether the input/output images are expected to be square.
        return_each_sample (bool): If True, provides per-sample metric granularity.
    """
    try:
        # Initialize paths and metric logic
        metric_func, test_paths, gt_paths = get_evaluation_configs(model_type, square=square)
        
        # Perform geometric metric calculation
        results = metric_func(test_paths, gt_paths, return_each_sample=return_each_sample)
        
        # Standardize results into a serializable float format
        formatted_res = {k: float(v) for k, v in results.items()}

        # Generate Console Report
        report_width = 50
        print("\n" + "=" * report_width)
        print(f" SHAPE GEOMETRY METRIC REPORT ".center(report_width, "#"))
        print(f" Model: {model_type.upper()} | Mode: {'SQUARE' if square else 'NON-SQUARE'} ".center(report_width, "-"))
        print("-" * report_width)


        for k, v in formatted_res.items():
            # Highlight primary metrics like 'mean' or 'IoU'
            if k.lower() in ['mean', 'average']:
                print(f" >> {k:<12} : {v:.4f}")
            else:
                print(f"    {k:<12} : {v:.4f}")

        print("=" * report_width + "\n")

    except Exception as e:
        print(f"Critical Error during shape evaluation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Shape Integrity & Geometry Evaluation.")
    parser.add_argument(
        '--model_type', 
        type=str, 
        required=True,
        help="Identifier for the model (e.g., gpt-4o, sd-xl)."
    )
    parser.add_argument(
        '--square', 
        action='store_true',
        help="Flag to enable square-specific shape metrics (defaults to non-square if omitted)."
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help="Flag to return individual metric data for every sample in the batch."
    )
    
    args = parser.parse_args()

    run_shape_evaluation(
        model_type=args.model_type, 
        square=args.square, 
        return_each_sample=args.detailed
    )