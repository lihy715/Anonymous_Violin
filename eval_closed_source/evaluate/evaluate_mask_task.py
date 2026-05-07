import os
import argparse
import re
import sys
from typing import Tuple, List, Callable, Dict

current_script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

# Import specialized mask metrics from the local evaluation suite
from violin_metrics.mask_metric import Mask_metrics_from_img_list, Mask_metrics_from_img_list_non_equal

def get_evaluation_configs(model_type: str) -> Tuple[Callable, List[str], List[str]]:
    """
    Dynamically configures paths for image mask evaluation relative to the project root.

    Returns:
        Tuple[Callable, List[str], List[str]]: Metric function and validated file path lists.
    """
    # Trace project root from script location: root/eval_closed_source/generate/generate/script.py
    current_script = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname((os.path.dirname(current_script))))

    # Define standardized root directories
    test_results_root = os.path.join(project_root, "closed_source_results")
    benchmark_root = os.path.join(project_root, "benchmark", "data", "Task_Image_Mask", "inpainting")


    # Specific directory for the target model
    test_imgs_dir = os.path.join(test_results_root, f"{model_type}_image_mask_inpainting")

    # Guard clauses for directory existence
    if not os.path.exists(test_imgs_dir):
        raise FileNotFoundError(f"Inference results directory missing: {test_imgs_dir}")
    if not os.path.exists(benchmark_root):
        raise FileNotFoundError(f"Ground truth benchmark directory missing: {benchmark_root}")

    # Standardized image extension filter
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')

    # Collect and sort paths to ensure consistent alignment across datasets
    test_imgs_path = [
        os.path.join(test_imgs_dir, f) for f in sorted(os.listdir(test_imgs_dir))
        if f.lower().endswith(valid_extensions)
    ]
    gt_imgs_path = [
        os.path.join(benchmark_root, f) for f in sorted(os.listdir(benchmark_root))
        if f.lower().endswith(valid_extensions)
    ]

    # Use 'non_equal' variant to handle potential set size mismatches or specific alignments
    metric_func = Mask_metrics_from_img_list_non_equal

    return metric_func, test_imgs_path, gt_imgs_path

def run_mask_evaluation(
    model_type: str, 
    return_each_sample: bool = False, 
    rescale_generated_image: bool = True
):
    """
    Executes the mask accuracy evaluation pipeline and generates a formatted report.

    Args:
        model_type (str): The model identifier for report headers.
        return_each_sample (bool): Whether to output per-image metric breakdowns.
        rescale_generated_image (bool): If true, aligns dimensions before metric calculation.
    """
    try:
        # Initialize configuration and paths
        metric_func, test_paths, gt_paths = get_evaluation_configs(model_type)

        # Run metric calculation logic
        results = metric_func(
            test_paths, 
            gt_paths, 
            rescale_generated_image=rescale_generated_image, 
            return_each_sample=return_each_sample
        )

        # Sanitize results into a serializable float dictionary
        formatted_res = {k: float(v) for k, v in results.items()}

        # Generate Console Report
        report_width = 55
        print("\n" + "=" * report_width)
        print(f" IMAGE MASK ACCURACY AUDIT ".center(report_width, "#"))
        print(f" Model: {model_type.upper()} ".center(report_width, "-"))
        print("-" * report_width)

        # Logic for prioritized display of metrics
        for k, v in formatted_res.items():
            if k.lower() in ['mean', 'average', 'iou', 'mAP']:
                # Highlight primary performance indicators
                print(f" >> {k:<15} : {v:.4f}")
            else:
                print(f"    {k:<15} : {v:.4f}")

        print("=" * report_width + "\n")

    except Exception as e:
        print(f"Critical Error during evaluation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Image Masking Quality Evaluation.")
    parser.add_argument(
        '--model_type', 
        type=str, 
        required=True,
    )
    parser.add_argument(
        '--no_rescale',
        action='store_false',
        dest='rescale',
        help="Disable automatic image rescaling before metric calculation."
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help="Enable granular reporting for individual samples."
    )
    
    # Defaults to rescaling=True
    parser.set_defaults(rescale=True)
    args = parser.parse_args()

    run_mask_evaluation(
        model_type=args.model_type, 
        return_each_sample=args.detailed, 
        rescale_generated_image=args.rescale
    )