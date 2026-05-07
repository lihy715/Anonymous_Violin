## VIOLIN: Visual L4 Obedience Benchmark
#### Anonymous implementation of the paper: "Exploring the AI Obedience: Why is Generating a Pure Color Image Harder than CyberPunk?"

## 🎨 Introduction
TODO

The repository is structured into three specialized modules:

Metric System:

Open-Source Inference: 

Closed-Source Model Evaluate: 

## 🖼️ Qualitative Results

## 📂 Repository Structure
```
├── violin_metrics/         # Part 1: evaluation metrics
│   ├── color_metric.py        # metric for color purity task both var1 and var2
│   ├── mask_metric.py         # metric for image mask task
│   └── shape_metric.py        # metric for geometric generation task
├── eval_open_source/       # Part 2: Inference scripts for open source models
│   ├── 
│   └── 
├── eval_closed_source/     # Part 3: API-based model testing
│   ├── evaluate               # evaluation code
│   └── generate               # image generation code by API
├── benchmark/              # Part 4: Benchmark data
└── requirements       
│   ├── requirement_closed_source.txt
│   └── 
```

## Anonymous Dataset Download

We have provided a complete anonymous link for our data on ...  TODO

## 🚀 Generate and Evaluate with Open-Source Models
1. Installation

2. Running the Benchmark

## 📦Evaluate with Closed-Source Models

### Prerequisites
Please make sure your dataset is in ./benchmark. Run following command to prepare for evaluate. (You can freely choose your pytorch version).

```python
conda create -n violin python=3.10

pip install -r requirements/requirement_closed_source.txt
```

### API Key Setup
We utilized [website](https://api.bltcy.ai/) for API calls, which is an integrated platform for different models.
```
# For Linux/macOS
export GENERATIVE_API_KEY="your_api_key_here"

# For Windows (Command Prompt)
set GENERATIVE_API_KEY=your_api_key_here

```

### Generate Images

Run the following command to generate images on tasks and models, results will be saved in closed_source_results/.

```
# Available models: gpt, nano_banana, doubao

# For Color Purity Task variation-1, single block color
python eval_closed_source/generate/generate_color_var1_task.py --model gpt

# For Color Purity Task variation-2, double block color
python eval_closed_source/generate/generate_color_var2_task.py --model gpt

# For Image Mask Task
python eval_closed_source/generate/generate_mask_task.py --model nano_banana

# For Geometric Generation Task
python eval_closed_source/generate/generate_geometric_task.py --model doubao
```

### Evaluate
Run the following command to evaluate models on three tasks, results will be displayed in the terminal.

```
# Available models: gpt, nano_banana, doubao

# For Color Purity Task variation-1, single block color
python eval_closed_source/evaluate/evaluate_color_task.py --model gpt --var_id 1

# For Color Purity Task variation-2, double block color
python eval_closed_source/evaluate/evaluate_color_task.py --model gpt --var_id 2

# For Image Mask Task
python eval_closed_source/evaluate/evaluate_mask_task.py --model nano_banana

# For Geometric Generation Task
python eval_closed_source/evaluate/evaluate_geometric_task.py --model doubao
```



## 🤝 Acknowledgements

The implementation of our open-source model evaluation suite is built upon the following repositories. We express our sincere gratitude to the authors and contributors for their pioneering work:

*   **[FLUX.1](https://github.com/black-forest-labs/flux)**
*   **[FLUX.2]()**
*   **[Z-Image-Model]()**
*   **[Qwen-Image]()**


These repositories have significantly facilitated our research on visual obedience.


## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
