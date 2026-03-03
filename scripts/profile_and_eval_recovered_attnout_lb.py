# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess
import sys
import csv
from collections import defaultdict

TOPK = 10  # Fixed TopK for LongBench evaluations

# LongBench datasets
longbench_datasets = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", 
    "hotpotqa", "2wikimqa", "musique", "dureader", 
    "gov_report", "qmsum", "multi_news", "vcsum", 
    "trec", "triviaqa", "samsum", "lsht", 
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", 
    "lcc", "repobench-p"
]

# Class mappings for averaging
dataset_classes = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5]
classes = ["Single-Doc QA", "Multi-Doc QA", "Summarization", "Fewshot", "Synthetic", "Code"]

# Dataset -> num_queries mapping
dataset_num_queries = {
    "lcc": 500,
    "repobench-p": 500,
    "multifieldqa_en": 150,
}

# Model configurations with model-specific strategy parameters
MODELS = [
    {
        "name": "Qwen/Qwen3-8B",
        "strategies": [
            {"name": "baseline", "args": []},
            {"name": "kascade", "args": [
                "--tile_size", "32",
                "--rolling_prefill",
                "--recompute_layers", "0", "2", "7", "13", "17", "23"
            ]},
        ]
    },
    {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "strategies": [
            {"name": "baseline", "args": []},
            {"name": "kascade", "args": [
                "--tile_size", "32",
                "--rolling_prefill",
                "--recompute_layers", "0", "2", "4", "8", "13", "16"
            ]},
        ]
    },
]


def run_profile(model_config, num_queries=1):
    """Run attn out profile for a model across all LongBench subsets,
        using 5 samples
    """
    model_name = model_config["name"]
    strategies = model_config["strategies"]
    
    # Build base command
    base_cmd = ["accelerate", "launch", "./scripts/eval_script.py"]
    
    # Add model
    base_cmd.extend(["--model_name", model_name])
    
    # Add dataset
    base_cmd.extend(["--dataset_name", "THUDM/LongBench"])
    
    # Add all subsets
    base_cmd.extend(["--subsets"] + longbench_datasets)
    
    # Add all strategy names
    strategy_names = ["baseline_profile"]
    base_cmd.extend(["--strategies"] + strategy_names)
    
    # Calculate num_queries (use max for simplicity)
    base_cmd.extend(["--num_queries", str(num_queries)])
    base_cmd.extend(["--topk", str(TOPK)])
    
    # Enable result storage
    base_cmd.append("--store_results")
    
    # Add strategy-specific arguments (assumes they apply globally for this model)
    # Note: This collects all unique args from all strategies
    # If strategies have conflicting args, you may need to run separately
    all_args = []
    for strategy in strategies:
        all_args.extend(strategy["args"])
    
    base_cmd.extend(all_args)
    
    print(f"Running evaluation for model: {model_name}")
    print(f"Command: {' '.join(base_cmd)}")
    
    # inherit all environment and run command 
    current_env = os.environ.copy()
    python_executable = sys.executable
    conda_bin_dir = os.path.dirname(python_executable)

    # Prepend the conda bin directory to the PATH in the subprocess environment
    current_env["PATH"] = f"{conda_bin_dir}:{current_env['PATH']}"

    result = subprocess.run(
        base_cmd,
        env=current_env, # Passes the Slurm & Conda variables
        check=True
    )


def main():
    """Main profile loop"""
    # Run profile for all models
    for model_config in MODELS:
        run_profile(model_config, num_queries=1)


if __name__ == "__main__":
    main()
