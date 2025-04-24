# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
import torch
import json
import time
import os
import csv
import datetime
import wandb
import requests
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import MethodType
from typing import List, Dict, Optional, Union, Any

def load_prompts(path: str) -> List[str]:
    try:
        with open(path, "r") as f:
            prompts_data = json.load(f)
            return prompts_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompts file {path} not found. Please create the file first.")

def load_prompts_from_csv(path: str, column_name: str = "question") -> List[str]:
    """
    Load prompts from a CSV file using the specified column.
    
    Args:
        path: Path to the CSV file containing prompts
        column_name: Name of the column containing the prompts (default: "question")
        
    Returns:
        List of prompt strings
    """
    try:
        # Read the CSV file
        prompts = []
        column_index = None
        
        with open(path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            # Get headers
            headers = next(reader)
            
            # Find column index
            try:
                column_index = headers.index(column_name)
            except ValueError:
                available_columns = ", ".join(headers)
                raise ValueError(f"Column '{column_name}' not found in CSV. Available columns: {available_columns}")
            
            # Extract prompts from the specified column
            for row in reader:
                if len(row) > column_index and row[column_index].strip():
                    prompts.append(row[column_index])
                    
                # Limit to first 8 prompts
                if len(prompts) >= 1:
                    break
        
        print(f"Loaded {len(prompts)} prompts from {path}")
        return prompts
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file {path} not found. Please create the file first.")
    except Exception as e:
        raise Exception(f"Error loading prompts from CSV: {str(e)}")

def get_prometheus_metric_value(metric_name: str, model_name: str, metrics_url: str = "http://localhost:8000/metrics") -> float:
    try:
        response = requests.get(metrics_url)
        response.raise_for_status()
        text = response.text

        pattern = rf'{re.escape(metric_name)}{{[^}}]*model_name="{re.escape(model_name)}"[^}}]*}} ([0-9.]+)'
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
        else:
            print(f"[WARN] Could not extract {metric_name}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to fetch metrics: {e}")
        return None

def run_inference_with_chunk_size(chunk_size, prompts, sampling_params, model_name):
    print(f"Initializing LLM with chunk_size={chunk_size}")
    
    # Create LLM with specified chunk size
    llm = LLM(
        model=model_name,
        enable_chunked_prefill=True,
        max_num_batched_tokens=chunk_size,
        max_num_seqs=chunk_size
    )

    e2e_times = []
    prefill_times = []
    decode_times = []
    prompt_tokens = []
    completion_tokens = []

    for i, prompt in enumerate(prompts):
        # Capture pre-inference metrics
        e2e_sum_before = get_prometheus_metric_value("vllm:e2e_request_latency_seconds_sum", model_name)
        prefill_sum_before = get_prometheus_metric_value("vllm:request_prefill_time_seconds_sum", model_name)
        decode_sum_before = get_prometheus_metric_value("vllm:request_decode_time_seconds_sum", model_name)

        # Run inference
        output = llm.generate([prompt], sampling_params)[0]

        # Capture post-inference metrics
        e2e_sum_after = get_prometheus_metric_value("vllm:e2e_request_latency_seconds_sum", model_name)
        prefill_sum_after = get_prometheus_metric_value("vllm:request_prefill_time_seconds_sum", model_name)
        decode_sum_after = get_prometheus_metric_value("vllm:request_decode_time_seconds_sum", model_name)

        # Compute timing deltas
        e2e_time = (e2e_sum_after - e2e_sum_before) if e2e_sum_before is not None and e2e_sum_after is not None else None
        prefill_time = (prefill_sum_after - prefill_sum_before) if prefill_sum_before is not None and prefill_sum_after is not None else None
        decode_time = (decode_sum_after - decode_sum_before) if decode_sum_before is not None and decode_sum_after is not None else None

        if e2e_time is None:
            print(f"[ERROR] Failed to extract E2E time from metrics for prompt {i}")
            continue

        # Tokens
        prompt_token_count = len(output.prompt_token_ids)
        completion_token_count = len(output.outputs[0].token_ids)
        generated_text = output.outputs[0].text

        e2e_times.append(e2e_time)
        prefill_times.append(prefill_time)
        decode_times.append(decode_time)
        prompt_tokens.append(prompt_token_count)
        completion_tokens.append(completion_token_count)

        print(f"Prompt {i}: {prompt_token_count} tokens -> {completion_token_count} generated, E2E: {e2e_time:.3f}s")

    # Averages
    avg_e2e_time = sum(e2e_times) / len(e2e_times)
    avg_prefill_time = sum(prefill_times) / len(prefill_times)
    avg_decode_time = sum(decode_times) / len(decode_times)
    avg_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens)
    avg_completion_tokens = sum(completion_tokens) / len(completion_tokens)

    total_tokens = sum(prompt_tokens) + sum(completion_tokens)
    total_time = sum(e2e_times)
    avg_token_per_sec = total_tokens / total_time if total_time > 0 else 0

    return {
        "chunk_size": chunk_size,
        "avg_e2e_time": avg_e2e_time,
        "avg_prefill_time": avg_prefill_time,
        "avg_decode_time": avg_decode_time,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_token_per_sec": avg_token_per_sec,
        "e2e_times": e2e_times,
        "prefill_times": prefill_times,
        "decode_times": decode_times,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }

def plot_combined_results(all_results: List[List[Dict]], output_dir: str, timestamp: str = None):
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    results_dir = f"{output_dir}/chunk_size_experiment_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot combined e2e time chart
    plt.figure(figsize=(12, 8))
    
    # Colors and markers for different prompts
    styles = [
        {'color': 'blue', 'marker': 'o', 'label': 'Prompt 1'},
        {'color': 'green', 'marker': 's', 'label': 'Prompt 2'},
        {'color': 'red', 'marker': '^', 'label': 'Prompt 3'}
    ]
    
    # Get all unique chunk sizes and create evenly spaced x positions
    all_chunk_sizes = sorted(list(set([
        size for results in all_results 
        for result in results 
        for size in [result["chunk_size"]]
    ])))
    
    # Create evenly spaced x positions for plotting
    x_positions = np.arange(len(all_chunk_sizes))
    
    # Create mapping from chunk size to x position
    chunk_size_to_pos = {size: pos for pos, size in zip(x_positions, all_chunk_sizes)}
    
    # Plot each prompt's results
    for idx, results in enumerate(all_results):
        chunk_sizes = [result["chunk_size"] for result in results]
        avg_e2e_times = [result["avg_e2e_time"] for result in results]
        avg_prompt_tokens = results[0]["avg_prompt_tokens"]
        
        # Sort by chunk size
        sorted_indices = np.argsort(chunk_sizes)
        sorted_chunk_sizes = [chunk_sizes[i] for i in sorted_indices]
        sorted_e2e_times = [avg_e2e_times[i] for i in sorted_indices]
        
        # Convert chunk sizes to x positions
        x_vals = [chunk_size_to_pos[size] for size in sorted_chunk_sizes]
        
        # Plot with different style for each prompt
        plt.plot(x_vals, sorted_e2e_times, 
                marker=styles[idx]['marker'], 
                color=styles[idx]['color'],
                linestyle='-',
                linewidth=2,
                markersize=8,
                label=f'{styles[idx]["label"]} (avg {avg_prompt_tokens:.1f} tokens)')
        
        # Add value annotations
        for x, y, chunk_size in zip(x_vals, sorted_e2e_times, sorted_chunk_sizes):
            plt.annotate(f"{y:.2f}s", 
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        color=styles[idx]['color'],
                        fontsize=8)
    
    # Set x-axis ticks to show all chunk sizes
    plt.xticks(x_positions, all_chunk_sizes, rotation=45)
    
    plt.xlabel('Chunk Size')
    plt.ylabel('E2E Time (s)')
    plt.title('Chunk Size vs Request End-to-End Time\nComparison of Different Prompt Lengths (llama-3-8b)')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    e2e_plot_path = f"{results_dir}/combined_chunk_size_vs_e2e_time.png"
    plt.savefig(e2e_plot_path, dpi=300, bbox_inches='tight')
    print(f"Combined E2E time plot saved to {e2e_plot_path}")
    
    plt.close()
    
    # Save detailed timing results as CSV
    detailed_results = []
    for prompt_idx, results in enumerate(all_results):
        for result in results:
            chunk_size = result["chunk_size"]
            for i in range(len(result["e2e_times"])):
                detailed_results.append({
                    "prompt_idx": prompt_idx + 1,
                    "chunk_size": chunk_size,
                    "e2e_time": result["e2e_times"][i],
                    "prefill_time": result["prefill_times"][i],
                    "decode_time": result["decode_times"][i],
                    "prompt_tokens": result["prompt_tokens"][i] if i < len(result["prompt_tokens"]) else None,
                })
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_csv_path = f"{results_dir}/detailed_timing_metrics.csv"
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"Detailed timing metrics saved to {detailed_csv_path}")
    
    # Create summary table with averages
    summary_results = []
    for prompt_idx, results in enumerate(all_results):
        for result in results:
            summary_results.append({
                "prompt_idx": prompt_idx + 1,
                "chunk_size": result["chunk_size"],
                "avg_prompt_tokens": result["avg_prompt_tokens"],
                "avg_e2e_time": result["avg_e2e_time"],
                "avg_prefill_time": result["avg_prefill_time"],
                "avg_decode_time": result["avg_decode_time"],
                "avg_token_per_sec": result["avg_token_per_sec"],
            })
    
    # Save summary results
    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = f"{results_dir}/timing_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Timing summary saved to {summary_csv_path}")
    
    # Create an additional plot for prefill and decode times
    plt.figure(figsize=(12, 8))
    
    # For each chunk size, plot e2e, prefill, and decode times as stacked bar chart
    chunk_sizes = sorted(list(set([result["chunk_size"] for results in all_results for result in results])))
    x = np.arange(len(chunk_sizes))
    width = 0.25
    
    # Get averages for each chunk size
    chunk_size_avg = {}
    for chunk_size in chunk_sizes:
        chunk_results = [r for results in all_results for r in results if r["chunk_size"] == chunk_size]
        chunk_size_avg[chunk_size] = {
            "prefill": sum(r["avg_prefill_time"] for r in chunk_results) / len(chunk_results),
            "decode": sum(r["avg_decode_time"] for r in chunk_results) / len(chunk_results)
        }
    
    # Plot prefill times
    prefill_times = [chunk_size_avg[cs]["prefill"] for cs in chunk_sizes]
    decode_times = [chunk_size_avg[cs]["decode"] for cs in chunk_sizes]
    
    plt.bar(x, prefill_times, width, label='Prefill Time')
    plt.bar(x, decode_times, width, bottom=prefill_times, label='Decode Time')
    
    plt.xlabel('Chunk Size')
    plt.ylabel('Time (s)')
    plt.title('Prefill and Decode Times by Chunk Size')
    plt.xticks(x, chunk_sizes)
    plt.legend()
    
    # Save timing breakdown plot
    timing_plot_path = f"{results_dir}/prefill_decode_time_breakdown.png"
    plt.savefig(timing_plot_path, dpi=300, bbox_inches='tight')
    print(f"Timing breakdown plot saved to {timing_plot_path}")
    
    plt.close()

def main():
    # Create timestamp for the experiment
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Initialize wandb with a dummy project if you don't want to use wandb directly
    wandb_run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=f"chunk_size_experiment_{timestamp}",
        config={
            "model": args.model,
            "output_dir": args.output_dir,
            "experiment_timestamp": timestamp,
            "experiment_name": args.experiment_name,
        }
    )
    
    print(f"Starting experiment at {timestamp}")
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
    
    # Print experiment configuration
    print("\nExperiment Configuration:")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sampling parameters: temperature={sampling_params.temperature}, top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}\n")
    
    # Define chunk sizes and prompt paths
    prompt_configs = [
        {
            "path": "/work/nvme/bdkz/yyu69/vllm/data/long-questions-1.csv",
            "chunk_sizes": [16, 32, 64]
        },
    ]
    
    all_results = []
    
    # Run experiments for each prompt file with its corresponding chunk sizes
    for config in prompt_configs:
        results = []
        prompts = load_prompts_from_csv(config["path"])
        
        print(f"\nRunning inference for prompts from {config['path']}")
        for chunk_size in config["chunk_sizes"]:
            print(f"\nRunning with chunk_size = {chunk_size}")
            result = run_inference_with_chunk_size(
                chunk_size=chunk_size,
                prompts=prompts,
                sampling_params=sampling_params,
                model_name=args.model
            )
            results.append(result)
        
        all_results.append(results)
    
    # Plot combined results
    plot_combined_results(all_results, args.output_dir, timestamp)
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/work/nvme/bdkz/yyu69/vllm/experiment_results", help="Directory to save experiment results and plots")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to use for inference")
    parser.add_argument("--experiment-name", type=str, default="", help="Optional name for the experiment (will be included in output directory)")
    parser.add_argument("--wandb-project", type=str, default="sarathi-chunk-size-experiment", help="Weights & Biases project name")
    parser.add_argument("--wandb-group", type=str, default="chunk-size-vs-e2e-time", help="Weights & Biases group name")
    args = parser.parse_args()
    
    main()
