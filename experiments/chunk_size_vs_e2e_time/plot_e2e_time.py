from typing import List, Dict
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse


def plot_e2e_time_chart(all_results: List[List[Dict]], output_dir: str):    
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
    e2e_plot_path = f"{output_dir}/combined_chunk_size_vs_e2e_time.png"
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
    detailed_csv_path = f"{output_dir}/detailed_timing_metrics.csv"
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
    summary_csv_path = f"{output_dir}/timing_summary.csv"
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
    timing_plot_path = f"{output_dir}/prefill_decode_time_breakdown.png"
    plt.savefig(timing_plot_path, dpi=300, bbox_inches='tight')
    print(f"Timing breakdown plot saved to {timing_plot_path}")
    
    plt.close()

def plot_e2e_time_chart_from_json(json_file_path: str, output_dir: str = None, model_name: str = ""):
    # Load data from JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Plot combined e2e time chart
    plt.figure(figsize=(12, 8))
    
    # Colors and markers for different prompts - updated for clarity
    styles = [
        {'color': 'blue', 'marker': 'o', 'label': 'Prompt 1'},
        {'color': 'green', 'marker': 's', 'label': 'Prompt 2'},
        {'color': 'red', 'marker': '^', 'label': 'Prompt 3'}
    ]
    
    # Process data from JSON structure
    all_chunk_sizes = set()
    prompt_results = []
    
    # Extract data from JSON
    for i, (prompt_key, prompt_data) in enumerate(data.items()):
        # Extract prompt tokens from the key (format: avg_prompt_tokens_XXX)
        prompt_tokens = int(prompt_key.split('_')[-1])
        
        results = []
        for chunk_key, metrics in prompt_data.items():
            # Extract chunk size from key (format: chunk_size_XXX)
            chunk_size = int(chunk_key.split('_')[-1])
            all_chunk_sizes.add(chunk_size)
            
            # Skip any entries with status='failed_startup' or error messages
            if isinstance(metrics, dict) and metrics.get('status') == 'failed_startup':
                continue
            if isinstance(metrics, dict) and 'error' in metrics:
                continue
                
            result = {
                "chunk_size": chunk_size,
                "avg_e2e_time": metrics["avg_e2e_time"],
                "avg_prefill_time": metrics["avg_prefill_time"],
                "avg_decode_time": metrics["avg_decode_time"],
                "avg_prompt_tokens": prompt_tokens
            }
            results.append(result)
        
        prompt_results.append(results)
    
    # Sort chunk sizes for the x-axis
    all_chunk_sizes = sorted(list(all_chunk_sizes))
    
    # Create evenly spaced x positions for plotting
    x_positions = np.arange(len(all_chunk_sizes))
    
    # Create mapping from chunk size to x position
    chunk_size_to_pos = {size: pos for pos, size in zip(x_positions, all_chunk_sizes)}
    
    # Plot each prompt's results
    for idx, results in enumerate(prompt_results):
        if idx >= len(styles):
            break  # Skip if we don't have a style defined
            
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
                label=f'{styles[idx]["label"]} ({avg_prompt_tokens} tokens)')
        
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
    plt.xticks(x_positions, all_chunk_sizes)
    
    plt.xlabel('Chunk Size')
    plt.ylabel('E2E Time (s)')
    title = 'Chunk Size vs Request End-to-End Time\nComparison of Different Prompt Lengths'
    if model_name:
        title += f" ({model_name})"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        e2e_plot_path = f"{output_dir}/combined_chunk_size_vs_e2e_time.png"
        plt.savefig(e2e_plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined E2E time plot saved to {e2e_plot_path}")
    else:
        plt.show()
    
    plt.close()