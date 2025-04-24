import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

def plot_time_utilization(results, output_dir):
    """
    Plot the time vs GPU utilization for each batch
    
    Args:
        results: List of result dictionaries from each run
        output_dir: Directory to save the plot
    """
    saved_fig_paths = []
    
    for result in results:
        token_length = result["token_length"]
        token_counts = result.get("token_counts", [])
        
        # Skip if we don't have timestamps or utilization data
        if "timestamps" not in result or "utilization_data" not in result:
            continue
        
        # Get timestamps and utilization data
        timestamps = result["timestamps"]
        utilization_data = result["utilization_data"]
        
        if not timestamps or not utilization_data:
            continue
            
        # Prefill and decode phases
        prefill_start = result.get("prefill_start", 0)
        prefill_end = result.get("prefill_end", 0)
        decode_end = result.get("decode_end", 0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot full utilization curve
        ax.plot(timestamps, utilization_data, color='gray', alpha=0.5)
        
        # Plot prefill phase
        prefill_indices = [i for i, t in enumerate(timestamps) 
                          if prefill_start <= t <= prefill_end]
        if prefill_indices:
            prefill_times = [timestamps[i] for i in prefill_indices]
            prefill_utils = [utilization_data[i] for i in prefill_indices]
            ax.plot(prefill_times, prefill_utils, label="Prefill", color='tab:blue')
        
        # Plot decode phase
        decode_indices = [i for i, t in enumerate(timestamps) 
                         if prefill_end <= t <= decode_end]
        if decode_indices:
            decode_times = [timestamps[i] for i in decode_indices]
            decode_utils = [utilization_data[i] for i in decode_indices]
            ax.plot(decode_times, decode_utils, label="Decode", color='tab:orange')
        
        # Calculate average utilization for each phase
        avg_prefill_util = result.get("prefill_utilization", 0)
        avg_decode_util = result.get("decode_utilization", 0)
        
        # Add phase markers
        if prefill_start is not None:
            ax.axvline(x=prefill_start, color='g', linestyle='--', label='Prefill Start')
        if prefill_end is not None:
            ax.axvline(x=prefill_end, color='r', linestyle='--', label='Prefill End/Decode Start')
        if decode_end is not None:
            ax.axvline(x=decode_end, color='b', linestyle='--', label='Decode End')
            
        # Add horizontal lines for averages
        if prefill_start is not None and prefill_end is not None:
            ax.axhline(y=avg_prefill_util, color='g', linestyle='-', 
                      xmin=prefill_start/timestamps[-1],
                      xmax=prefill_end/timestamps[-1])
            ax.text(prefill_start + (prefill_end-prefill_start)/2, avg_prefill_util + 2, 
                   f"Prefill: {avg_prefill_util:.1f}%", ha='center')
            
        if prefill_end is not None and decode_end is not None:
            ax.axhline(y=avg_decode_util, color='r', linestyle='-',
                      xmin=prefill_end/timestamps[-1], 
                      xmax=decode_end/timestamps[-1])
            ax.text(prefill_end + (decode_end-prefill_end)/2, avg_decode_util + 2, 
                   f"Decode: {avg_decode_util:.1f}%", ha='center')
        
        # Set plot labels and title
        ax.set_title(f"GPU Utilization - ~{token_length} tokens (Batch Size = {len(token_counts)})")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("GPU Utilization (%)")
        ax.set_ylim(0, 105)  # Give space for annotations
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        save_path = os.path.join(output_dir, f"gpu_util_time_{token_length}_tokens.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_fig_paths.append((token_length, save_path))
        
        # Print summary stats
        print(f"\nTime utilization results for {token_length} tokens:")
        print(f"  Prefill duration = {result.get('prefill_duration', 0):.4f} seconds")
        print(f"  Decode duration = {result.get('decode_duration', 0):.4f} seconds")
        print(f"  Prefill GPU utilization = {avg_prefill_util:.2f}%")
        print(f"  Decode GPU utilization = {avg_decode_util:.2f}%")
    
    return saved_fig_paths

def plot_bar_gpu_utilization(results, output_dir, output_filename="gpu_utilization_comparison.png"):
    """
    Create a side-by-side bar chart of average GPU utilization 
    for prefill vs. decode, keyed by prompt token length
    
    Args:
        results: List of result dictionaries from each run
        output_dir: Directory to save the plot
        output_filename: Filename for the bar chart
    """
    # Sort results by token length
    results = sorted(results, key=lambda x: x["token_length"])
    
    token_lengths = [r["token_length"] for r in results]
    prefill_utils = [r["prefill_utilization"] for r in results]
    decode_utils = [r["decode_utilization"] for r in results]
    
    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(token_lengths))
    width = 0.35
    
    bars_prefill = ax.bar(
        x - width/2, prefill_utils, width, 
        label='Prefill', color='tab:blue'
    )
    bars_decode = ax.bar(
        x + width/2, decode_utils, width, 
        label='Decode', color='tab:orange'
    )
    
    ax.set_title('GPU Compute Utilization: Prefill vs Decode')
    ax.set_xlabel('Token Length')
    ax.set_ylabel('GPU Utilization (%)')
    
    # Use token lengths as x-axis labels
    token_length_labels = [f"{tl} tokens" for tl in token_lengths]
    ax.set_xticks(x)
    ax.set_xticklabels(token_length_labels)
    
    # Add value labels on bars
    for bar in bars_prefill:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 3),
                  textcoords="offset points",
                  ha='center', va='bottom')
    
    for bar in bars_decode:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 3),
                  textcoords="offset points",
                  ha='center', va='bottom')
    
    ax.legend()
    ax.set_ylim(0, 105)  # Give space for annotations
    ax.grid(axis='y', alpha=0.3)
    
    # Save chart
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
