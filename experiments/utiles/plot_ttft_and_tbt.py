import matplotlib.pyplot as plt
import json
import numpy as np
import os

# Load the data for each model
# with open('/work/nvme/bdkz/yyu69/vllm/experiment_results/chunk_size_experiment_20250423_023236/Qwen2.5-3B-Instruct/chunk_size_results_Qwen2.5-3B-Instruct.json', 'r') as f:
#     data_3b = json.load(f)

# with open('/work/nvme/bdkz/yyu69/vllm/experiment_results/chunk_size_experiment_20250423_023236/Qwen2.5-7B-Instruct/chunk_size_results_Qwen2.5-7B-Instruct.json', 'r') as f:
#     data_7b = json.load(f)

with open('/work/nvme/bdkz/yyu69/vllm/experiment_results/chunk_size_vs_e2e_time_experiments/chunk_size_vs_e2e_time_experiment_20250426_005614/Qwen2.5-14B-Instruct/chunk_size_results_Qwen2.5-14B-Instruct.json', 'r') as f:
    data_14b = json.load(f)

# Extract chunk sizes and metrics for each model
chunk_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Define models and their data
models = [
    # {
    #     "name": "Qwen2.5-3B-Instruct",
    #     "data": data_3b,
    #     "save_dir": "/work/nvme/bdkz/yyu69/vllm/experiment_results/chunk_size_experiment_20250423_023236/Qwen2.5-3B-Instruct"
    # },
    # {
    #     "name": "Qwen2.5-7B-Instruct",
    #     "data": data_7b,
    #     "save_dir": "/work/nvme/bdkz/yyu69/vllm/experiment_results/chunk_size_experiment_20250423_023236/Qwen2.5-7B-Instruct"
    # },
    {
        "name": "Qwen2.5-14B-Instruct",
        "data": data_14b,
        "save_dir": "/work/nvme/bdkz/yyu69/vllm/experiment_results/chunk_size_vs_e2e_time_experiments/chunk_size_vs_e2e_time_experiment_20250426_005614/Qwen2.5-14B-Instruct"
    }
]

# Plot TTFT and TBT for each model separately
for model in models:
    model_name = model["name"]
    model_data = model["data"]
    save_dir = model["save_dir"]
    
    # TTFT data (avg_prefill_time)
    ttft = [model_data["avg_prompt_tokens_5180"][f"chunk_size_{cs}"]["avg_prefill_time"] for cs in chunk_sizes]
    
    # TBT data (avg_time_between_tokens)
    tbt = [model_data["avg_prompt_tokens_5180"][f"chunk_size_{cs}"]["avg_time_between_tokens"] for cs in chunk_sizes]
    
    # Create evenly spaced x positions for plotting
    x_positions = np.arange(len(chunk_sizes))
    
    # Plot TTFT with improved styling and evenly spaced x positions
    plt.figure(figsize=(12, 8))
    line = plt.plot(x_positions, ttft, 'o-', color='blue', linewidth=2, markersize=8, label='Prompt 1 (5180 tokens)')
    
    # Add value annotations
    for x, y in zip(x_positions, ttft):
        plt.annotate(f"{y:.2f}s", 
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    color='blue',
                    fontsize=8)
    
    plt.xlabel('Chunk Size')
    plt.ylabel('Time to First Token (s)')
    plt.title(f'Chunk Size vs TTFT ({model_name})')
    plt.grid(True)
    plt.legend()
    
    # Set x-axis ticks to show all chunk sizes at evenly spaced positions
    plt.xticks(x_positions, chunk_sizes)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f'chunk_size_vs_ttft_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot TBT with improved styling and evenly spaced x positions
    plt.figure(figsize=(12, 8))
    plt.plot(x_positions, tbt, 'o-', color='blue', linewidth=2, markersize=8, label='Prompt 1 (1274 tokens)')
    
    # Add value annotations
    for x, y in zip(x_positions, tbt):
        plt.annotate(f"{y:.2f}s", 
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    color='blue',
                    fontsize=8)
    
    plt.xlabel('Chunk Size')
    plt.ylabel('Time Between Tokens (s)')
    plt.title(f'Chunk Size vs TBT ({model_name})')
    plt.grid(True)
    plt.legend()
    
    # Set x-axis ticks to show all chunk sizes at evenly spaced positions
    plt.xticks(x_positions, chunk_sizes)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f'chunk_size_vs_tbt_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
