import torch
import matplotlib.pyplot as plt
import time
import os
import json
import threading
import pynvml
import numpy as np
from vllm import LLM, SamplingParams
import sys
sys.path.append('/work/nvme/bdkz/yyu69/vllm/experiments/chunk_size_vs_e2e_time')
from load_prompts import load_prompts_from_csv, load_prompts_from_json

import wandb

BATCH_SIZE = 8
NUM_RUNS_PER_BATCH = 1

class PrefillDecodeGPUComputeMonitor:
    def __init__(self, device_index=0, poll_interval=0.01):
        self.device_index = device_index
        self.poll_interval = poll_interval

        # Timeline measurements
        self.prefill_data = {}       # prefill_data[prompt_id] -> list of (time, utilization)
        self.decode_data = {}        # decode_data[prompt_id]  -> list of (time, utilization)

        # Prompt info
        self.prompt_token_lengths = {}  # prompt_token_lengths[prompt_id] -> int
        self.avg_prompt_lengths = {}    # avg_prompt_lengths[prompt_set_id] -> avg_tokens

        # We'll store the raw generation outputs here:
        self.decoded_outputs = {}       # decoded_outputs[prompt_id] -> text from model.generate()

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    def _poll_gpu_utilization(self, running_flag, data_store):
        """
        Continuously poll GPU utilization and store (elapsed_time, util)
        until running_flag[0] becomes False.
        """
        start_time = time.time()
        while running_flag[0]:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            elapsed = time.time() - start_time
            data_store.append((elapsed, util))
            time.sleep(self.poll_interval)

    def _run_with_monitoring(self, func):
        """
        1) Starts a monitoring thread
        2) Calls 'func()'
        3) Stops the monitoring thread
        4) Returns the collected data as a list of (time, util)
        """
        data_store = []
        running_flag = [True]
        monitor_thread = threading.Thread(
            target=self._poll_gpu_utilization,
            args=(running_flag, data_store)
        )
        monitor_thread.start()

        # Run the actual function
        func()

        # Signal the monitor thread to stop
        running_flag[0] = False
        monitor_thread.join()

        return data_store

    def process_batch_with_timing(self, prompt_set_id, llm, prompts, max_new_tokens=50):
        """
        Process a batch of prompts and record GPU utilization for prefill and decode phases.
        
        Args:
            prompt_set_id (str): Identifier for this set of prompts
            llm: The vLLM LLM object
            prompts (list): List of prompt strings to process
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            None
        """
        # Calculate average token length for this prompt set
        tokenizer = llm.get_tokenizer()
        total_tokens = sum(len(tokenizer.encode(p)) for p in prompts)
        avg_tokens = total_tokens / len(prompts)
        self.avg_prompt_lengths[prompt_set_id] = avg_tokens
        
        # Store token lengths for each prompt
        self.prompt_token_lengths[prompt_set_id] = total_tokens
        
        # Setup timing variables
        prefill_end_time = None
        total_end_time = None
        first_token_received = False
        
        # Measure GPU utilization during the entire process
        data_store = []
        running_flag = [True]
        monitor_thread = threading.Thread(
            target=self._poll_gpu_utilization,
            args=(running_flag, data_store)
        )
        monitor_thread.start()
        
        # Start timing
        start_time = time.time()
        
        # Generate text
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_new_tokens, max_num_seqs=8)
        outputs = llm.generate(prompts, sampling_params)
        
        # Detect when the first token is generated (prefill completion)
        # We'll consider prefill complete when the first output token is generated from any prompt
        for i, output in enumerate(outputs):
            if hasattr(output, 'generated_tokens') and len(output.generated_tokens) > 0:
                if not first_token_received:
                    prefill_end_time = time.time()
                    first_token_received = True
                    break
        
        # Record total time
        total_end_time = time.time()
        
        # Stop monitoring thread
        running_flag[0] = False
        monitor_thread.join()
        
        # Calculate times
        prefill_duration = prefill_end_time - start_time if prefill_end_time else None
        decode_duration = total_end_time - prefill_end_time if prefill_end_time else None
        
        # Separate the GPU utilization data for prefill and decode phases
        if prefill_end_time:
            prefill_cutoff = prefill_end_time - start_time
            prefill_data = [(t, util) for t, util in data_store if t <= prefill_cutoff]
            decode_data = [(t - prefill_cutoff, util) for t, util in data_store if t > prefill_cutoff]
            
            self.prefill_data[prompt_set_id] = prefill_data
            self.decode_data[prompt_set_id] = decode_data
        else:
            # If we couldn't detect when the first token was generated
            self.prefill_data[prompt_set_id] = data_store
            self.decode_data[prompt_set_id] = []
        
        # Save the generated texts
        self.decoded_outputs[prompt_set_id] = [output.outputs[0].text for output in outputs]
        
        # Print timing information
        print(f"Prompt set: {prompt_set_id}")
        print(f"  Average token length: {avg_tokens:.1f}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Number of prompts: {len(prompts)}")
        if prefill_duration:
            print(f"  Prefill time: {prefill_duration:.4f}s")
        else:
            print(f"  Prefill time: N/A (couldn't detect first token)")
        if decode_duration:
            print(f"  Decode time: {decode_duration:.4f}s")
        else:
            print(f"  Decode time: N/A")
        print(f"  Total time: {total_end_time - start_time:.4f}s")
        print("-" * 40)

    def plot_time_utilization(self, output_dir="."):
        """
        Modified to show batch information in the plot titles and printouts.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_fig_paths = []

        all_prompt_ids = set(self.prefill_data.keys()) | set(self.decode_data.keys())

        for pid in sorted(all_prompt_ids):
            prefill = self.prefill_data.get(pid, [])
            decode = self.decode_data.get(pid, [])

            if not prefill and not decode:
                continue

            # Durations
            prefill_duration = prefill[-1][0] if prefill else 0.0
            decode_duration = decode[-1][0] if decode else 0.0

            # Token length or average token length
            token_len = self.prompt_token_lengths.get(pid, "N/A")
            avg_tokens = self.avg_prompt_lengths.get(pid, "N/A")

            print(f"Prompt set {pid}:")
            print(f"  Average token length = {avg_tokens}")
            print(f"  Total tokens = {token_len}")
            print(f"  Prefill duration = {prefill_duration:.4f} seconds")
            print(f"  Decode duration = {decode_duration:.4f} seconds\n")

            # Shift decode times so it follows prefill
            decode_shifted = []
            if prefill:
                for (t, util) in decode:
                    decode_shifted.append((t + prefill_duration, util))
            else:
                decode_shifted = decode

            fig, ax = plt.subplots()

            # Prefill
            if prefill:
                times_prefill = [x[0] for x in prefill]
                utils_prefill = [x[1] for x in prefill]
                ax.plot(times_prefill, utils_prefill, label="Prefill")

            # Decode
            if decode_shifted:
                times_decode = [x[0] for x in decode_shifted]
                utils_decode = [x[1] for x in decode_shifted]
                ax.plot(times_decode, utils_decode, label="Decode")

            ax.set_title(f"GPU Utilization - {pid} - Avg {avg_tokens:.1f} tokens (Batch Size = {BATCH_SIZE})")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("GPU Utilization (%)")
            ax.set_ylim(0, 100)
            ax.legend()

            save_path = os.path.join(output_dir, f"gpu_util_{pid}.png")
            plt.savefig(save_path)
            plt.close(fig)
            saved_fig_paths.append((pid, save_path))

        return saved_fig_paths

    def plot_bar_gpu_utilization(self, output_path="gpu_bar.png", title="GPU Compute Utilization: Prefill vs Decode"):
        """
        Create a side-by-side bar chart of average GPU utilization 
        for prefill vs. decode, keyed by prompt token length.
        
        Returns output_path so we can log it to wandb.
        """
        prompt_ids_sorted = sorted(set(self.prefill_data.keys()) | set(self.decode_data.keys()))
        
        token_lengths = []
        avg_prefill_utils = []
        avg_decode_utils = []

        for pid in prompt_ids_sorted:
            prefill = self.prefill_data.get(pid, [])
            decode = self.decode_data.get(pid, [])

            # Use average token length for this prompt set
            token_len = self.avg_prompt_lengths.get(pid, 0)
            token_lengths.append(int(token_len))

            if len(prefill) > 0:
                avg_prefill = sum(u for (_, u) in prefill) / len(prefill)
            else:
                avg_prefill = 0.0

            if len(decode) > 0:
                avg_decode = sum(u for (_, u) in decode) / len(decode)
            else:
                avg_decode = 0.0

            avg_prefill_utils.append(avg_prefill)
            avg_decode_utils.append(avg_decode)

        x_indices = np.arange(len(prompt_ids_sorted))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars_prefill = ax.bar(
            x_indices - bar_width/2, avg_prefill_utils, 
            width=bar_width, label="Prefill", color="tab:blue"
        )
        bars_decode = ax.bar(
            x_indices + bar_width/2, avg_decode_utils, 
            width=bar_width, label="Decode", color="tab:orange"
        )

        ax.set_title(title)
        ax.set_ylabel("GPU Compute Utilization (%)")
        ax.set_ylim(0, 100)

        x_labels = [f"{tl} tokens" for tl in token_lengths]
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, rotation=0)

        ax.legend()

        # Numeric labels
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

        fig.tight_layout()

        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)

        return output_path

def run_compute_utilization_test(model_path, 
                                 prompt_files,
                                 output_dir=".", 
                                 max_new_tokens=50,
                                 use_wandb=False,
                                 wandb_project="gpu-utilization",
                                 wandb_run_name=None,
                                 tensor_parallel_size=1,
                                 csv_column_name="question"):
    """
    Main driver function to:
      1. Load model with vLLM & prompts from multiple files
      2. Process each set of prompts and capture GPU utilization
      3. Plot line charts & bar chart comparing prefill vs decode across prompt sets
      4. Optionally log to Weights & Biases (wandb).
    """
    # Optionally init wandb
    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_run_name)

    monitor = PrefillDecodeGPUComputeMonitor(device_index=0, poll_interval=0.01)

    # Initialize vLLM engine
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    
    # Process each prompt file
    for file_idx, prompt_file in enumerate(prompt_files):
        # Generate an ID for this prompt set
        file_name = os.path.basename(prompt_file).split('.')[0]
        prompt_set_id = f"set_{file_idx+1}_{file_name}"
        
        # Load prompts from file
        if prompt_file.endswith('.csv'):
            prompts = load_prompts_from_csv(prompt_file, column_name=csv_column_name)
        else:
            prompts = load_prompts_from_json(prompt_file)
        
        # Ensure we have enough prompts for the batch size
        if len(prompts) < BATCH_SIZE:
            print(f"Warning: Not enough prompts in {prompt_file}. Need {BATCH_SIZE}, but only have {len(prompts)}.")
            # Extend the list by repeating prompts if needed
            while len(prompts) < BATCH_SIZE:
                prompts.extend(prompts[:BATCH_SIZE - len(prompts)])
        
        # Limit to BATCH_SIZE prompts
        prompts = prompts[:BATCH_SIZE]
        
        print(f"\nProcessing prompt set {prompt_set_id} from {prompt_file}")
        print(f"Using {len(prompts)} prompts for evaluation")
        
        # Process this batch of prompts
        monitor.process_batch_with_timing(prompt_set_id, llm, prompts, max_new_tokens=max_new_tokens)
        
        # Print a sample of the generated outputs
        print(f"\nSample outputs from prompt set {prompt_set_id}:")
        outputs = monitor.decoded_outputs[prompt_set_id]
        for i, output in enumerate(outputs[:2]):  # Show just the first 2
            print(f"  Prompt {i+1}: {output[:100]}..." if len(output) > 100 else output)
        print("-" * 40)

    # Create plots
    line_plot_info = monitor.plot_time_utilization(output_dir)
    bar_plot_path = os.path.join(output_dir, "gpu_bar.png")
    monitor.plot_bar_gpu_utilization(
        output_path=bar_plot_path, 
        title="GPU Compute Utilization: Prefill vs Decode (Batch Size = 8)"
    )

    # Log to wandb if enabled
    if use_wandb:
        # Log the line plots
        for (prompt_set_id, fig_path) in line_plot_info:
            wandb.log({f"util_plot_{prompt_set_id}": wandb.Image(fig_path)})

        # Log the bar chart
        wandb.log({"gpu_bar_chart": wandb.Image(bar_plot_path)})

        # Calculate and log overall average utilization
        total_prefill, count_prefill = 0.0, 0
        total_decode, count_decode = 0.0, 0
        
        for pid in monitor.prefill_data.keys():
            for (_, util) in monitor.prefill_data[pid]:
                total_prefill += util
                count_prefill += 1
                
        for pid in monitor.decode_data.keys():
            for (_, util) in monitor.decode_data[pid]:
                total_decode += util
                count_decode += 1

        avg_prefill_overall = total_prefill / count_prefill if count_prefill else 0
        avg_decode_overall = total_decode / count_decode if count_decode else 0

        wandb.log({
            "average_prefill_util": avg_prefill_overall,
            "average_decode_util": avg_decode_overall
        })
        
        # Log per-prompt-set metrics
        for pid in sorted(set(monitor.prefill_data.keys()) | set(monitor.decode_data.keys())):
            avg_tokens = monitor.avg_prompt_lengths.get(pid, 0)
            
            prefill = monitor.prefill_data.get(pid, [])
            decode = monitor.decode_data.get(pid, [])
            
            if len(prefill) > 0:
                avg_prefill = sum(u for (_, u) in prefill) / len(prefill)
            else:
                avg_prefill = 0.0
                
            if len(decode) > 0:
                avg_decode = sum(u for (_, u) in decode) / len(decode)
            else:
                avg_decode = 0.0
                
            wandb.log({
                f"{pid}_avg_tokens": avg_tokens,
                f"{pid}_prefill_util": avg_prefill,
                f"{pid}_decode_util": avg_decode
            })

        wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path or name of the model.")
    parser.add_argument("--prompt-files", required=True, nargs='+', 
                        help="List of CSV or JSON files with prompts, one file per prompt set.")
    parser.add_argument("--csv-column", default="question", help="Column name for prompts in CSV files.")
    parser.add_argument("--output-dir", default=".", help="Output dir for the images.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Tokens for decode.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, 
                      help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--use-wandb", action='store_true', help="Log GPU usage to wandb?")
    parser.add_argument("--wandb-project", default="gpu-utilization", help="wandb project name.")
    parser.add_argument("--wandb-run-name", default=None, help="Optional wandb run name.")
    args = parser.parse_args()

    run_compute_utilization_test(
        model_path=args.model,
        prompt_files=args.prompt_files,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        tensor_parallel_size=args.tensor_parallel_size,
        csv_column_name=args.csv_column
    )
