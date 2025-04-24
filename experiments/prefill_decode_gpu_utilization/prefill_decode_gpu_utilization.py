import argparse
import json
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import os
import re
import requests
import subprocess
import signal
import pynvml
import openai
import torch
import sys
import queue

# Add parent directory to Python path for importing utilities
sys.path.append('/work/nvme/bdkz/yyu69/vllm')
from experiments.utiles.load_prompts import load_prompts_from_csv, load_prompts_from_json

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
HEALTH_CHECK_URL = f"{SERVER_URL}/health"
OUTPUT_DIR = "/work/nvme/bdkz/yyu69/vllm/experiment_results/prefill_decode_gpu_utilization_experiments/prefill_decode_gpu_utilization_experiment"

# CSV prompt files
PROMPT_FILES = [
    "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/select-text-by-length_1250_1375.csv",
    "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/select-text-by-length_2500_2750.csv",
    "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/select-text-by-length_5000_5500.csv",
    "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/select-text-by-length_10000_11000.csv",
    "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/select-text-by-length_20000_22000.csv",
]

BATCH_SIZE = 8  # Fixed batch size per your requirement
MAX_GEN_TOKENS = 64  # Small number to keep experiment focused

# --- GPU Monitoring ---
class GPUMonitor:
    def __init__(self, poll_interval=0.01, device_id=0):
        """
        Initialize GPU monitoring with pynvml
        
        Args:
            poll_interval: How often to poll GPU metrics in seconds
            device_id: GPU device ID to monitor
        """
        self.poll_interval = poll_interval
        self.device_id = device_id
        self.monitoring = False
        self.utilization_data = []
        self.timestamps = []
        self.phase_markers = []  # To mark prefill/decode phase transitions
        
    def start_monitoring(self):
        """Start GPU monitoring in a separate thread"""
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        self.monitoring = True
        self.utilization_data = []
        self.timestamps = []
        self.phase_markers = []
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """Main monitoring loop that collects GPU metrics"""
        self._start_time = time.time()
        
        while self.monitoring:
            # Get GPU utilization (0-100%)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            
            # Record timestamp and GPU utilization
            current_time = time.time() - self._start_time
            self.timestamps.append(current_time)
            self.utilization_data.append(utilization.gpu)
            
            time.sleep(self.poll_interval)
    
    def mark_phase(self, phase_name):
        """Mark a phase transition in the monitoring data"""
        if not self.monitoring:
            print(f"Warning: Cannot mark phase '{phase_name}': monitoring not active")
            return
        
        if not self.timestamps:
            print(f"Warning: Cannot mark phase '{phase_name}': no timestamps recorded yet")
            return
        
        current_time = time.time() - self._start_time if hasattr(self, '_start_time') else self.timestamps[-1]
        self.phase_markers.append((current_time, phase_name))
        print(f"Marked phase '{phase_name}' at time {current_time:.4f}s")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        pynvml.nvmlShutdown()
    
    def get_utilization_metrics(self, avg_token_length):
        """
        Calculate utilization metrics for prefill and decode phases
        
        Args:
            avg_token_length: Average token length for this batch
            
        Returns:
            Dictionary with utilization metrics and raw data
        """
        # Find phase markers
        prefill_start = None
        prefill_end = None
        decode_end = None
        
        for time_point, phase in self.phase_markers:
            if phase == "prefill_start":
                prefill_start = time_point
            elif phase == "prefill_end":
                prefill_end = time_point
            elif phase == "decode_end":
                decode_end = time_point
        
        # Calculate average utilization for prefill and decode phases
        prefill_indices = []
        if prefill_start is not None and prefill_end is not None:
            prefill_indices = [i for i, t in enumerate(self.timestamps) 
                             if prefill_start <= t <= prefill_end]
        
        decode_indices = []
        if prefill_end is not None and decode_end is not None:
            decode_indices = [i for i, t in enumerate(self.timestamps) 
                            if prefill_end <= t <= decode_end]
        
        # Print detailed utilization data for prefill phase
        print("\nDetailed Prefill Phase GPU Utilization:")
        print("Phase Markers: prefill_start={:.4f}s, prefill_end={:.4f}s".format(
            prefill_start if prefill_start is not None else 0,
            prefill_end if prefill_end is not None else 0
        ))
        print("Prefill Duration: {:.4f}s".format(
            prefill_end - prefill_start if prefill_start is not None and prefill_end is not None else 0
        ))
        print("Number of data points in prefill phase: {}".format(len(prefill_indices)))
        
        # Prepare detailed prefill data for JSON export
        detailed_prefill_data = []
        
        if prefill_indices:
            # Print header
            print("\n{:<12} {:<12} {:<12}".format("Index", "Timestamp(s)", "GPU Util(%)"))
            print("-" * 40)
            
            # Print each data point and collect for JSON
            for idx in prefill_indices:
                print("{:<12} {:<12.4f} {:<12}".format(
                    idx, 
                    self.timestamps[idx],
                    self.utilization_data[idx]
                ))
                detailed_prefill_data.append({
                    "index": idx,
                    "timestamp": float(self.timestamps[idx]),
                    "gpu_utilization": float(self.utilization_data[idx])
                })
            
            # Print statistics
            utilization_values = [self.utilization_data[i] for i in prefill_indices]
            print("\nPrefill Statistics:")
            print("Min: {:.2f}%".format(min(utilization_values)))
            print("Max: {:.2f}%".format(max(utilization_values)))
            print("Median: {:.2f}%".format(np.median(utilization_values)))
            print("Mean: {:.2f}%".format(np.mean(utilization_values)))
            print("Std Dev: {:.2f}%".format(np.std(utilization_values)))
            
            # Print utilization histogram
            print("\nUtilization Distribution:")
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            hist, _ = np.histogram(utilization_values, bins=bins)
            for i in range(len(bins)-1):
                print(f"{bins[i]}-{bins[i+1]}%: {'#' * int(hist[i]/max(1, max(hist))*20)} ({hist[i]})")
        
        # Calculate average utilization
        prefill_util = np.mean([self.utilization_data[i] for i in prefill_indices]) if prefill_indices else 0
        decode_util = np.mean([self.utilization_data[i] for i in decode_indices]) if decode_indices else 0
        
        # Prepare prefill statistics for JSON export
        prefill_stats = {
            "min": float(min(utilization_values)) if utilization_values else 0,
            "max": float(max(utilization_values)) if utilization_values else 0,
            "median": float(np.median(utilization_values)) if utilization_values else 0,
            "mean": float(np.mean(utilization_values)) if utilization_values else 0,
            "std_dev": float(np.std(utilization_values)) if utilization_values else 0,
            "histogram": [float(h) for h in hist] if 'hist' in locals() else [],
            "histogram_bins": bins if 'bins' in locals() else []
        }
        
        # Return metrics
        return {
            "token_length": avg_token_length,
            "prefill_utilization": float(prefill_util),
            "decode_utilization": float(decode_util),
            "prefill_duration": float(prefill_end - prefill_start) if prefill_start is not None and prefill_end is not None else 0,
            "decode_duration": float(decode_end - prefill_end) if prefill_end is not None and decode_end is not None else 0,
            "prefill_start": prefill_start,
            "prefill_end": prefill_end,
            "decode_end": decode_end,
            # Store timestamps and utilization data for custom plots
            "timestamps": self.timestamps,
            "utilization_data": self.utilization_data,
            # Add detailed prefill phase data and statistics
            "detailed_prefill_data": detailed_prefill_data,
            "prefill_statistics": prefill_stats
        }

# --- Server Management ---
def log_reader(process, log_queue):
    """Read logs from process and put them in queue."""
    for line in iter(process.stdout.readline, ''):
        log_queue.put(line.strip())
    log_queue.put(None)  # Signal end of stream

def start_vllm_server(model_name, no_chunked_prefill=True):
    """
    Start the vLLM server without chunked prefill
    
    Args:
        model_name: Model to load
        no_chunked_prefill: Set to True to disable chunked prefill
    """
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--max-num-seqs", "16",
        "--max-model-len", "8192"
    ]
    
    # Use the --no-enable-chunked-prefill flag instead of trying to set value to False
    if no_chunked_prefill:
        command.append("--enable-chunked-prefill=False")
        print("Running with chunked prefill explicitly disabled")
    else:
        command.append("--enable-chunked-prefill")
        print("Running with chunked prefill enabled")
        
    print(f"Starting server with command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                               text=True, bufsize=1, universal_newlines=True, 
                               preexec_fn=os.setsid)
    
    # Start log reader thread
    log_queue = queue.Queue()
    log_thread = threading.Thread(target=log_reader, args=(process, log_queue))
    log_thread.daemon = True
    log_thread.start()
    
    return process, log_queue

def wait_for_server_ready(process, log_queue, timeout=300):
    """Wait for the server to be ready using both logs and HTTP polling."""
    print(f"Waiting for server to be ready at {HEALTH_CHECK_URL}...")
    start_time = time.time()
    
    # Check for both log indicator and HTTP readiness
    log_ready_indicators = [
        "Uvicorn running on",
        "Starting vLLM API server on",
        "Application startup complete",
        "Chunked prefill is enabled",
        "INFO .* args:"
    ]
    
    server_ready = False
    
    while time.time() - start_time < timeout:
        # Check if process is still running
        if process.poll() is not None:
            print(f"Server process terminated unexpectedly with code {process.poll()}.")
            return False
        
        # Check logs (non-blocking)
        try:
            while not log_queue.empty():
                line = log_queue.get_nowait()
                if line is None:  # End of stream
                    break
                print(f"Server log: {line}")
                for indicator in log_ready_indicators:
                    if re.search(indicator, line):
                        print(f"Server ready indicator found in logs: '{indicator}'")
                        # Found indicator in logs, but give some extra time for server to be fully ready
                        time.sleep(3)
                        server_ready = True
        except queue.Empty:
            pass
        
        # Check HTTP endpoint
        try:
            response = requests.get(HEALTH_CHECK_URL, timeout=5)
            if response.status_code == 200:
                print(f"Server responded OK at {HEALTH_CHECK_URL}.")
                return True
        except requests.ConnectionError:
            if server_ready:
                print(f"Server logs indicated ready, but HTTP endpoint not responding yet. Retrying...")
            pass
        except requests.Timeout:
            print(f"Timeout polling {HEALTH_CHECK_URL}, retrying...")
        except Exception as e:
            print(f"Error polling {HEALTH_CHECK_URL}: {e}")
        
        time.sleep(3)

    print("Server failed to become ready within timeout.")
    return False

def shutdown_server(process):
    """Shut down the vLLM server process"""
    if process and process.poll() is None:
        print("Shutting down server...")
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGINT)
            process.wait(timeout=30)
            print("Server shut down via SIGINT.")
        except Exception as e:
            print(f"Error shutting down server: {e}")
            try:
                process.terminate()
                process.wait(timeout=10)
            except:
                pass
    
    # Continue reading any remaining logs for a few seconds (optional)
    if 'log_queue' in globals() and log_queue:
        print("Reading any final server logs...")
        end_time = time.time() + 3
        while time.time() < end_time:
            try:
                if not log_queue.empty():
                    line = log_queue.get_nowait()
                    if line is None:
                        break
                    print(f"Server log: {line}")
                else:
                    time.sleep(0.1)
            except queue.Empty:
                break

# --- Prompt Loading and Token Counting ---
def calculate_token_count(prompt, tokenizer):
    """Calculate the number of tokens in a prompt"""
    return len(tokenizer.encode(prompt))

def load_prompts_and_calculate_tokens(file_path, tokenizer):
    """
    Load prompts from CSV and calculate their token lengths
    
    Args:
        file_path: Path to CSV file with prompts
        tokenizer: Tokenizer to use for token counting
    
    Returns:
        tuple: (prompts, token_counts, avg_token_count)
    """
    prompts = load_prompts_from_csv(file_path, column_name="text")
    token_counts = [calculate_token_count(prompt, tokenizer) for prompt in prompts]
    avg_token_count = int(np.mean(token_counts[:BATCH_SIZE])) if token_counts else 0
    
    print(f"Loaded {len(prompts)} prompts from {file_path}")
    print(f"Average token count for first {BATCH_SIZE} prompts: {avg_token_count}")
    
    # Return only the first BATCH_SIZE prompts for our experiment
    return prompts[:BATCH_SIZE], token_counts[:BATCH_SIZE], avg_token_count

# --- Main Experiment ---
def run_experiment(prompt_files, model_name, output_dir, batch_size=8):
    """
    Run the experiment to measure GPU utilization during prefill and decode
    
    Args:
        prompt_files: List of CSV files with prompts
        model_name: Model to use
        output_dir: Directory to save results
        batch_size: Number of prompts to use in each batch
    """
    # Set environment variable to use V0 engine
    os.environ["VLLM_USE_V1"] = "0"
    
    # Create timestamped output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Start the vLLM server without chunked prefill
    server_process, log_queue = start_vllm_server(model_name, no_chunked_prefill=True)
    
    try:
        # Wait for server to be ready
        if not wait_for_server_ready(server_process, log_queue):
            print("Failed to start server. Exiting.")
            return

        time.sleep(5)  # Additional wait time to ensure everything is initialized
        
        # Initialize OpenAI client
        client = openai.OpenAI(
            base_url=f"{SERVER_URL}/v1",
            api_key="EMPTY"
        )
        
        # Run multiple warm-up generations to ensure CUDA graphs are built
        print("Running warm-up generations...")
        warmup_prompt = "Hello, how are you doing today? Can you tell me about yourself in a few sentences?"

        # First warm-up with short output (for CUDA graph compilation)
        warmup_response = client.completions.create(
            model=model_name,
            prompt=warmup_prompt,
            max_tokens=1
        )

        # Second warm-up with normal output (for prefill/decode paths)
        warmup_response = client.completions.create(
            model=model_name,
            prompt=warmup_prompt,
            max_tokens=32
        )

        # Third warm-up with a batch (similar to main experiment)
        batch_prompt = ["Hello there"] * BATCH_SIZE
        warmup_response = client.completions.create(
            model=model_name,
            prompt=batch_prompt,
            max_tokens=8
        )

        print("Warm-up generations completed.")
        time.sleep(5)  # Longer wait to ensure everything is settled
        
        # Test each CSV file (which corresponds to a different token length range)
        for csv_file in prompt_files:
            print(f"\n{'='*20}\nTesting prompts from: {csv_file}\n{'='*20}")
            
            # Load prompts and calculate token counts
            prompts, token_counts, avg_token_count = load_prompts_and_calculate_tokens(csv_file, tokenizer)
            
            if not prompts:
                print(f"No prompts loaded from {csv_file}. Skipping.")
                continue
                
            # Limit to batch_size
            if len(prompts) > batch_size:
                prompts = prompts[:batch_size]
                token_counts = token_counts[:batch_size]
            
            # Initialize GPU monitor
            gpu_monitor = GPUMonitor(poll_interval=0.001)
            gpu_monitor.start_monitoring()
            
            # Wait longer to ensure monitoring is active and stable
            time.sleep(0.5)

            try:
                # Start timing just before you send the request
                print(f"Sending batch request with {len(prompts)} prompts (avg {avg_token_count} tokens each)")

                # Send request with streaming to detect first token
                start_time = time.time()
                first_token_received = False
                stream = client.completions.create(
                    model=model_name,
                    prompt=prompts,
                    max_tokens=MAX_GEN_TOKENS,
                    temperature=0.8,
                    top_p=0.95,
                    stream=True
                )

                gpu_monitor.mark_phase("prefill_start")

                # Collect the response
                full_response = []
                collected_chunks = [[] for _ in range(len(prompts))]
                for chunk in stream:
                    if not first_token_received:
                        # First token received marks the end of prefill
                        gpu_monitor.mark_phase("prefill_end")
                        first_token_received = True
                        print(f"First token received at {time.time() - start_time:.4f}s")
                    
                    # Collect the chunk
                    full_response.append(chunk)
                    
                    # Store the tokens by prompt index if available
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        if hasattr(choice, 'index') and hasattr(choice, 'text'):
                            prompt_idx = choice.index
                            if prompt_idx < len(collected_chunks):
                                collected_chunks[prompt_idx].append(choice.text)

                # Mark end of decode phase
                gpu_monitor.mark_phase("decode_end")
                
                # Wait for generation to complete
                end_time = time.time()
                print(f"Generation completed in {end_time - start_time:.2f} seconds")
                
                # Print outputs for each prompt
                print("\nGenerated Outputs:\n" + "-" * 60)
                for i, (prompt, tokens) in enumerate(zip(prompts[:len(collected_chunks)], collected_chunks)):
                    output_text = ''.join(tokens)
                    
                    # Print truncated prompt (first 50 chars) to avoid too much output
                    truncated_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
                    print(f"Prompt {i+1}: {truncated_prompt}")
                    
                    # Print truncated output (first 100 chars) to avoid too much output
                    truncated_output = output_text[:100] + "..." if len(output_text) > 100 else output_text
                    print(f"Generated: {truncated_output}")
                    print("-" * 60)

                # Save the complete outputs to a file
                outputs_file = os.path.join(output_dir, f"outputs_{avg_token_count}_tokens.json")
                with open(outputs_file, 'w') as f:
                    output_data = [
                        {
                            "prompt_idx": i,
                            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,  # Truncate for readability
                            "output": ''.join(tokens)
                        }
                        for i, (prompt, tokens) in enumerate(zip(prompts[:len(collected_chunks)], collected_chunks))
                    ]
                    json.dump(output_data, f, indent=2)
                print(f"Complete outputs saved to {outputs_file}")
                
                # Process and save utilization results
                utilization_results = gpu_monitor.get_utilization_metrics(avg_token_count)
                utilization_results["token_counts"] = token_counts
                results.append(utilization_results)
                
                # Save detailed prefill utilization data to a separate JSON file
                detailed_prefill_file = os.path.join(output_dir, f"detailed_prefill_{avg_token_count}_tokens.json")
                with open(detailed_prefill_file, 'w') as f:
                    detailed_data = {
                        "token_length": avg_token_count,
                        "prefill_duration": utilization_results['prefill_duration'],
                        "detailed_data": utilization_results['detailed_prefill_data'],
                        "statistics": utilization_results['prefill_statistics']
                    }
                    json.dump(detailed_data, f, indent=2)
                print(f"Detailed prefill phase data saved to {detailed_prefill_file}")
                
                print(f"Results for {avg_token_count} tokens:")
                print(f"  Prefill GPU utilization: {utilization_results['prefill_utilization']:.2f}%")
                print(f"  Decode GPU utilization: {utilization_results['decode_utilization']:.2f}%")
                
                # Add this after processing
                prefill_duration = utilization_results['prefill_duration']
                decode_duration = utilization_results['decode_duration']
                print(f"Prefill duration: {prefill_duration:.4f}s")
                print(f"Decode duration: {decode_duration:.4f}s")
                print(f"Total generation time: {prefill_duration + decode_duration:.4f}s")
                
                # Wait between runs
                time.sleep(5)
                
            finally:
                gpu_monitor.stop_monitoring()
                
        # Save aggregated results
        save_and_plot_results(results, output_dir)
        
    finally:
        # Shut down the server
        shutdown_server(server_process)

def save_and_plot_results(results: List[Dict[str, Any]], output_dir: str):
    """
    Save and plot the aggregated results
    
    Args:
        results: List of result dictionaries from each run
        output_dir: Directory to save results
    """
    # Save raw results
    with open(os.path.join(output_dir, "gpu_utilization_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Import plotting functions from the utilities module
    from experiments.utiles.plot_prefill_decode_gpu_utilization import (
        plot_time_utilization, 
        plot_bar_gpu_utilization
    )
    
    # Create the time vs utilization plots for each batch
    for i, result in enumerate(results):
        print(f"Result {i}: token_length={result['token_length']}")
        print(f"  prefill_start={result.get('prefill_start')}")
        print(f"  prefill_end={result.get('prefill_end')}")
        print(f"  decode_end={result.get('decode_end')}")
    plot_time_utilization(results, output_dir)
    
    # Create the bar chart for all batches
    plot_bar_gpu_utilization(results, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure GPU utilization during prefill and decode phases")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    
    run_experiment(PROMPT_FILES, args.model, args.output_dir, batch_size=args.batch_size)
