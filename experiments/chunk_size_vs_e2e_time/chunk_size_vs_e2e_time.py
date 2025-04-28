import subprocess
import time
import requests
import re
import signal
import os
import statistics
import json
import openai
import threading
import queue
import pandas as pd
from datetime import datetime

import sys
sys.path.append('/work/nvme/bdkz/yyu69/vllm')
from experiments.utiles.load_prompts import load_prompts_from_csv, load_prompts_from_json
from experiments.utiles.plot_e2e_time import plot_e2e_time_chart_from_json, plot_ttft_time_chart_from_json, plot_tbt_time_chart_from_json

# --- 1. Define Parameters ---
MODEL_NAMES = ["Qwen/Qwen2.5-14B-Instruct"]
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
METRICS_URL = f"{SERVER_URL}/metrics"
HEALTH_CHECK_URL = f"{SERVER_URL}/health"

# Your different sets of prompts with different token lengths
PROMPT_CONFIGS = [
    # {
    #     "path": "/work/nvme/bdkz/yyu69/vllm/data/chunk_prefill/dataset_2/data_1000.csv",
    #     "column_name": "prompt",
    #     "chunk_sizes": [32, 64, 128, 256, 512, 1024]
    # },
    # {
    #     "path": "/work/nvme/bdkz/yyu69/vllm/data/chunk_prefill/dataset_2/data_2000.csv",
    #     "column_name": "prompt",
    #     "chunk_sizes": [32, 64, 128, 256, 512, 1024, 2048]
    # },
    # {
    #     "path": "/work/nvme/bdkz/yyu69/vllm/data/chunk_prefill/dataset_2/data_4000.csv",
    #     "column_name": "prompt",
    #     "chunk_sizes": [32, 64, 128, 256, 512, 1024, 2048, 4096]
    # },
    {
        "path": "/work/nvme/bdkz/yyu69/vllm/data/chunk_prefill/dataset_2/data_8000.csv",
        "column_name": "prompt",
        "chunk_sizes": [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    },
]

GENERATION_PARAMS = {
    "max_tokens": 4096,
    "temperature": 0.0,
    "top_p": 1.0
}

# Exact names of the metrics you want to average
TARGET_METRIC_NAMES = [
    "vllm:e2e_request_latency_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_inference_time_seconds",
]

OUTPUT_DIR = "/work/nvme/bdkz/yyu69/vllm/experiment_results/chunk_size_vs_e2e_time_experiments"

BATCH_SIZE = 1
NUM_RUNS_PER_BATCH = 5

# Generate a timestamp for the filename - creating this at the start
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create main results directory right away
results_dir = f"{OUTPUT_DIR}/chunk_size_vs_e2e_time_experiment_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# --- 2. Initialize Results Storage ---
server_process = None # Keep track of the server process

def log_reader(process, log_queue):
    """Read logs from process and put them in queue."""
    for line in iter(process.stdout.readline, ''):
        log_queue.put(line.strip())
    log_queue.put(None)  # Signal end of stream

def start_vllm_server(model_name, chunk_size):
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--max-num-batched-tokens", str(chunk_size),
        "--max-num-seqs", str(16)
    ]
    print(f"Starting server with command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, preexec_fn=os.setsid)
    return process

def wait_for_server_ready(process, readiness_url, timeout=300):
    """Wait for the server to be ready using both logs and HTTP polling."""
    print(f"Waiting for server to be ready at {readiness_url}...")
    start_time = time.time()
    log_queue = queue.Queue()
    
    # Start log reader thread
    log_thread = threading.Thread(target=log_reader, args=(process, log_queue))
    log_thread.daemon = True
    log_thread.start()
    
    # Check for both log indicator and HTTP readiness
    log_ready_indicators = [
        "Uvicorn running on",
        "Starting vLLM API server on",
        "Application startup complete"
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
                    if indicator in line:
                        print(f"Server ready indicator found in logs: '{indicator}'")
                        # Found indicator in logs, but give some extra time for server to be fully ready
                        time.sleep(3)
                        server_ready = True
        except queue.Empty:
            pass
        
        # Check HTTP endpoint
        try:
            response = requests.get(readiness_url, timeout=5)
            if response.status_code == 200:
                print(f"Server responded OK at {readiness_url}.")
                return True
        except requests.ConnectionError:
            if server_ready:
                print(f"Server logs indicated ready, but HTTP endpoint not responding yet. Retrying...")
            pass
        except requests.Timeout:
            print(f"Timeout polling {readiness_url}, retrying...")
        except Exception as e:
            print(f"Error polling {readiness_url}: {e}")
        
        time.sleep(3)

    print("Server failed to become ready within timeout.")
    return False

def scrape_and_parse_metrics(metric_names):
    print("Scraping metrics...")
    metrics_data = {}
    try:
        response = requests.get(METRICS_URL, timeout=10)
        response.raise_for_status()
        metrics_text = response.text

        for name in metric_names:
            # Regex to find sum and count, ignoring labels for simplicity here
            # It captures the numeric value after the whitespace
            sum_match = re.search(rf"^{re.escape(name)}_sum(?:{{.*?}})?\s+([\d\.eE+-]+)", metrics_text, re.MULTILINE)
            count_match = re.search(rf"^{re.escape(name)}_count(?:{{.*?}})?\s+([\d\.eE+-]+)", metrics_text, re.MULTILINE)

            if sum_match and count_match:
                metrics_data[name] = {
                    "sum": float(sum_match.group(1)),
                    "count": float(count_match.group(1))
                }
            else:
                 print(f"Warning: Metric {name} (or sum/count) not found in output.")
                 metrics_data[name] = {"sum": 0.0, "count": 0.0} # Default

    except requests.exceptions.RequestException as e:
        print(f"Failed to scrape metrics: {e}")
        for name in metric_names:
             metrics_data[name] = {"sum": 0.0, "count": 0.0}
    return metrics_data

def shutdown_server(process):
     if process and process.poll() is None: # Check if process exists and is running
        print("Shutting down server...")
        pgid = -1
        try:
            # Get the process group ID
            pgid = os.getpgid(process.pid)
            print(f"Sending SIGINT to process group {pgid} (process {process.pid})...")
            os.killpg(pgid, signal.SIGINT)
            process.wait(timeout=30) # Wait for graceful shutdown
            print("Server shut down via SIGINT.")
            return # Success
        except ProcessLookupError:
             print("Server process group already gone.")
             if process.poll() is None: process.wait(timeout=5) # Try waiting for main process just in case
             return
        except subprocess.TimeoutExpired:
            print("Server did not shut down gracefully via SIGINT, sending SIGTERM...")
        except Exception as e:
             print(f"Error sending SIGINT or waiting: {e}")

        # If SIGINT failed or timed out, try SIGTERM
        try:
             if pgid != -1 and process.poll() is None: # Check process is still running
                 print(f"Sending SIGTERM to process group {pgid}...")
                 os.killpg(pgid, signal.SIGTERM)
                 process.wait(timeout=15)
                 print("Server shut down via SIGTERM.")
                 return # Success
        except ProcessLookupError:
             print("Server process group already gone.")
             if process.poll() is None: process.wait(timeout=5)
             return
        except subprocess.TimeoutExpired:
             print("Server did not respond to SIGTERM, sending SIGKILL...")
        except Exception as e:
             print(f"Error sending SIGTERM or waiting: {e}")

        # If SIGTERM failed or timed out, try SIGKILL
        try:
            if pgid != -1 and process.poll() is None: # Check process is still running
                 print(f"Sending SIGKILL to process group {pgid}...")
                 os.killpg(pgid, signal.SIGKILL)
                 process.wait(timeout=5)
                 print("Server shut down via SIGKILL.")
            elif process.poll() is None: # If pgid failed, try killing main pid directly (last resort)
                 print(f"Sending SIGKILL to process {process.pid}...")
                 process.kill()
                 process.wait(timeout=5)
                 print("Server shut down via SIGKILL (PID).")

        except ProcessLookupError:
             print("Server process group/PID already gone.")
        except Exception as e:
             print(f"Failed during SIGKILL attempt: {e}")

        if process.poll() is None:
             print("WARNING: Server process may still be running!")

def send_single_request(prompts, model_name, generation_params):
    """
    Send a single batch request to the vLLM OpenAI API server.
    
    Args:
        prompts: List of text prompts to send (for batch processing)
        model_name: The model to use
        generation_params: Dictionary of generation parameters
        
    Returns:
        dict: Results including success status, response info, and output token counts
    """
    print(f"Sending batch request with {len(prompts)} prompts")
    
    # Create OpenAI client
    client = openai.OpenAI(
        base_url=f"{SERVER_URL}/v1",
        api_key="EMPTY"
    )
    
    result = {
        "success": False,
        "response_text": None,
        "error": None,
        "output_token_counts": []  # To store counts of generated tokens
    }
    
    try:
        # Add min_tokens as extra_body parameter
        extra_body = {"min_tokens": 4096}
        
        if isinstance(prompts, list) and len(prompts) > 1:
            response = client.completions.create(
                model=model_name,
                prompt=prompts,
                extra_body=extra_body,
                **generation_params
            )
        else:
            response = client.completions.create(
                model=model_name,
                prompt=prompts[0] if isinstance(prompts, list) else prompts,
                extra_body=extra_body,
                **generation_params
            )
        
        # Extract response
        result["success"] = True
        
        # Store response text(s) and count tokens
        responses = []
        
        if hasattr(response.choices[0], 'text'):
            # For single completions
            text = response.choices[0].text.strip()
            result["response_text"] = text
            # Count tokens in the output
            output_tokens = count_tokens(model_name, text)
            result["output_token_counts"] = [output_tokens]
            
            # Print the prompt and response
            print(f"\nPROMPT: {prompts[0][:200]}..." if len(prompts[0]) > 200 else f"\nPROMPT: {prompts[0]}")
            print(f"RESPONSE: {text[:200]}..." if len(text) > 200 else f"RESPONSE: {text}")
            print(f"Output tokens: {output_tokens}")
            
        else:
            # For multiple completions in a batch
            responses = [choice.text.strip() for choice in response.choices]
            result["response_text"] = responses
            
            # Count tokens for each response
            for text in responses:
                output_tokens = count_tokens(model_name, text)
                result["output_token_counts"].append(output_tokens)
            
            # Print all prompts and responses in the batch
            print("\nAll prompts and responses in batch:")
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                print(f"\n--- Pair {i+1}/{len(prompts)} ---")
                print(f"PROMPT: {prompt[:200]}..." if len(prompt) > 200 else f"PROMPT: {prompt}")
                print(f"RESPONSE: {response[:200]}..." if len(response) > 200 else f"RESPONSE: {response}")
                print(f"Output tokens: {result['output_token_counts'][i]}")
            
            # Print total tokens
            total_output_tokens = sum(result["output_token_counts"])
            print(f"\nTotal output tokens across all responses: {total_output_tokens}")
        
        print(f"Request successful")
        
    except Exception as e:
        print(f"Request failed: {e}")
        result["error"] = str(e)
    
    return result

def count_tokens(model_name, text):
    """Count tokens in the given text using the model's tokenizer."""
    from transformers import AutoTokenizer
    
    # Initialize tokenizer only once for efficiency (using global variable)
    global tokenizer
    if 'tokenizer' not in globals():
        print("Initializing tokenizer...")
        # Use a default model for tokenization - doesn't need to be precise for our measurements
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return len(tokenizer.encode(text))

# --- Main Execution Logic ---
try:
    # Process each model
    for model_name in MODEL_NAMES:
        print(f"\n{'='*15} Testing model {model_name} {'='*15}")
        
        # Extract just the model name without organization prefix for directory naming
        model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
        
        # Create model-specific directory using short name
        model_dir = os.path.join(results_dir, model_short_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize results dictionary for this model
        final_results = {}
        
        # Process each prompt configuration
        for config in PROMPT_CONFIGS:
            # Load prompts from the specified path
            all_prompts = load_prompts_from_csv(config["path"], column_name=config["column_name"])
            # all_prompts = load_prompts_from_json(config["path"])

            # Make sure we have enough unique prompts for all runs
            required_prompts = BATCH_SIZE * NUM_RUNS_PER_BATCH + BATCH_SIZE # +BATCH_SIZE for warm-up
            if len(all_prompts) < required_prompts:
                print(f"WARNING: Not enough prompts for {NUM_RUNS_PER_BATCH} runs with batch size {BATCH_SIZE}.")
                print(f"Need {required_prompts} prompts, but only have {len(all_prompts)}.")
                
                # Handle special case when we have just one prompt
                if len(all_prompts) == 1:
                    print(f"Only one prompt found. Using the same prompt for all {NUM_RUNS_PER_BATCH} runs.")
                    single_prompt = all_prompts[0]
                    all_prompts = [single_prompt] * required_prompts
                else:
                    print("Will repeat prompts which may cause caching effects.")
                    # Extend the prompts list by repeating if needed
                    while len(all_prompts) < required_prompts:
                        all_prompts.extend(all_prompts[:required_prompts - len(all_prompts)])
            
            # Calculate actual average token count for all batches
            if len(all_prompts) > 0:
                total_tokens = sum(count_tokens(model_name, prompt) for prompt in all_prompts[:required_prompts])
                avg_tokens = int(total_tokens / required_prompts)
                prompt_set_key = f"avg_prompt_tokens_{avg_tokens}"
            else:
                prompt_set_key = "unknown_tokens"
            
            print(f"\n{'='*15} Testing prompt set {prompt_set_key} {'='*15}")
            
            # Create a directory for this prompt set 
            prompt_set_dir = os.path.join(model_dir, prompt_set_key)
            os.makedirs(prompt_set_dir, exist_ok=True)
            
            # Initialize results for this prompt set
            if prompt_set_key not in final_results:
                final_results[prompt_set_key] = {}
            
            # Test each chunk size specified for this prompt set
            for chunk_size in config["chunk_sizes"]:
                print(f"\n{'-'*15} Testing chunk_size = {chunk_size} with {prompt_set_key} {'-'*15}")
                
                # 1. Check if any server running
                if server_process and server_process.poll() is None:
                    print("Existing server found. Shutting it down first...")
                    shutdown_server(server_process)
                    server_process = None
                    time.sleep(5)  # Give time for resources to be released
                
                # 2. Start the server with chunk_size
                server_process = start_vllm_server(model_name, chunk_size)
                
                # 3. Check if server started and became ready
                if not wait_for_server_ready(server_process, HEALTH_CHECK_URL):
                    print(f"ERROR: Server failed to start or become ready for chunk size {chunk_size}.")
                    shutdown_server(server_process)
                    server_process = None
                    final_results[prompt_set_key][f"chunk_size_{chunk_size}"] = {'status': 'failed_startup'}
                    time.sleep(5)
                    continue  # Skip to next chunk size
                
                # 4. Check if metrics are all zero (initial state)
                initial_metrics = scrape_and_parse_metrics(TARGET_METRIC_NAMES)
                metrics_zeroed = True
                for name in TARGET_METRIC_NAMES:
                    if initial_metrics.get(name, {}).get("sum", 0.0) > 0 or initial_metrics.get(name, {}).get("count", 0.0) > 0:
                        metrics_zeroed = False
                        break
                
                if not metrics_zeroed:
                    print("WARNING: Initial metrics are not zero. Server may have residual data.")
                
                # Add single warm-up run before starting measurements
                print(f"\n{'-'*10} Performing GPU warm-up {'-'*10}")
                # Use the first batch of prompts for warm-up
                warmup_prompts = all_prompts[:BATCH_SIZE]
                print(f"Using first {len(warmup_prompts)} prompts for warm-up")

                # Single warm-up run to initialize GPU and compile any needed kernels
                print("Executing warm-up run...")
                warm_up_result = send_single_request(warmup_prompts, model_name, GENERATION_PARAMS)
                if not warm_up_result["success"]:
                    print(f"WARNING: Warm-up request failed: {warm_up_result.get('error', 'Unknown error')}")
                else:
                    print("Warm-up request completed successfully")

                # Allow system to stabilize after warm-up
                print("Waiting for system to stabilize...")
                time.sleep(5)  

                # Get clean baseline metrics AFTER warm-up
                print("Getting clean baseline metrics after warm-up...")
                baseline_metrics = scrape_and_parse_metrics(TARGET_METRIC_NAMES)

                print(f"{'-'*10} Warm-up complete, starting actual measurements {'-'*10}")

                # Adjust the starting index for actual runs to skip warm-up prompts
                # Start using prompts after the first BATCH_SIZE used for warm-up
                warmup_offset = 0

                # Initialize results for this chunk size
                all_e2e_times = []
                all_prefill_times = []
                all_decode_times = []
                all_inference_times = []
                all_tbts = []
                successful_requests = 0
                
                # Store detailed info for each request
                all_request_details = []
                
                # Use baseline_metrics as pre_metrics for the first run
                prev_metrics = baseline_metrics

                for run_idx in range(NUM_RUNS_PER_BATCH):
                    # Calculate indices with offset to skip warm-up prompts
                    start_idx = warmup_offset + (run_idx * BATCH_SIZE)
                    end_idx = start_idx + BATCH_SIZE
                    batch_prompts = all_prompts[start_idx:end_idx]
                    
                    print(f"\n{'-'*10} Run {run_idx+1}/{NUM_RUNS_PER_BATCH} with batch of {len(batch_prompts)} NEW prompts {'-'*10}")
                    
                    # Use previous metrics as pre-request metrics
                    pre_metrics = prev_metrics
                    
                    # Send request using the dedicated function
                    request_result = send_single_request(batch_prompts, model_name, GENERATION_PARAMS)
                    
                    # Allow metrics to update
                    time.sleep(2)
                    
                    # 6. Get current metrics and calculate times
                    post_metrics = scrape_and_parse_metrics(TARGET_METRIC_NAMES)
                    
                    # Save current metrics to use as pre-metrics for next run
                    prev_metrics = post_metrics
                    
                    # Calculate metrics for this batch
                    batch_metrics = {}
                    
                    # Create a batch detail record
                    batch_detail = {
                        "run_index": run_idx,
                        "batch_size": len(batch_prompts),
                        "success": request_result["success"],
                        "metrics": {}
                    }
                    
                    for name in TARGET_METRIC_NAMES:
                        pre_sum = pre_metrics.get(name, {}).get("sum", 0.0)
                        pre_count = pre_metrics.get(name, {}).get("count", 0.0)
                        post_sum = post_metrics.get(name, {}).get("sum", 0.0)
                        post_count = post_metrics.get(name, {}).get("count", 0.0)
                        
                        # Calculate delta
                        delta_sum = post_sum - pre_sum
                        delta_count = post_count - pre_count
                        
                        if delta_count > 0:
                            avg_latency = delta_sum / delta_count
                            batch_metrics[name] = avg_latency
                            batch_detail["metrics"][name] = avg_latency
                            print(f"  {name}: {avg_latency:.6f}s")
                            
                            # Collect metrics for averaging later
                            if name == "vllm:e2e_request_latency_seconds":
                                all_e2e_times.append(avg_latency)
                            elif name == "vllm:request_prefill_time_seconds":
                                all_prefill_times.append(avg_latency)
                            elif name == "vllm:request_decode_time_seconds":
                                all_decode_times.append(avg_latency)
                            elif name == "vllm:request_inference_time_seconds":
                                all_inference_times.append(avg_latency)
                        else:
                            batch_metrics[name] = None
                            batch_detail["metrics"][name] = None
                            print(f"  {name}: N/A (no count increase)")
                    
                    # Calculate Time Between Tokens (TBT) using vllm:generation_tokens_total
                    if "vllm:request_decode_time_seconds" in batch_metrics and batch_metrics["vllm:request_decode_time_seconds"] is not None:
                        # Get tokens directly from your API response
                        output_tokens = request_result.get("output_token_counts", [0])[0]
                        
                        if output_tokens > 0:
                            time_between_tokens = batch_metrics["vllm:request_decode_time_seconds"] / output_tokens
                            batch_metrics["time_between_tokens"] = time_between_tokens
                            batch_detail["metrics"]["time_between_tokens"] = time_between_tokens
                            print(f"  time_between_tokens: {time_between_tokens:.6f}s (based on {output_tokens} tokens from response)")
                            all_tbts.append(time_between_tokens)
                        else:
                            print("  time_between_tokens: N/A (no generated tokens according to response)")
                            batch_metrics["time_between_tokens"] = None
                            batch_detail["metrics"]["time_between_tokens"] = None
                    else:
                        print("  time_between_tokens: N/A (decode time not available)")
                        batch_metrics["time_between_tokens"] = None
                        batch_detail["metrics"]["time_between_tokens"] = None
                    
                    # Save batch details
                    all_request_details.append(batch_detail)
                    
                    if request_result["success"]:
                        successful_requests += 1
                
                # 7. Shut down the server
                shutdown_server(server_process)
                server_process = None
                
                # 8. Calculate averages and store results in the JSON format needed for plot_e2e_time_chart_from_json
                # Only add metrics if we had successful requests
                if successful_requests > 0:
                    # Calculate average metrics
                    avg_e2e_time = statistics.mean(all_e2e_times) if all_e2e_times else 0
                    avg_prefill_time = statistics.mean(all_prefill_times) if all_prefill_times else 0
                    avg_decode_time = statistics.mean(all_decode_times) if all_decode_times else 0
                    avg_inference_time = statistics.mean(all_inference_times) if all_inference_times else 0
                    avg_tbt = statistics.mean(all_tbts) if all_tbts else 0
                    
                    final_results[prompt_set_key][f"chunk_size_{chunk_size}"] = {
                        "avg_e2e_time": avg_e2e_time,
                        "avg_prefill_time": avg_prefill_time,
                        "avg_decode_time": avg_decode_time,
                        "avg_inference_time": avg_inference_time,
                        "avg_time_between_tokens": avg_tbt
                    }
                else:
                    final_results[prompt_set_key][f"chunk_size_{chunk_size}"] = {
                        "avg_e2e_time": 0,
                        "avg_prefill_time": 0,
                        "avg_decode_time": 0,
                        "avg_inference_time": 0,
                        "avg_time_between_tokens": 0,
                        "error": "No successful requests"
                    }
                
                # After the loop, save the detailed results for this chunk size
                detailed_results = {
                    "experiment_info": {
                        "prompt_set": prompt_set_key,
                        "chunk_size": chunk_size,
                        "model": model_name,
                        "generation_params": GENERATION_PARAMS
                    },
                    "requests": all_request_details
                }

                # Save to prompt-specific directory with a cleaner filename
                chunk_json_filename = f"chunk_size_{chunk_size}.json"
                detailed_json_path = os.path.join(prompt_set_dir, chunk_json_filename)
                try:
                    with open(detailed_json_path, 'w') as f:
                        json.dump(detailed_results, f, indent=2)
                    print(f"Detailed request data saved to: {detailed_json_path}")
                except Exception as e:
                    print(f"Error saving detailed results to file: {e}")
                
                print(f"Finished testing chunk_size = {chunk_size} with {prompt_set_key}. Waiting before next run...")
                time.sleep(10)  # Wait before next iteration

            # Save individual results JSON for this prompt set right after processing it
            # Create a single-prompt results dictionary
            prompt_results = {prompt_set_key: final_results[prompt_set_key]}
            
            # Save prompt-specific results JSON
            prompt_json_path = os.path.join(prompt_set_dir, f"chunk_size_results.json")
            
            try:
                with open(prompt_json_path, 'w') as f:
                    json.dump(prompt_results, f, indent=2)
                print(f"Prompt-specific results saved to: {prompt_json_path}")
            except Exception as e:
                print(f"Error saving prompt-specific results for {prompt_set_key}: {e}")
            
            print(f"Finished processing prompt set {prompt_set_key}.")

        # Save and plot results for this model
        # Pretty print summarized results
        print(f"\n{'='*15} Final Results for {model_name} {'='*15}")
        print(json.dumps(final_results, indent=2))

        # Save aggregated results in the format needed for plotting
        results_json_path = os.path.join(model_dir, f"chunk_size_results_{model_short_name}.json")
        try:
            with open(results_json_path, 'w') as f:
                json.dump(final_results, f, indent=2)
            print(f"Aggregated results saved to: {results_json_path}")
        except Exception as e:
            print(f"Error saving results to file: {e}")

        # Create plot from the results
        try:
            # Plot end-to-end time metrics
            plot_e2e_time_chart_from_json(results_json_path, model_dir, model_short_name)
            
            # Also plot TTFT metrics
            plot_ttft_time_chart_from_json(results_json_path, model_dir, model_short_name)
            
            # And plot TBT metrics if they exist
            plot_tbt_time_chart_from_json(results_json_path, model_dir, model_short_name)
            
            print(f"Plots created in {model_dir}")
        except Exception as e:
            print(f"Error creating plots: {e}")

except KeyboardInterrupt:
    print("\nExperiment interrupted by user.")
except ImportError:
    print("\nERROR: OpenAI Python package not found. Install with: pip install openai>=1.0.0")
finally:
    # Final cleanup
    if server_process and server_process.poll() is None:
        print("Performing final cleanup on exit...")
        shutdown_server(server_process)

# --- Print final experiment summary ---
print(f"\n{'='*15} Experiment Complete {'='*15}")
print(f"Results saved to: {results_dir}")
print(f"Models tested: {', '.join(MODEL_NAMES)}")