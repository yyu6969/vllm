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
from load_prompts import load_prompts_from_csv, load_prompts_from_json
from plot_e2e_time import plot_e2e_time_chart_from_json

# --- 1. Define Parameters ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
METRICS_URL = f"{SERVER_URL}/metrics"
HEALTH_CHECK_URL = f"{SERVER_URL}/health"

# Your different sets of prompts with different token lengths
PROMPT_CONFIGS = [
    {
        "path": "/work/nvme/bdkz/yyu69/vllm/data/long-prompts_6000_6500.csv",
        "column_name": "prompt",
        "chunk_sizes": [32, 64, 128, 256, 512, 1024, 2048]
    },
]

GENERATION_PARAMS = {"max_tokens": 512, "temperature": 0.8, "top_p": 0.95}

# Exact names of the metrics you want to average
TARGET_METRIC_NAMES = [
    "vllm:e2e_request_latency_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
]

OUTPUT_DIR = "/work/nvme/bdkz/yyu69/vllm/experiment_results"

# Generate a timestamp for the filename - creating this at the start
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create main results directory right away
results_dir = f"{OUTPUT_DIR}/chunk_size_experiment_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Create a directory for detailed results
detailed_dir = os.path.join(results_dir, "detailed_results")
os.makedirs(detailed_dir, exist_ok=True)

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
        "--max-num-batched-tokens", str(chunk_size),
        "--max-num-seqs", str(8)
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

def send_single_request(prompt, model_name, generation_params):
    """
    Send a single request to the vLLM OpenAI API server and return the results.
    
    Args:
        prompt: The text prompt to send
        model_name: The model to use
        generation_params: Dictionary of generation parameters (max_tokens, temperature, etc.)
        
    Returns:
        dict: Results including success status and response text
    """
    print(f"Sending request: {prompt[:50]}..." if len(prompt) > 50 else f"Sending request: {prompt}")
    
    # Create OpenAI client
    client = openai.OpenAI(
        base_url=f"{SERVER_URL}/v1",
        api_key="EMPTY"
    )
    
    result = {
        "success": False,
        "response_text": None,
        "error": None
    }
    
    try:
        # Send the request
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            **generation_params
        )
        
        # Extract response
        result["success"] = True
        result["response_text"] = response.choices[0].text.strip()
        
        print(f"Request successful")
        print(f"Response: {result['response_text'][:50]}..." if len(result['response_text']) > 50 else f"Response: {result['response_text']}")
        
    except Exception as e:
        print(f"Request failed: {e}")
        result["error"] = str(e)
    
    return result

def count_tokens(text):
    """Count tokens in the given text using the model's tokenizer."""
    from transformers import AutoTokenizer
    
    # Initialize tokenizer only once for efficiency (using global variable)
    global tokenizer
    if 'tokenizer' not in globals():
        print("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    return len(tokenizer.encode(text))

# --- Main Execution Logic ---
try:
    # Initialize results dictionary
    final_results = {}
    
    # Process each prompt configuration
    for config in PROMPT_CONFIGS:
        # Load prompts from the specified path
        prompts = load_prompts_from_csv(config["path"], column_name=config["column_name"])
        
        # Calculate actual average token count
        if len(prompts) > 0:
            total_tokens = sum(count_tokens(prompt) for prompt in prompts)
            avg_tokens = int(total_tokens / len(prompts))
            prompt_set_key = f"avg_prompt_tokens_{avg_tokens}"
        else:
            prompt_set_key = "unknown_tokens"
        
        print(f"\n{'='*15} Testing prompt set {prompt_set_key} {'='*15}")
        
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
            server_process = start_vllm_server(MODEL_NAME, chunk_size)
            
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
            
            # Initialize results for this chunk size
            all_e2e_times = []
            all_prefill_times = []
            all_decode_times = []
            successful_requests = 0
            
            # For each prompt in the set, send individual request and measure metrics
            all_request_details = []  # Store detailed info for each request

            for i, prompt in enumerate(prompts):
                print(f"\nTesting prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
                
                # 5. Measure pre-request metrics
                pre_metrics = scrape_and_parse_metrics(TARGET_METRIC_NAMES)
                
                # Send request using the dedicated function
                request_result = send_single_request(prompt, MODEL_NAME, GENERATION_PARAMS)
                
                # Allow metrics to update
                time.sleep(2)
                
                # 6. Get current metrics and calculate times
                post_metrics = scrape_and_parse_metrics(TARGET_METRIC_NAMES)
                
                # Calculate metrics for this prompt
                prompt_metrics = {}
                request_detail = {
                    "prompt_index": i,
                    "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "prompt_length": len(prompt),
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
                        prompt_metrics[name] = avg_latency
                        request_detail["metrics"][name] = avg_latency
                        print(f"  {name}: {avg_latency:.6f}s")
                        
                        # Collect metrics for averaging later
                        if name == "vllm:e2e_request_latency_seconds":
                            all_e2e_times.append(avg_latency)
                        elif name == "vllm:request_prefill_time_seconds":
                            all_prefill_times.append(avg_latency)
                        elif name == "vllm:request_decode_time_seconds":
                            all_decode_times.append(avg_latency)
                    else:
                        prompt_metrics[name] = None
                        request_detail["metrics"][name] = None
                        print(f"  {name}: N/A (no count increase)")
                
                # Save each request's details
                all_request_details.append(request_detail)
                
                if request_result["success"]:
                    successful_requests += 1
            
            # 7. Shut down the server
            shutdown_server(server_process)
            server_process = None
            
            # 8. Calculate averages and store results in the JSON format needed for plot_e2e_time_chart_from_json
            # Format: {"avg_prompt_tokens_XXX": {"chunk_size_YYY": {"avg_e2e_time": Z.Z, ...}}}
            
            # Only add metrics if we had successful requests
            if successful_requests > 0:
                # Calculate average metrics
                avg_e2e_time = statistics.mean(all_e2e_times) if all_e2e_times else 0
                avg_prefill_time = statistics.mean(all_prefill_times) if all_prefill_times else 0
                avg_decode_time = statistics.mean(all_decode_times) if all_decode_times else 0
                
                final_results[prompt_set_key][f"chunk_size_{chunk_size}"] = {
                    "avg_e2e_time": avg_e2e_time,
                    "avg_prefill_time": avg_prefill_time,
                    "avg_decode_time": avg_decode_time
                }
            else:
                final_results[prompt_set_key][f"chunk_size_{chunk_size}"] = {
                    "avg_e2e_time": 0,
                    "avg_prefill_time": 0,
                    "avg_decode_time": 0,
                    "error": "No successful requests"
                }
            
            # After the loop, save the detailed results for this chunk size
            detailed_results = {
                "experiment_info": {
                    "prompt_set": prompt_set_key,
                    "chunk_size": chunk_size,
                    "model": MODEL_NAME,
                    "generation_params": GENERATION_PARAMS
                },
                "requests": all_request_details
            }

            # Use the existing detailed_dir
            detailed_json_path = os.path.join(detailed_dir, f"{prompt_set_key}_chunk{chunk_size}.json")
            try:
                with open(detailed_json_path, 'w') as f:
                    json.dump(detailed_results, f, indent=2)
                print(f"Detailed request data saved to: {detailed_json_path}")
            except Exception as e:
                print(f"Error saving detailed results to file: {e}")
            
            print(f"Finished testing chunk_size = {chunk_size} with {prompt_set_key}. Waiting before next run...")
            time.sleep(10)  # Wait before next iteration

except KeyboardInterrupt:
    print("\nExperiment interrupted by user.")
except ImportError:
    print("\nERROR: OpenAI Python package not found. Install with: pip install openai>=1.0.0")
finally:
    # Final cleanup
    if server_process and server_process.poll() is None:
        print("Performing final cleanup on exit...")
        shutdown_server(server_process)

# --- 4. Save Results ---
print(f"\n{'='*15} Final Experiment Results {'='*15}")

# Pretty print summarized results
print(json.dumps(final_results, indent=2))

# Save aggregated results in the format needed for plotting
results_json_path = os.path.join(results_dir, f"chunk_size_results_{timestamp}.json")
try:
    with open(results_json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Aggregated results saved to: {results_json_path}")
except Exception as e:
    print(f"Error saving results to file: {e}")

# Create plot from the results
try:
    plot_e2e_time_chart_from_json(results_json_path, results_dir, MODEL_NAME)
    print(f"Plot created in {results_dir}")
except Exception as e:
    print(f"Error creating plot: {e}")