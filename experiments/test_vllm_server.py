import argparse
import json
import time
import threading
import os
import re
import requests
import subprocess
import signal
import openai
import sys
import queue
import wandb

# Add parent directory to Python path for importing utilities
sys.path.append('/work/nvme/bdkz/yyu69/vllm')
from experiments.utiles.load_prompts import load_prompts_from_csv, load_prompts_from_json

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
HEALTH_CHECK_URL = f"{SERVER_URL}/health"
OUTPUT_DIR = "/work/nvme/bdkz/yyu69/vllm/experiment_results/prefill_decode_test"
WANDB_PROJECT = "vllm-server-test"
WANDB_GROUP = "prefill-decode-test"

# CSV prompt files
PROMPT_FILES = [
    # "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/long-prompts-selection-1250-1375.csv",
    # "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/long-prompts-selection-2500-2750.csv",
    # "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/long-prompts-selection-5000-5500.csv",
    # "/work/nvme/bdkz/yyu69/vllm/data/long-prompts-6000_6500_100.csv",
    "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/select-text-by-length_5000_5500.csv"
]

BATCH_SIZE = 8  # Fixed batch size per your requirement
MAX_GEN_TOKENS = 64  # Small number to keep experiment focused

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
    avg_token_count = int(sum(token_counts) / len(token_counts)) if token_counts else 0
    
    print(f"Loaded {len(prompts)} prompts from {file_path}")
    print(f"Average token count for all prompts: {avg_token_count}")
    
    # Return all prompts, not just the first batch
    return prompts, token_counts, avg_token_count

# --- Main Experiment ---
def run_experiment(prompt_files, model_name, output_dir, batch_size=8, 
                   use_wandb=True, wandb_project=WANDB_PROJECT, wandb_group=WANDB_GROUP,
                   enable_chunked_prefill=False):
    """
    Run the experiment with text generation
    
    Args:
        prompt_files: List of CSV files with prompts
        model_name: Model to use
        output_dir: Directory to save results
        batch_size: Number of prompts to use in each batch
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: W&B project name
        wandb_group: W&B group name
        enable_chunked_prefill: Whether to enable chunked prefill
    """
    # Set environment variable to use V0 engine
    os.environ["VLLM_USE_V1"] = "0"
    
    # Create timestamped output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb
    if use_wandb:
        run_name = f"{model_name.split('/')[-1]}_{'with' if enable_chunked_prefill else 'without'}_chunked_prefill_{timestamp}"
        wandb.init(
            project=wandb_project,
            group=wandb_group,
            name=run_name,
            config={
                "model": model_name,
                "output_dir": output_dir,
                "batch_size": batch_size,
                "max_gen_tokens": MAX_GEN_TOKENS,
                "enable_chunked_prefill": enable_chunked_prefill,
                "experiment_timestamp": timestamp,
                "prompt_files": prompt_files,
            }
        )
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Start the vLLM server with specified chunked prefill setting
    server_process, log_queue = start_vllm_server(model_name, no_chunked_prefill=(not enable_chunked_prefill))
    
    try:
        # Wait for server to be ready
        if not wait_for_server_ready(server_process, log_queue):
            print("Failed to start server. Exiting.")
            if use_wandb:
                wandb.log({"server_startup": "failed"})
                wandb.finish()
            return

        if use_wandb:
            wandb.log({"server_startup": "successful"})

        time.sleep(5)  # Additional wait time to ensure everything is initialized
        
        # Initialize OpenAI client
        client = openai.OpenAI(
            base_url=f"{SERVER_URL}/v1",
            api_key="EMPTY"
        )
        
        # Track aggregate metrics
        all_csv_metrics = {
            "avg_token_counts": [],
            "avg_first_token_times": [],
            "avg_generation_times": [],
            "avg_tokens_per_second": []
        }
        
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
        batch_prompt = ["Hello there"] * batch_size
        warmup_response = client.completions.create(
            model=model_name,
            prompt=batch_prompt,
            max_tokens=8
        )

        print("Warm-up generations completed.")
        time.sleep(5)  # Longer wait to ensure everything is settled
        
        # Test each CSV file (which corresponds to a different token length range)
        for csv_idx, csv_file in enumerate(prompt_files):
            print(f"\n{'='*20}\nTesting prompts from: {csv_file}\n{'='*20}")
            
            # Extract CSV file basename for logging
            csv_basename = os.path.basename(csv_file)
            
            # Load prompts and calculate token counts
            all_prompts, all_token_counts, avg_token_count = load_prompts_and_calculate_tokens(csv_file, tokenizer)
            
            if not all_prompts:
                print(f"No prompts loaded from {csv_file}. Skipping.")
                continue
            
            # Log stats about the prompts
            if use_wandb:
                wandb.log({
                    f"csv_{csv_idx}/file_name": csv_basename,
                    f"csv_{csv_idx}/num_prompts": len(all_prompts),
                    f"csv_{csv_idx}/avg_token_count": avg_token_count,
                    f"csv_{csv_idx}/min_token_count": min(all_token_counts),
                    f"csv_{csv_idx}/max_token_count": max(all_token_counts),
                })
            
            csv_metrics = {
                "first_token_times": [],
                "generation_times": [],
                "prompt_tokens": [],
                "completion_tokens": [],
                "tokens_per_second": []
            }
            
            # Process prompts in batches
            num_batches = (len(all_prompts) + batch_size - 1) // batch_size  # Ceiling division
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(all_prompts))
                
                # Get current batch
                current_batch_prompts = all_prompts[batch_start:batch_end]
                current_batch_token_counts = all_token_counts[batch_start:batch_end]
                
                # Calculate average token count for this specific batch
                batch_avg_token_count = int(sum(current_batch_token_counts) / len(current_batch_token_counts)) if current_batch_token_counts else 0
                
                print(f"\nProcessing batch {batch_idx+1}/{num_batches} (prompts {batch_start+1}-{batch_end})")
                print(f"Batch average token count: {batch_avg_token_count}")
                
                # Start timing just before you send the request
                print(f"Sending batch request with {len(current_batch_prompts)} prompts")

                # Send request with streaming to detect first token
                start_time = time.time()
                first_token_time = None
                first_token_received = False
                stream = client.completions.create(
                    model=model_name,
                    prompt=current_batch_prompts,
                    max_tokens=MAX_GEN_TOKENS,
                    temperature=0.8,
                    top_p=0.95,
                    stream=True
                )

                # Collect the response
                full_response = []
                collected_chunks = [[] for _ in range(len(current_batch_prompts))]
                for chunk in stream:
                    if not first_token_received:
                        # First token received
                        first_token_received = True
                        first_token_time = time.time() - start_time
                        print(f"First token received at {first_token_time:.4f}s")
                    
                    # Collect the chunk
                    full_response.append(chunk)
                    
                    # Store the tokens by prompt index if available
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        if hasattr(choice, 'index') and hasattr(choice, 'text'):
                            prompt_idx = choice.index
                            if prompt_idx < len(collected_chunks):
                                collected_chunks[prompt_idx].append(choice.text)

                # Wait for generation to complete
                end_time = time.time()
                generation_time = end_time - start_time
                print(f"Generation completed in {generation_time:.2f} seconds")
                
                # Calculate metrics
                if first_token_time:
                    csv_metrics["first_token_times"].append(first_token_time)
                
                csv_metrics["generation_times"].append(generation_time)
                csv_metrics["prompt_tokens"].extend(current_batch_token_counts)
                
                # Calculate completion tokens and tokens per second
                batch_completion_tokens = []
                for tokens in collected_chunks:
                    output_text = ''.join(tokens)
                    token_count = calculate_token_count(output_text, tokenizer)
                    batch_completion_tokens.append(token_count)
                
                csv_metrics["completion_tokens"].extend(batch_completion_tokens)
                
                # Calculate tokens per second
                for prompt_tokens, completion_tokens in zip(current_batch_token_counts, batch_completion_tokens):
                    total_tokens = prompt_tokens + completion_tokens
                    tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
                    csv_metrics["tokens_per_second"].append(tokens_per_second)
                
                # Log batch metrics
                if use_wandb:
                    wandb.log({
                        f"csv_{csv_idx}/batch_{batch_idx}/avg_prompt_tokens": batch_avg_token_count,
                        f"csv_{csv_idx}/batch_{batch_idx}/first_token_time": first_token_time if first_token_time else 0,
                        f"csv_{csv_idx}/batch_{batch_idx}/generation_time": generation_time,
                        f"csv_{csv_idx}/batch_{batch_idx}/avg_completion_tokens": sum(batch_completion_tokens) / len(batch_completion_tokens) if batch_completion_tokens else 0,
                        f"csv_{csv_idx}/batch_{batch_idx}/avg_tokens_per_second": sum(tokens_per_second for tokens_per_second in csv_metrics["tokens_per_second"][-len(current_batch_prompts):]) / len(current_batch_prompts) if current_batch_prompts else 0
                    })
                
                # Print outputs for each prompt in this batch
                print("\nGenerated Outputs:\n" + "-" * 60)
                for i, (prompt, tokens) in enumerate(zip(current_batch_prompts[:len(collected_chunks)], collected_chunks)):
                    output_text = ''.join(tokens)
                    
                    # Print truncated prompt (first 50 chars) to avoid too much output
                    truncated_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
                    print(f"Prompt {batch_start+i+1}: {truncated_prompt}")
                    
                    # Print truncated output (first 100 chars) to avoid too much output
                    truncated_output = output_text[:100] + "..." if len(output_text) > 100 else output_text
                    print(f"Generated: {truncated_output}")
                    print("-" * 60)

                # Save the complete outputs for this batch to a file
                outputs_file = os.path.join(output_dir, f"outputs_batch_{batch_idx+1}_of_{num_batches}.json")
                with open(outputs_file, 'w') as f:
                    output_data = [
                        {
                            "global_prompt_idx": batch_start + i,
                            "batch_prompt_idx": i,
                            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,  # Truncate for readability
                            "output": ''.join(tokens),
                            "prompt_tokens": current_batch_token_counts[i] if i < len(current_batch_token_counts) else 0,
                            "completion_tokens": batch_completion_tokens[i] if i < len(batch_completion_tokens) else 0,
                            "first_token_time": first_token_time,
                            "generation_time": generation_time
                        }
                        for i, (prompt, tokens) in enumerate(zip(current_batch_prompts[:len(collected_chunks)], collected_chunks))
                    ]
                    json.dump(output_data, f, indent=2)
                print(f"Complete outputs for batch {batch_idx+1} saved to {outputs_file}")
                
                # Wait between batches
                if batch_idx < num_batches - 1:  # Don't wait after the last batch
                    print("Waiting for next batch...")
                    time.sleep(5)
            
            # Calculate and log CSV-level metrics
            if csv_metrics["first_token_times"]:
                avg_first_token_time = sum(csv_metrics["first_token_times"]) / len(csv_metrics["first_token_times"])
                all_csv_metrics["avg_first_token_times"].append(avg_first_token_time)
                if use_wandb:
                    wandb.log({f"csv_{csv_idx}/avg_first_token_time": avg_first_token_time})
            
            if csv_metrics["generation_times"]:
                avg_generation_time = sum(csv_metrics["generation_times"]) / len(csv_metrics["generation_times"])
                all_csv_metrics["avg_generation_times"].append(avg_generation_time)
                if use_wandb:
                    wandb.log({f"csv_{csv_idx}/avg_generation_time": avg_generation_time})
            
            if csv_metrics["tokens_per_second"]:
                avg_tokens_per_second = sum(csv_metrics["tokens_per_second"]) / len(csv_metrics["tokens_per_second"])
                all_csv_metrics["avg_tokens_per_second"].append(avg_tokens_per_second)
                if use_wandb:
                    wandb.log({f"csv_{csv_idx}/avg_tokens_per_second": avg_tokens_per_second})
            
            # Save these metrics to a file
            metrics_file = os.path.join(output_dir, f"metrics_{csv_basename}.json")
            with open(metrics_file, "w") as f:
                json.dump({
                    "csv_file": csv_file,
                    "num_prompts": len(all_prompts),
                    "avg_token_count": avg_token_count,
                    "min_token_count": min(all_token_counts),
                    "max_token_count": max(all_token_counts),
                    "first_token_times": csv_metrics["first_token_times"],
                    "avg_first_token_time": avg_first_token_time if csv_metrics["first_token_times"] else None,
                    "generation_times": csv_metrics["generation_times"],
                    "avg_generation_time": avg_generation_time if csv_metrics["generation_times"] else None,
                    "avg_tokens_per_second": avg_tokens_per_second if csv_metrics["tokens_per_second"] else None
                }, f, indent=2)
        
        # Log overall experiment metrics
        if use_wandb:
            if all_csv_metrics["avg_first_token_times"]:
                wandb.log({"overall/avg_first_token_time": sum(all_csv_metrics["avg_first_token_times"]) / len(all_csv_metrics["avg_first_token_times"])})
            
            if all_csv_metrics["avg_generation_times"]:
                wandb.log({"overall/avg_generation_time": sum(all_csv_metrics["avg_generation_times"]) / len(all_csv_metrics["avg_generation_times"])})
            
            if all_csv_metrics["avg_tokens_per_second"]:
                wandb.log({"overall/avg_tokens_per_second": sum(all_csv_metrics["avg_tokens_per_second"]) / len(all_csv_metrics["avg_tokens_per_second"])})
                
    finally:
        # Shut down the server
        shutdown_server(server_process)
        
        # Close wandb
        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vLLM server with various prompts")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--enable-chunked-prefill", action="store_true", help="Enable chunked prefill")
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT, help="W&B project name")
    parser.add_argument("--wandb-group", type=str, default=WANDB_GROUP, help="W&B group name")
    args = parser.parse_args()
    
    run_experiment(
        PROMPT_FILES, 
        args.model, 
        args.output_dir, 
        batch_size=args.batch_size,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        enable_chunked_prefill=args.enable_chunked_prefill
    )
