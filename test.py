# test.py for testing server startup and OpenAI client
import openai
import time
import os
import signal
import subprocess
import requests
import re
import threading
import queue

# Server configuration
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CHUNK_SIZE = 1024
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
METRICS_URL = f"{SERVER_URL}/metrics"
HEALTH_CHECK_URL = f"{SERVER_URL}/health"

def start_vllm_server(chunk_size):
    """Start a vLLM server with the specified chunk size."""
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--max-num-batched-tokens", str(chunk_size),
    ]
    print(f"Starting server with command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                               text=True, bufsize=1, universal_newlines=True, preexec_fn=os.setsid)
    return process

# Create a log reader that won't block the main thread
def log_reader(process, log_queue):
    """Read logs from process and put them in queue."""
    for line in iter(process.stdout.readline, ''):
        log_queue.put(line.strip())
    log_queue.put(None)  # Signal end of stream

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

def shutdown_server(process):
    """Shut down the server gracefully."""
    if process and process.poll() is None:
        print("Shutting down server...")
        try:
            pgid = os.getpgid(process.pid)
            print(f"Sending SIGINT to process group {pgid}...")
            os.killpg(pgid, signal.SIGINT)
            process.wait(timeout=30)
            print("Server shut down via SIGINT.")
        except Exception as e:
            print(f"Error shutting down server: {e}")
            try:
                process.kill()
                print("Killed server process.")
            except Exception:
                pass

def get_metrics():
    """Fetch and display metrics from the metrics endpoint."""
    print(f"Fetching metrics from {METRICS_URL}...")
    try:
        response = requests.get(METRICS_URL, timeout=10)
        response.raise_for_status()
        
        # Get the full metrics text
        metrics_text = response.text
        
        # Print the first 20 lines to get a sample
        print("\n=== Sample of metrics (first 20 lines) ===")
        lines = metrics_text.split('\n')
        for i, line in enumerate(lines[:20]):
            if line.strip() and not line.startswith('#'):
                print(line)
        
        print(f"\nTotal metrics lines: {len(lines)}")
        
        # Also extract some specific metrics of interest
        print("\n=== Key metrics ===")
        interest_metrics = [
            "vllm:e2e_request_latency_seconds",
            "vllm:request_prefill_time_seconds",
            "vllm:request_decode_time_seconds",
            "vllm:num_running_tokens",
            "vllm:num_waiting_tokens"
        ]
        
        for metric in interest_metrics:
            pattern = rf'^{re.escape(metric)}' + r'(_sum|_count|)({[^}]*})?\s+([\d\.eE+-]+)'
            matches = re.findall(pattern, metrics_text, re.MULTILINE)
            if matches:
                print(f"\n{metric}:")
                for match in matches:
                    suffix, labels, value = match
                    metric_type = "value" if not suffix else suffix[1:]  # Remove underscore
                    print(f"  {metric_type}{labels}: {value}")
        
        return metrics_text
    except requests.RequestException as e:
        print(f"Error fetching metrics: {e}")
        return None

def test_server_startup_and_request():
    # Start the server
    print("Testing server startup with chunk size:", CHUNK_SIZE)
    server_process = start_vllm_server(CHUNK_SIZE)
    
    try:
        # Wait for the server to be ready
        if not wait_for_server_ready(server_process, HEALTH_CHECK_URL, timeout=300):
            print("Failed to start server, exiting test.")
            return
        
        print("Server started successfully!")
        
        # Get initial metrics (before any requests)
        print("\n=== Initial metrics (before sending requests) ===")
        get_metrics()
        
        print("\nNow sending a test request...")
        
        # Create OpenAI client
        client = openai.OpenAI(
            base_url=f"{SERVER_URL}/v1",
            api_key="EMPTY"
        )
        
        # Test prompt
        prompt = "Hello, my name is"
        
        # Send request
        start = time.time()
        try:
            response = client.completions.create(
                model=MODEL_NAME,
                prompt=prompt,
                max_tokens=32,
                temperature=0.7,
            )
            end = time.time()
            
            print("\nTest request successful!")
            print("Prompt:", prompt)
            print("Generated:", response.choices[0].text.strip())
            print(f"E2E Time: {end - start:.3f} seconds")
            
            # Get metrics after the request
            print("\n=== Metrics after request ===")
            time.sleep(1)  # Give a moment for metrics to update
            get_metrics()
            
        except Exception as e:
            print(f"Error sending test request: {e}")
    
    finally:
        # Always shut down the server
        print("\nShutting down test server...")
        shutdown_server(server_process)

if __name__ == "__main__":
    test_server_startup_and_request()
