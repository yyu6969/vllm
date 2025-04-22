import asyncio
import requests
import subprocess
import time
import json
import os
import sys
import signal
from argparse import Namespace

from vllm.entrypoints.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs


async def test_server():
    """Test function that starts a vLLM server and sends a test request."""
    # Define server config
    host = "127.0.0.1"
    port = 8000
    server_url = f"http://{host}:{port}"
    
    # Start server in a separate process
    print("Starting vLLM server...")
    # This uses a small model for quick testing
    server_process = subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.api_server",
        "--model", "facebook/opt-125m",  # Small model for testing
        "--host", host,
        "--port", str(port)
    ])
    
    # Wait for server to start
    max_retries = 30
    retry_interval = 2
    for i in range(max_retries):
        try:
            response = requests.get(f"{server_url}/health")
            if response.status_code == 200:
                print(f"Server is ready after {i * retry_interval} seconds")
                break
        except requests.exceptions.ConnectionError:
            print(f"Waiting for server to start... ({i+1}/{max_retries})")
            time.sleep(retry_interval)
    else:
        print("Failed to start server")
        server_process.terminate()
        return
    
    # Test the generate endpoint
    try:
        print("Testing generation...")
        test_prompt = "Hello, world!"
        request_data = {
            "prompt": test_prompt,
            "max_tokens": 20,
            "temperature": 0.7
        }
        
        response = requests.post(f"{server_url}/generate", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            print("Generation successful!")
            print(f"Input: {test_prompt}")
            print(f"Output: {result['text'][0]}")
        else:
            print(f"Generation failed with status code {response.status_code}")
            print(response.text)
    
    finally:
        # Clean up - terminate the server process
        print("Shutting down server...")
        server_process.send_signal(signal.SIGINT)
        server_process.wait()
        print("Server terminated")


if __name__ == "__main__":
    asyncio.run(test_server())
