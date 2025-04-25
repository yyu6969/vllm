import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any
from transformers import AutoTokenizer

# Add parent directory to Python path for importing utilities
sys.path.append('/work/nvme/bdkz/yyu69/vllm')
from experiments.utiles.load_prompts import load_prompts_from_csv

# Define target token lengths for each prompt file
PROMPT_FILES_TOKEN_LENGTHS = {
    "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/select-text-by-length_1250_1375.csv": 300,
    "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/select-text-by-length_2500_2750.csv": 600,
    "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/select-text-by-length_5000_5500.csv": 1200,
    "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/select-text-by-length_10000_11000.csv": 2600,
    "/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/select-text-by-length_20000_22000.csv": 5300,
}

def calculate_token_count(prompt: str, tokenizer) -> int:
    """Calculate the number of tokens in a prompt"""
    return len(tokenizer.encode(prompt))

def truncate_or_pad_to_token_length(prompt: str, tokenizer, target_length: int) -> str:
    """
    Truncate or pad a prompt to ensure it has exactly target_length tokens.
    
    Args:
        prompt: The text prompt to modify
        tokenizer: The tokenizer to use
        target_length: The desired token length
        
    Returns:
        Modified prompt with exactly target_length tokens
    """
    tokens = tokenizer.encode(prompt)
    current_length = len(tokens)
    
    if current_length == target_length:
        return prompt  # Already the right length
    
    if current_length > target_length:
        # Truncate from the right (end of the text)
        truncated_tokens = tokens[:target_length]
        return tokenizer.decode(truncated_tokens)
    else:
        # For padding, instead of repeatedly adding one space at a time,
        # we'll use a more efficient approach
        tokens_to_add = target_length - current_length
        
        # Method 1: If the tokenizer has a pad token, use that
        if tokenizer.pad_token_id is not None:
            padded_tokens = tokens + [tokenizer.pad_token_id] * tokens_to_add
            return tokenizer.decode(padded_tokens)
        
        # Method 2: For tokenizers without a pad token, we'll add spaces efficiently
        # First, check if a single space is a token
        space = " "
        space_tokens = tokenizer.encode(space, add_special_tokens=False)
        
        if len(space_tokens) == 1:
            # If a space is a single token, we can add the right number directly
            padded_tokens = tokens + space_tokens * tokens_to_add
            return tokenizer.decode(padded_tokens)
        
        # Method 3: If spaces don't work well, try creating a padding string first
        # then tokenize it just once
        padding_str = " " * (tokens_to_add * 2)  # Extra spaces to be safe
        padding_tokens = tokenizer.encode(padding_str, add_special_tokens=False)
        
        # Take only the tokens we need
        if len(padding_tokens) >= tokens_to_add:
            padded_tokens = tokens + padding_tokens[:tokens_to_add]
            return tokenizer.decode(padded_tokens)
        
        # Method 4: If all else fails, fall back to a token-by-token approach
        # but doing it much more efficiently by adding chunks
        # Start with the original tokens
        result_tokens = tokens.copy()
        
        # Try padding with different characters
        pad_chars = [" ", ".", ",", "-", "x"]
        
        for char in pad_chars:
            # First try adding 10 at a time to speed things up
            chunk = char * 10
            chunk_tokens = tokenizer.encode(chunk, add_special_tokens=False)
            
            while len(result_tokens) + len(chunk_tokens) <= target_length:
                result_tokens.extend(chunk_tokens)
            
            # Then add one at a time for precision
            single_token = tokenizer.encode(char, add_special_tokens=False)
            while len(result_tokens) < target_length:
                result_tokens.extend(single_token)
            
            if len(result_tokens) == target_length:
                return tokenizer.decode(result_tokens)
            
            # If we've overshot, truncate
            if len(result_tokens) > target_length:
                return tokenizer.decode(result_tokens[:target_length])
        
        # If we get here, something went wrong
        print(f"Warning: Could not pad prompt to exact length. Current: {len(result_tokens)}, Target: {target_length}")
        return tokenizer.decode(result_tokens[:target_length])

def normalize_prompts(file_path: str, tokenizer, target_length: int, batch_size: int) -> Dict[str, Any]:
    """
    Load prompts from CSV and normalize them to have exactly target_length tokens.
    
    Args:
        file_path: Path to CSV file with prompts
        tokenizer: Tokenizer to use for token counting
        target_length: Target token length for all prompts
        batch_size: Number of prompts to process
        
    Returns:
        Dictionary with normalized prompts and metadata
    """
    print(f"\nProcessing file: {file_path}")
    print(f"Target token length: {target_length}")
    
    # Load prompts from CSV
    prompts = load_prompts_from_csv(file_path, column_name="text")
    
    # Limit to batch_size prompts
    prompts = prompts[:batch_size]
    
    if not prompts:
        print(f"No prompts loaded from {file_path}")
        return {
            "original_file": file_path,
            "target_length": target_length,
            "prompts": [],
            "token_counts": []
        }
    
    # Calculate original token counts
    print("Calculating original token counts...")
    original_token_counts = [calculate_token_count(prompt, tokenizer) for prompt in prompts]
    print(f"Original token counts: {original_token_counts}")
    print(f"Min: {min(original_token_counts)}, Max: {max(original_token_counts)}")
    
    # Normalize to target length
    normalized_prompts = []
    normalized_token_counts = []
    
    print("Starting prompt normalization...")
    start_time = time.time()
    
    for i, prompt in enumerate(prompts):
        prompt_start = time.time()
        print(f"Processing prompt {i+1}/{len(prompts)}...", end="", flush=True)
        
        normalized_prompt = truncate_or_pad_to_token_length(prompt, tokenizer, target_length)
        normalized_token_count = calculate_token_count(normalized_prompt, tokenizer)
        
        # Verify token count - if not exact, try one more time
        if normalized_token_count != target_length:
            print(f"\nWarning: Token count mismatch after normalization. Expected {target_length}, got {normalized_token_count}. Retrying...")
            normalized_prompt = truncate_or_pad_to_token_length(normalized_prompt, tokenizer, target_length)
            normalized_token_count = calculate_token_count(normalized_prompt, tokenizer)
        
        normalized_prompts.append(normalized_prompt)
        normalized_token_counts.append(normalized_token_count)
        
        prompt_time = time.time() - prompt_start
        print(f" done in {prompt_time:.2f}s, token count: {normalized_token_count}")
    
    total_time = time.time() - start_time
    print(f"Normalization completed in {total_time:.2f}s, average {total_time/len(prompts):.2f}s per prompt")
    print(f"Normalized token counts: {normalized_token_counts}")
    print(f"All prompts normalized to exactly {target_length} tokens: {all(count == target_length for count in normalized_token_counts)}")
    
    # Return normalized prompts and metadata
    return {
        "original_file": file_path,
        "target_length": target_length,
        "prompts": normalized_prompts,
        "token_counts": normalized_token_counts
    }

def preprocess_all_prompts(model_name: str, output_dir: str, batch_size: int = 8, verbose: bool = False):
    """
    Preprocess all prompt files with normalization
    
    Args:
        model_name: Model name to load tokenizer from
        output_dir: Directory to save normalized prompts
        batch_size: Number of prompts per file
        verbose: Whether to print verbose output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer for model: {model_name}")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
    
    # Process each prompt file
    for file_path, target_length in PROMPT_FILES_TOKEN_LENGTHS.items():
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}, skipping")
            continue
        
        # Normalize prompts
        result = normalize_prompts(file_path, tokenizer, target_length, batch_size)
        
        if not result["prompts"]:
            print(f"No prompts processed for {file_path}, skipping")
            continue
        
        # Create output filename
        base_filename = os.path.basename(file_path).replace(".csv", "")
        output_file = os.path.join(output_dir, f"{base_filename}_normalized_{target_length}.json")
        
        # Save normalized prompts
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Saved {len(result['prompts'])} normalized prompts to {output_file}")
        
        # Print sample if verbose
        if verbose and result["prompts"]:
            print("\nSample prompt (first 100 chars):")
            print(f"Original: {prompts[0][:100]}...")
            print(f"Normalized: {result['prompts'][0][:100]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize prompts to exact token lengths")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct", 
                        help="Model name to load tokenizer from")
    parser.add_argument("--output-dir", type=str, 
                        default="/work/nvme/bdkz/yyu69/vllm/data/prefill_decode/normalized", 
                        help="Directory to save normalized prompts")
    parser.add_argument("--batch-size", type=int, default=8, 
                        help="Number of prompts per file")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print verbose output")
    args = parser.parse_args()
    
    preprocess_all_prompts(args.model, args.output_dir, args.batch_size, args.verbose)
