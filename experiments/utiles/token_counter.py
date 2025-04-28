# Token counter for models with different tokenization behaviors
# including handling of different sampling parameters

import os
import time
import json
import csv
import argparse
from typing import Dict, List, Union, Tuple, Optional, Any

# Import required tokenizers
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


class DiffModelTokenCounter:
    """
    A token counter for different LLM models with various tokenization behaviors.
    Supports counting tokens in prompts and responses, and handles diff sampling parameters.
    """
    
    def __init__(self, model_name_or_path: str, tokenizer_mode: str = "auto", trust_remote_code: bool = False):
        """
        Initialize the token counter with the model's tokenizer.
        
        Args:
            model_name_or_path: HuggingFace model name or path to model
            tokenizer_mode: Mode for tokenizer ("auto", "fast", or "slow")
            trust_remote_code: Whether to trust remote code for the tokenizer
        """
        self.model_name = model_name_or_path
        
        # Try to use vLLM's tokenizer if available
        if HAS_VLLM:
            try:
                self.tokenizer = get_tokenizer(
                    model_name_or_path,
                    tokenizer_mode=tokenizer_mode,
                    trust_remote_code=trust_remote_code
                )
                self.tokenizer_source = "vllm"
                print(f"Using vLLM tokenizer for {model_name_or_path}")
                return
            except Exception as e:
                print(f"Failed to load vLLM tokenizer: {e}")
        
        # Fall back to HuggingFace tokenizer
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    use_fast=(tokenizer_mode != "slow"),
                    trust_remote_code=trust_remote_code
                )
                self.tokenizer_source = "transformers"
                print(f"Using HuggingFace tokenizer for {model_name_or_path}")
                return
            except Exception as e:
                print(f"Failed to load HuggingFace tokenizer: {e}")
        
        raise ValueError(
            "Could not initialize tokenizer. Please install either 'transformers' or 'vllm'."
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer_source == "vllm":
            return len(self.tokenizer.encode(text))
        else:  # transformers
            return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def count_tokens_with_details(self, text: str) -> Dict[str, Union[int, List[int]]]:
        """
        Count tokens with detailed information.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Dict with token count and token IDs
        """
        if self.tokenizer_source == "vllm":
            token_ids = self.tokenizer.encode(text)
        else:  # transformers
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        return {
            "count": len(token_ids),
            "token_ids": token_ids
        }
    
    def calculate_diff_token_count(self, 
                                   prompt: str, 
                                   response: str, 
                                   sampling_params: Optional[Dict] = None) -> Dict[str, Union[int, float]]:
        """
        Calculate token counts and related metrics for a prompt-response pair.
        
        Args:
            prompt: Input prompt text
            response: Model's generated response text
            sampling_params: Dictionary of sampling parameters used
            
        Returns:
            Dictionary with token counts and metrics
        """
        prompt_tokens = self.count_tokens(prompt)
        response_tokens = self.count_tokens(response)
        total_tokens = prompt_tokens + response_tokens
        
        result = {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "sampling_params": sampling_params or {}
        }
        
        # Calculate tokens per second if sampling params include timing info
        if sampling_params and "time_elapsed" in sampling_params:
            time_elapsed = sampling_params["time_elapsed"]
            result["tokens_per_second"] = response_tokens / time_elapsed if time_elapsed > 0 else 0
        
        return result

    def analyze_conversation(self, conversation: List[Dict[str, str]]) -> Dict:
        """
        Analyze a conversation, counting tokens for each message and total.
        
        Args:
            conversation: List of message dicts, each with 'role' and 'content'
            
        Returns:
            Dictionary with token counts by role and total
        """
        total_tokens = 0
        result = {"messages": [], "roles": {}}
        
        for message in conversation:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            message_tokens = self.count_tokens(content)
            total_tokens += message_tokens
            
            # Add to role-specific counts
            if role not in result["roles"]:
                result["roles"][role] = 0
            result["roles"][role] += message_tokens
            
            # Add message details
            result["messages"].append({
                "role": role,
                "content_tokens": message_tokens
            })
        
        result["total_tokens"] = total_tokens
        return result
    
    def token_usage_report(self, 
                         input_texts: List[str], 
                         output_texts: List[str], 
                         sampling_params: Optional[List[Dict]] = None) -> Dict:
        """
        Generate a comprehensive token usage report for batches of inputs and outputs.
        
        Args:
            input_texts: List of input prompts
            output_texts: List of model responses
            sampling_params: Optional list of sampling parameter dicts
            
        Returns:
            Detailed usage report with token counts and costs
        """
        if len(input_texts) != len(output_texts):
            raise ValueError("Input and output text lists must have the same length")
            
        if sampling_params and len(sampling_params) != len(input_texts):
            raise ValueError("Sampling parameters must match the number of inputs")
            
        total_input_tokens = 0
        total_output_tokens = 0
        
        samples = []
        for i, (input_text, output_text) in enumerate(zip(input_texts, output_texts)):
            input_count = self.count_tokens(input_text)
            output_count = self.count_tokens(output_text)
            
            total_input_tokens += input_count
            total_output_tokens += output_count
            
            sample_data = {
                "input_tokens": input_count,
                "output_tokens": output_count,
                "total_tokens": input_count + output_count
            }
            
            if sampling_params:
                sample_data["sampling_params"] = sampling_params[i]
            
            samples.append(sample_data)
        
        report = {
            "model": self.model_name,
            "tokenizer_source": self.tokenizer_source,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "samples": samples
        }
        
        return report
    
    def process_json_file(self, json_file_path: str, prompt_field: str = "prompt") -> Dict[str, Any]:
        """
        Process a JSON file containing multiple prompts.
        
        Args:
            json_file_path: Path to the JSON file
            prompt_field: The field name that contains the prompt text
            
        Returns:
            Dictionary with token counts for each prompt and total
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        prompts = []
        if isinstance(data, list):
            # JSON array of objects
            for item in data:
                if prompt_field in item:
                    prompts.append(item[prompt_field])
        elif isinstance(data, dict):
            # Single JSON object with multiple prompts
            if prompt_field in data:
                if isinstance(data[prompt_field], list):
                    prompts = data[prompt_field]
                else:
                    prompts = [data[prompt_field]]
            else:
                # Check if the dict has numbered keys or other prompt fields
                for key, value in data.items():
                    if isinstance(value, dict) and prompt_field in value:
                        prompts.append(value[prompt_field])
                    elif isinstance(value, str):
                        # Assume any string values might be prompts
                        prompts.append(value)
        
        # Count tokens for each prompt
        results = []
        total_tokens = 0
        for i, prompt in enumerate(prompts):
            count = self.count_tokens(prompt)
            total_tokens += count
            results.append({
                "prompt_index": i,
                "tokens": count,
                "text_length": len(prompt)
            })
        
        return {
            "file": json_file_path,
            "prompt_count": len(prompts),
            "total_tokens": total_tokens,
            "prompts": results
        }
    
    def process_csv_file(self, csv_file_path: str, prompt_column: str = "prompt", 
                        has_header: bool = True, delimiter: str = ',') -> Dict[str, Any]:
        """
        Process a CSV file containing multiple prompts.
        
        Args:
            csv_file_path: Path to the CSV file
            prompt_column: The column name that contains the prompt text
            has_header: Whether the CSV file has a header row
            delimiter: The delimiter used in the CSV file
            
        Returns:
            Dictionary with token counts for each prompt and total
        """
        prompts = []
        
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    if prompt_column in row:
                        prompts.append(row[prompt_column])
            else:
                reader = csv.reader(f, delimiter=delimiter)
                # Try to figure out which column contains the prompts
                # For now, just use the first column if no header
                for row in reader:
                    if row:  # Ensure the row is not empty
                        prompts.append(row[0])
        
        # Count tokens for each prompt
        results = []
        total_tokens = 0
        for i, prompt in enumerate(prompts):
            count = self.count_tokens(prompt)
            total_tokens += count
            results.append({
                "prompt_index": i,
                "tokens": count,
                "text_length": len(prompt)
            })
        
        return {
            "file": csv_file_path,
            "prompt_count": len(prompts),
            "total_tokens": total_tokens,
            "prompts": results
        }


def main():
    parser = argparse.ArgumentParser(description="Token Counter for Diff Models")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--mode", choices=["text", "json", "csv"], default="text", 
                       help="Count tokens directly from text, from JSON, or from CSV")
    parser.add_argument("--input", type=str, help="Input text or file path")
    parser.add_argument("--prompt-field", type=str, default="prompt", 
                       help="Field name for prompts in JSON or column name in CSV")
    parser.add_argument("--no-header", action="store_true", 
                       help="Specify if CSV file has no header row")
    parser.add_argument("--delimiter", type=str, default=",", 
                       help="Delimiter for CSV files (default is comma)")
    parser.add_argument("--trust-remote-code", action="store_true", 
                       help="Trust remote code for tokenizer")
    parser.add_argument("--tokenizer-mode", choices=["auto", "fast", "slow"], default="auto", 
                       help="Tokenizer mode")
    parser.add_argument("--output", type=str, help="Output file path for results")
    
    args = parser.parse_args()
    
    # Initialize token counter
    counter = DiffModelTokenCounter(
        args.model, 
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code
    )
    
    # Process input based on mode
    start_time = time.time()
    
    if args.mode == "json":
        if not args.input:
            parser.error("--input file path is required when mode is 'json'")
        result = counter.process_json_file(args.input, prompt_field=args.prompt_field)
    
    elif args.mode == "csv":
        if not args.input:
            parser.error("--input file path is required when mode is 'csv'")
        result = counter.process_csv_file(
            args.input, 
            prompt_column=args.prompt_field, 
            has_header=not args.no_header,
            delimiter=args.delimiter
        )
    
    else:  # text mode
        if not args.input:
            parser.error("--input text is required when mode is 'text'")
        result = counter.count_tokens_with_details(args.input)
        result["text_length"] = len(args.input)
    
    elapsed_time = time.time() - start_time
    
    # Add metadata
    result["model"] = args.model
    result["processing_time_seconds"] = elapsed_time
    
    # Output results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
