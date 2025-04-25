import csv
from typing import List
import json
def load_prompts_from_csv(path: str, column_name: str = "question") -> List[str]:
    """
    Load prompts from a CSV file using the specified column.
    
    Args:
        path: Path to the CSV file containing prompts
        column_name: Name of the column containing the prompts (default: "question")
        
    Returns:
        List of prompt strings
    """
    try:
        # Read the CSV file
        prompts = []
        column_index = None
        
        with open(path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            # Get headers
            headers = next(reader)
            
            # Find column index
            try:
                column_index = headers.index(column_name)
            except ValueError:
                available_columns = ", ".join(headers)
                raise ValueError(f"Column '{column_name}' not found in CSV. Available columns: {available_columns}")
            
            # Extract prompts from the specified column
            for row in reader:
                if len(row) > column_index and row[column_index].strip():
                    prompts.append(row[column_index])
                    
                # Limit to first 8 prompts
                # if len(prompts) >= 8:
                #     break
        
        print(f"Loaded {len(prompts)} prompts from {path}")
        return prompts
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file {path} not found. Please create the file first.")
    except Exception as e:
        raise Exception(f"Error loading prompts from CSV: {str(e)}")


def load_prompts_from_json(path: str) -> List[str]:
    try:
        with open(path, "r") as f:
            prompts_data = json.load(f)
            return prompts_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompts file {path} not found. Please create the file first.")

def load_normalized_prompts(path: str) -> tuple:
    """
    Load normalized prompts from a JSON file
    
    Args:
        path: Path to JSON file with normalized prompts
        
    Returns:
        tuple: (prompts, token_counts, target_token_count)
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        prompts = data.get('prompts', [])
        token_counts = data.get('token_counts', [])
        target_length = data.get('target_length', 0)
        
        print(f"Loaded {len(prompts)} normalized prompts from {path}")
        print(f"All prompts have exactly {target_length} tokens: {all(count == target_length for count in token_counts)}")
        
        return prompts, token_counts, target_length
    except FileNotFoundError:
        print(f"Error: Normalized prompts file {path} not found.")
        return [], [], 0
    except Exception as e:
        print(f"Error loading normalized prompts from {path}: {e}")
        return [], [], 0