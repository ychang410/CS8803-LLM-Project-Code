import json
import random
import os
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], output_path: str):
    """Save data to JSONL format"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} examples to {output_path}")


def merge_and_shuffle_datasets(
    file_paths: List[str],
    output_path: str = "dpo_data/merged_dpo.jsonl",
    seed: int = 42
):
    """
    Merge multiple DPO datasets and shuffle them
    
    Args:
        file_paths: List of JSONL file paths to merge
        output_path: Output file path
        seed: Random seed for shuffling
    """
    print("="*50)
    print("MERGING AND SHUFFLING DPO DATASETS")
    print("="*50)
    
    all_data = []
    
    # Load all datasets
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"\nLoading {file_path}...")
        data = load_jsonl(file_path)
        print(f"  Loaded {len(data)} examples")
        all_data.extend(data)
    
    print(f"\nTotal examples before shuffling: {len(all_data)}")
    
    # Shuffle
    print(f"\nShuffling with seed {seed}...")
    random.seed(seed)
    random.shuffle(all_data)
    
    # Save merged and shuffled dataset
    print(f"\nSaving merged dataset...")
    save_jsonl(all_data, output_path)
    
    # Show statistics
    print("\n" + "="*50)
    print("Statistics:")
    print("="*50)
    print(f"Total examples: {len(all_data)}")
    
    # Show sample
    if all_data:
        print("\n" + "="*50)
        print("Sample examples (first 3):")
        print("="*50)
        for i in range(min(3, len(all_data))):
            print(f"\nExample {i+1}:")
            print(json.dumps(all_data[i], indent=2, ensure_ascii=False))
    
    return all_data


def main():
    """Main function"""
    # Define file paths
    file_paths = [
        "dpo_data/medqa_dpo.jsonl",
        "dpo_data/medmcqa_dpo.jsonl",
        "dpo_data/headqa_dpo.jsonl"
    ]
    
    # Merge and shuffle
    merged_data = merge_and_shuffle_datasets(
        file_paths=file_paths,
        output_path="dpo_data/merged_dpo.jsonl",
        seed=42
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

