import json
import random
from typing import List, Dict, Any
from datasets import load_dataset
import os


def analyze_medmcqa_dataset(dataset_name: str = "openlifescienceai/medmcqa", split: str = "train"):
    """
    Analyze the medmcqa dataset structure
    """
    print(f"Loading {dataset_name} ({split})...")
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Try without split
        try:
            dataset = load_dataset(dataset_name)
            if isinstance(dataset, dict):
                split = list(dataset.keys())[0]
                dataset = dataset[split]
        except Exception as e2:
            print(f"Error loading dataset without split: {e2}")
            return None
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"\nDataset features: {dataset.features}")
    
    # Show first few examples
    print("\n" + "="*50)
    print("First 3 examples:")
    print("="*50)
    for i in range(min(3, len(dataset))):
        print(f"\nExample {i+1}:")
        example = dict(dataset[i])
        # Truncate long strings for readability
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 200:
                example[key] = value[:200] + "..."
        print(json.dumps(example, indent=2, ensure_ascii=False))
    
    # Analyze field names
    print("\n" + "="*50)
    print("Field Analysis:")
    print("="*50)
    if len(dataset) > 0:
        first_item = dict(dataset[0])
        print(f"Available fields: {list(first_item.keys())}")
        for key, value in first_item.items():
            print(f"  - {key}: {type(value).__name__}")
            if isinstance(value, list) and len(value) > 0:
                print(f"    (list of {type(value[0]).__name__}, length: {len(value)})")
    
    return dataset


def create_dpo_dataset(dataset_name: str = "openlifescienceai/medmcqa", 
                       split: str = "train",
                       output_path: str = "dpo_data/medmcqa_dpo.jsonl",
                       seed: int = 42):
    """
    Create DPO dataset from medmcqa
    
    Format:
    - prompt: question + "Options: " + opa, opb, opc, opd
    - chosen: cop (번호를 알파벳으로 변환: 0=A, 1=B, 2=C, 3=D) + 맞는 op
    - rejected: 나머지 숫자 중 랜덤 cop + 해당 op
    """
    print(f"Loading {dataset_name} ({split})...")
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Try without split
        try:
            dataset = load_dataset(dataset_name)
            if isinstance(dataset, dict):
                split = list(dataset.keys())[0]
                dataset = dataset[split]
                print(f"Using split: {split}")
        except Exception as e2:
            print(f"Error loading dataset: {e2}")
            return []
    
    random.seed(seed)
    dpo_data = []
    
    skipped_count = 0
    option_labels = ['A', 'B', 'C', 'D']  # 0=A, 1=B, 2=C, 3=D
    
    for idx, item in enumerate(dataset):
        # Extract fields
        question = item.get('question', '')
        opa = item.get('opa', '')
        opb = item.get('opb', '')
        opc = item.get('opc', '')
        opd = item.get('opd', '')
        cop = item.get('cop', None)  # Correct option number (0, 1, 2, or 3)
        
        # Skip if missing required fields
        if not question:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Skipping item {idx} - missing question")
            continue
        
        if not all([opa, opb, opc, opd]):
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Skipping item {idx} - missing options")
            continue
        
        if cop is None:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Skipping item {idx} - missing cop")
            continue
        
        # Ensure cop is an integer and in valid range
        if isinstance(cop, str):
            try:
                cop = int(cop)
            except:
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"Warning: Skipping item {idx} - invalid cop: {cop}")
                continue
        
        # Validate cop is in range [0, 3]
        if not isinstance(cop, int) or cop < 0 or cop >= 4:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Skipping item {idx} - cop out of range: {cop}")
            continue
        
        # Build options list
        options = [opa, opb, opc, opd]
        
        # Build prompt: question + "Options: " + opa, opb, opc, opd
        options_text = "\n".join([f"{option_labels[i]}. {opt}" for i, opt in enumerate(options)])
        prompt = f"{question}\n\nOptions:\n{options_text}"
        
        # Build chosen: cop (알파벳으로 변환) + 맞는 op
        chosen_label = option_labels[cop]
        chosen_answer = options[cop]
        chosen = f"{chosen_label}. {chosen_answer}"
        
        # Build rejected: 나머지 숫자 중 랜덤 cop + 해당 op
        wrong_indices = [i for i in range(4) if i != cop]
        if not wrong_indices:
            continue  # Skip if no wrong options available
        
        rejected_idx = random.choice(wrong_indices)
        rejected_label = option_labels[rejected_idx]
        rejected_answer = options[rejected_idx]
        rejected = f"{rejected_label}. {rejected_answer}"
        
        # Create DPO format
        dpo_item = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
        
        dpo_data.append(dpo_item)
    
    # Save to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nSaved {len(dpo_data)} DPO examples to {output_path}")
    if skipped_count > 5:
        print(f"(Skipped {skipped_count} items due to missing or invalid fields)")
    
    # Show sample
    if dpo_data:
        print("\n" + "="*50)
        print("Sample DPO example:")
        print("="*50)
        print(json.dumps(dpo_data[0], indent=2, ensure_ascii=False))
        
        # Show statistics
        print("\n" + "="*50)
        print("Statistics:")
        print("="*50)
        print(f"Total examples: {len(dpo_data)}")
        print(f"Skipped examples: {skipped_count}")
        avg_prompt_len = sum(len(item['prompt']) for item in dpo_data) / len(dpo_data)
        avg_chosen_len = sum(len(item['chosen']) for item in dpo_data) / len(dpo_data)
        avg_rejected_len = sum(len(item['rejected']) for item in dpo_data) / len(dpo_data)
        print(f"Average prompt length: {avg_prompt_len:.1f} characters")
        print(f"Average chosen length: {avg_chosen_len:.1f} characters")
        print(f"Average rejected length: {avg_rejected_len:.1f} characters")
    
    return dpo_data


def main():
    """Main function"""
    # First, analyze the dataset
    print("="*50)
    print("ANALYZING DATASET")
    print("="*50)
    dataset = analyze_medmcqa_dataset("openlifescienceai/medmcqa", "train")
    
    # Then create DPO dataset
    print("\n" + "="*50)
    print("CREATING DPO DATASET")
    print("="*50)
    dpo_data = create_dpo_dataset(
        dataset_name="openlifescienceai/medmcqa",
        split="train",
        output_path="dpo_data/medmcqa_dpo.jsonl",
        seed=42
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

