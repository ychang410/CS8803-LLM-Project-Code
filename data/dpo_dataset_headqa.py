import json
import random
from typing import List, Dict, Any
from datasets import load_dataset
import os


def analyze_headqa_dataset(dataset_name: str = "EleutherAI/headqa", 
                          config: str = "en",
                          split: str = "train"):
    """
    Analyze the headqa dataset structure
    """
    print(f"Loading {dataset_name} ({config}, {split})...")
    try:
        dataset = load_dataset(dataset_name, config, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
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
                if isinstance(value[0], dict):
                    print(f"    List item keys: {list(value[0].keys())}")
    
    return dataset


def create_dpo_dataset(dataset_name: str = "EleutherAI/headqa",
                       config: str = "en",
                       split: str = "train",
                       output_path: str = "dpo_data/headqa_dpo.jsonl",
                       seed: int = 42):
    """
    Create DPO dataset from headqa
    
    Format:
    - prompt: qtext + answers
    - chosen: ra (right answer) 번호에 따른 answers에서 추출
    - rejected: 나머지 랜덤 추출
    """
    print(f"Loading {dataset_name} ({config}, {split})...")
    try:
        dataset = load_dataset(dataset_name, config, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []
    
    random.seed(seed)
    dpo_data = []
    
    skipped_count = 0
    
    for idx, item in enumerate(dataset):
        # Extract fields
        qtext = item.get('qtext', '')
        ra = item.get('ra', None)  # Right answer number
        answers = item.get('answers', [])  # List of answer dicts with 'aid' and 'atext'
        
        # Skip if missing required fields
        if not qtext:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Skipping item {idx} - missing qtext")
            continue
        
        if not answers or not isinstance(answers, list):
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Skipping item {idx} - missing or invalid answers")
            continue
        
        if ra is None:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Skipping item {idx} - missing ra")
            continue
        
        # Ensure ra is an integer
        if isinstance(ra, str):
            try:
                ra = int(ra)
            except:
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"Warning: Skipping item {idx} - invalid ra: {ra}")
                continue
        
        # Find the correct answer by matching ra with aid
        chosen_answer = None
        answer_dict = {}
        
        for answer in answers:
            if not isinstance(answer, dict):
                continue
            aid = answer.get('aid', None)
            atext = answer.get('atext', '')
            
            if aid is not None:
                # Convert aid to int if needed
                if isinstance(aid, str):
                    try:
                        aid = int(aid)
                    except:
                        continue
                
                answer_dict[aid] = atext
                
                # Check if this is the correct answer
                if aid == ra:
                    chosen_answer = atext
        
        # If we couldn't find chosen answer by matching, skip
        if chosen_answer is None:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Skipping item {idx} - could not find answer with aid={ra}. Available aids: {list(answer_dict.keys())}")
            continue
        
        # Build prompt: qtext + answers
        # Format answers as numbered list
        answers_text = "\n".join([f"{aid}. {atext}" for aid, atext in sorted(answer_dict.items())])
        prompt = f"{qtext}\n\n{answers_text}"
        
        # Build chosen: ra 번호에 따른 answers에서 추출
        chosen = f"{ra}. {chosen_answer}"
        
        # Build rejected: 나머지 랜덤 추출
        wrong_aids = [aid for aid in answer_dict.keys() if aid != ra]
        if not wrong_aids:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Skipping item {idx} - no wrong answers available")
            continue
        
        rejected_aid = random.choice(wrong_aids)
        rejected_answer = answer_dict[rejected_aid]
        rejected = f"{rejected_aid}. {rejected_answer}"
        
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
    dataset = analyze_headqa_dataset("EleutherAI/headqa", "en", "train")
    
    # Then create DPO dataset
    print("\n" + "="*50)
    print("CREATING DPO DATASET")
    print("="*50)
    dpo_data = create_dpo_dataset(
        dataset_name="EleutherAI/headqa",
        config="en",
        split="train",
        output_path="dpo_data/headqa_dpo.jsonl",
        seed=42
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

