import json
import random
from typing import List, Dict, Any
from datasets import load_dataset
import os


def analyze_medqa_dataset(dataset_name: str = "truehealth/medqa", split: str = "train"):
    """
    Analyze the medqa dataset structure
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


def create_dpo_dataset(dataset_name: str = "truehealth/medqa", 
                       split: str = "train",
                       output_path: str = "dpo_data/medqa_dpo.jsonl",
                       seed: int = 42):
    """
    Create DPO dataset from medqa
    
    Format:
    - prompt: question + options
    - chosen: answer_idx + answer (correct answer)
    - rejected: random different answer_idx + corresponding wrong answer
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
    for idx, item in enumerate(dataset):
        # Extract fields (adjust based on actual dataset structure)
        # Try multiple possible field names for question
        question = (item.get('question', '') or 
                   item.get('Question', '') or 
                   item.get('qtext', '') or
                   item.get('query', ''))
        
        # Try multiple possible field names for options
        options = (item.get('options', None) or 
                  item.get('choices', None) or
                  item.get('Options', None) or
                  item.get('Choices', None) or
                  item.get('answers', None))
        
        # Try multiple possible field names for answer index
        # Note: 'answer' field contains the answer TEXT, not the index, so we check it last
        answer_idx = (item.get('answer_idx', None) or
                     item.get('correct_answer', None) or
                     item.get('label', None) or
                     item.get('answer_id', None) or
                     item.get('ra', None))  # 'ra' is used in HEAD_EN format
        
        # Handle different possible field names
        if answer_idx is None:
            # Try to find answer by matching text
            answer_text = item.get('answer', '') or item.get('Answer', '')
            if answer_text and options:
                for opt_idx, opt in enumerate(options):
                    if isinstance(opt, dict):
                        opt_text = opt.get('atext', '') or opt.get('text', '') or str(opt)
                    else:
                        opt_text = str(opt)
                    if str(opt_text).strip() == str(answer_text).strip():
                        answer_idx = opt_idx
                        break
        
        # Skip if we don't have required fields
        if not question:
            skipped_count += 1
            if skipped_count <= 5:  # Only print first 5 warnings
                print(f"Warning: Skipping item {idx} - missing question")
            continue
            
        if not options:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Skipping item {idx} - missing options")
            continue
            
        if answer_idx is None:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Skipping item {idx} - missing answer_idx. Fields: {list(item.keys())}")
            continue
        
        # Handle options - could be dict (like {'A': '...', 'B': '...'}) or list
        if isinstance(options, dict):
            # Convert dict to ordered list (A, B, C, D, E)
            option_keys = ['A', 'B', 'C', 'D', 'E']
            processed_options = []
            for key in option_keys:
                if key in options:
                    processed_options.append(options[key])
            options = processed_options
        elif isinstance(options, str):
            # If options is a string, try to parse it
            try:
                options = json.loads(options)
                # If parsed result is still a dict, convert it
                if isinstance(options, dict):
                    option_keys = ['A', 'B', 'C', 'D', 'E']
                    processed_options = []
                    for key in option_keys:
                        if key in options:
                            processed_options.append(options[key])
                    options = processed_options
            except:
                options = [options]
        elif not isinstance(options, list):
            options = [options]
        
        # Extract text from option dicts if needed (e.g., HEAD_EN format)
        processed_options = []
        for opt in options:
            if isinstance(opt, dict):
                # Try common field names
                opt_text = (opt.get('atext', '') or 
                           opt.get('text', '') or 
                           opt.get('option', '') or
                           str(opt))
                processed_options.append(opt_text)
            else:
                processed_options.append(str(opt))
        options = processed_options
        
        # Convert answer_idx to integer index
        # answer_idx could be: 'A'/'B'/'C'/'D'/'E' or integer or None
        if isinstance(answer_idx, str):
            # Convert letter to index (A=0, B=1, C=2, D=3, E=4)
            answer_idx_upper = answer_idx.upper().strip()
            if answer_idx_upper in ['A', 'B', 'C', 'D', 'E']:
                answer_idx = ord(answer_idx_upper) - ord('A')  # A=0, B=1, C=2, D=3, E=4
            else:
                # Try to convert to int
                try:
                    answer_idx = int(answer_idx)
                except:
                    # Try to find answer_idx by matching answer text
                    answer_text = item.get('answer', '') or item.get('Answer', '')
                    if answer_text:
                        for opt_idx, opt in enumerate(options):
                            if str(opt).strip() == str(answer_text).strip():
                                answer_idx = opt_idx
                                break
                        else:
                            skipped_count += 1
                            if skipped_count <= 5:
                                print(f"Warning: Could not find answer_idx for item {idx}")
                            continue
                    else:
                        skipped_count += 1
                        if skipped_count <= 5:
                            print(f"Warning: Could not determine answer_idx for item {idx}")
                        continue
        elif answer_idx is None:
            # Try to find answer_idx by matching answer text
            answer_text = item.get('answer', '') or item.get('Answer', '')
            if answer_text:
                for opt_idx, opt in enumerate(options):
                    if str(opt).strip() == str(answer_text).strip():
                        answer_idx = opt_idx
                        break
                else:
                    skipped_count += 1
                    if skipped_count <= 5:
                        print(f"Warning: Could not find answer_idx for item {idx}")
                    continue
            else:
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"Warning: Missing answer_idx and answer text for item {idx}")
                continue
        
        # Convert to 0-indexed if needed (if answer_idx is 1-indexed)
        # Note: If answer_idx came from letter conversion (A=0, B=1, etc.), it's already 0-indexed
        # Only convert if answer_idx is > len(options), suggesting it might be 1-indexed
        if isinstance(answer_idx, int):
            if answer_idx > len(options):
                # If answer_idx is > len(options), it might be 1-indexed, convert to 0-indexed
                answer_idx = answer_idx - 1
            # If answer_idx is between 1 and len(options), we assume it's already correct
            # (either 0-indexed or 1-indexed, but we'll validate it's in range)
        
        # Validate answer_idx
        if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx >= len(options):
            skipped_count += 1
            if skipped_count <= 5:
                print(f"Warning: Invalid answer_idx {answer_idx} (type: {type(answer_idx)}) for {len(options)} options. Skipping item {idx}.")
            continue
        
        # Build prompt: question + options (use A, B, C, D, E labels)
        option_labels = ['A', 'B', 'C', 'D', 'E']
        options_text = "\n".join([f"{option_labels[i]}. {opt}" for i, opt in enumerate(options)])
        prompt = f"Question: {question}\n\nOptions:\n{options_text}"
        
        # Build chosen: answer_idx + answer (use A, B, C, D, E labels)
        chosen_label = option_labels[answer_idx]
        chosen_answer = options[answer_idx]
        chosen = f"{chosen_label}. {chosen_answer}"
        
        # Build rejected: random different answer_idx + corresponding wrong answer
        wrong_indices = [i for i in range(len(options)) if i != answer_idx]
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
    dataset = analyze_medqa_dataset("truehealth/medqa", "train")
    
    # Then create DPO dataset
    print("\n" + "="*50)
    print("CREATING DPO DATASET")
    print("="*50)
    dpo_data = create_dpo_dataset(
        dataset_name="truehealth/medqa",
        split="train",
        output_path="dpo_data/medqa_dpo.jsonl",
        seed=42
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

