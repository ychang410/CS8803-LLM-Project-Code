import json
from typing import List, Dict, Any
from datasets import load_dataset
import os
import random
import csv


def save_to_jsonl(data: List[Dict[str, Any]], output_path: str):
    """Save data to JSONL format"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} examples to {output_path}")


def convert_to_sft_format(prompt_content: str, completion_content: str) -> Dict[str, Any]:
    """
    Convert to SFT format:
    {"prompt": [{"role": "user", "content": "..."}],
     "completion": [{"role": "assistant", "content": "..."}]}
    """
    return {
        "prompt": [{"role": "user", "content": prompt_content}],
        "completion": [{"role": "assistant", "content": completion_content}]
    }


def convert_pubmedqa_format(dataset_name: str = "qiaojin/PubMedQA", 
                           config: str = "pqa_labeled", 
                           split: str = "train") -> List[Dict[str, Any]]:
    """
    Convert PubMedQA dataset to SFT format
    
    Dataset structure:
    - pubid: int32
    - question: string
    - context: sequence (list of strings)
    - long_answer: string
    - final_decision: string (yes/no/maybe)
    """
    print(f"Loading {dataset_name} ({config})...")
    dataset = load_dataset(dataset_name, config, split=split)
    
    converted_data = []
    for item in dataset:
        question = item.get('question', '')
        context = item.get('context', {})
        final_decision = item.get('final_decision', '')
        
        # Extract only 'contexts' from context dict (exclude meshes, labels, etc.)
        contexts = context.get('contexts', []) if isinstance(context, dict) else []
        
        # Combine context sentences into a single string
        context_text = ' '.join(contexts) if isinstance(contexts, list) else str(contexts)
        
        # Create prompt with question and context
        prompt_content = f"Question: {question}\n\nContext: {context_text}"
        
        # Create completion with only final_decision (yes/no/maybe)
        completion_content = final_decision
        
        converted_data.append(convert_to_sft_format(prompt_content, completion_content))
    
    print(f"Converted {len(converted_data)} examples from PubMedQA")
    return converted_data


def convert_head_en_format(file_path: str) -> List[Dict[str, Any]]:
    """
    Convert HEAD_EN.json dataset to SFT format
    
    Dataset structure:
    - exams: dict of exam papers, each containing:
      - data: list of questions
        - qid: question ID
        - qtext: question text
        - answers: list of answer objects with 'aid' and 'atext'
        - ra: right answer ID (the correct answer, as string)
    """
    print(f"Loading {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    # Iterate through all exams
    exams = data.get('exams', {})
    for exam_name, exam_info in exams.items():
        questions = exam_info.get('data', [])
        
        # Process each question in the exam
        for item in questions:
            qtext = item.get('qtext', '')
            answers = item.get('answers', [])
            ra = item.get('ra', '')
            
            # Build options string
            options_lines = []
            correct_answer = None
            for idx, answer in enumerate(answers, 1):
                aid = answer.get('aid', '')
                atext = answer.get('atext', '')
                options_lines.append(f"{idx}. {atext}")
                
                # Find the correct answer (compare as strings)
                if str(aid) == str(ra):
                    correct_answer = f"{idx}. {atext}"
            
            # Create prompt with question and options
            options_text = '\n'.join(options_lines)
            prompt_content = f"Question: {qtext}\n\nOptions:\n{options_text}"
            
            # Create completion with just the answer number and text
            completion_content = correct_answer if correct_answer else ""
            
            if completion_content:  # Only add if we have a valid answer
                converted_data.append(convert_to_sft_format(prompt_content, completion_content))
    
    print(f"Converted {len(converted_data)} examples from HEAD_EN")
    return converted_data


def convert_opengpt_format(dataset_name: str = "openchat/cogstack-opengpt-sharegpt", 
                          split: str = "train") -> List[Dict[str, Any]]:
    """
    Convert OpenGPT dataset to SFT format
    
    Dataset structure:
    - id: string
    - conversations: list of messages
      - from: "human" or "gpt"
      - value: message content
    """
    print(f"Loading {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split)
    
    converted_data = []
    for item in dataset:
        conversations = item.get('conversations', [])
        
        # Extract first human message as prompt and first gpt message as completion
        prompt_content = None
        completion_content = None
        
        for conv in conversations:
            from_who = conv.get('from', '')
            value = conv.get('value', '')
            
            if from_who == 'human' and prompt_content is None:
                prompt_content = value
            elif from_who == 'gpt' and completion_content is None:
                completion_content = value
            
            # Stop after finding both
            if prompt_content and completion_content:
                break
        
        # Only add if we have both prompt and completion
        if prompt_content and completion_content:
            converted_data.append(convert_to_sft_format(prompt_content, completion_content))
    
    print(f"Converted {len(converted_data)} examples from OpenGPT")
    return converted_data


def convert_medquad_format(dataset_name: str = "keivalya/MedQuad-MedicalQnADataset", 
                          split: str = "train") -> List[Dict[str, Any]]:
    """
    Convert MedQuad dataset to SFT format
    
    Dataset structure:
    - qtype: question type
    - Question: question text
    - Answer: answer text
    """
    print(f"Loading {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split)
    
    converted_data = []
    for item in dataset:
        question = item.get('Question', '')
        answer = item.get('Answer', '')
        
        # Create prompt with question
        prompt_content = question
        
        # Create completion with answer
        completion_content = answer
        
        if prompt_content and completion_content:
            converted_data.append(convert_to_sft_format(prompt_content, completion_content))
    
    print(f"Converted {len(converted_data)} examples from MedQuad")
    return converted_data


def convert_cord19_format(csv_path: str) -> List[Dict[str, Any]]:
    """
    Convert CORD-19 dataset to SFT format
    
    CSV structure:
    - title: paper title
    - abstract: paper abstract
    - (other metadata fields)
    """
    print(f"Loading {csv_path}...")
    
    converted_data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            title = row.get('title', '').strip()
            abstract = row.get('abstract', '').strip()
            
            # Only include entries with both title and abstract
            if title and abstract:
                # Create prompt with instruction and abstract
                prompt_content = f"Please summerize the given abstract to a title\n\n{abstract}"
                
                # Create completion with title
                completion_content = title
                
                converted_data.append(convert_to_sft_format(prompt_content, completion_content))
    
    print(f"Converted {len(converted_data)} examples from CORD-19")
    return converted_data


def main():
    """Main function to convert datasets"""
    output_dir = "sft_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PubMedQA dataset (full)
    # pubmedqa_data = convert_pubmedqa_format("qiaojin/PubMedQA", "pqa_labeled", "train")
    # save_to_jsonl(pubmedqa_data, f"{output_dir}/PubMedQA_full.jsonl")
    
    # # Save random 499 samples to a separate file
    # random.seed(42)  # For reproducibility
    # pubmedqa_sample = random.sample(pubmedqa_data, 499)
    # save_to_jsonl(pubmedqa_sample, f"{output_dir}/PubMedQA.jsonl")
    
    # # Convert HEAD_EN dataset
    # head_en_data = convert_head_en_format("HEAD_EN.json")
    # save_to_jsonl(head_en_data, f"{output_dir}/HeadQA_full.jsonl")
    
    # # Save random 2657 samples to a separate file
    # headqa_sample = random.sample(head_en_data, 2657)
    # save_to_jsonl(headqa_sample, f"{output_dir}/HeadQA.jsonl")
    
    # Convert OpenGPT dataset
    # opengpt_data = convert_opengpt_format("openchat/cogstack-opengpt-sharegpt", "train")
    # save_to_jsonl(opengpt_data, f"{output_dir}/OpenGPT_full.jsonl")
    
    # Convert MedQuad dataset and sample 14553 examples
    # random.seed(42)  # For reproducibility
    # medquad_data = convert_medquad_format("keivalya/MedQuad-MedicalQnADataset", "train")
    # medquad_sample = random.sample(medquad_data, 14553)
    # save_to_jsonl(medquad_sample, f"{output_dir}/MedQuad.jsonl")
    
    # Convert CORD-19 dataset and sample 17721 examples
    random.seed(42)  # For reproducibility
    cord19_data = convert_cord19_format("original/metadata.csv")
    cord19_sample = random.sample(cord19_data, 17721)
    save_to_jsonl(cord19_sample, f"{output_dir}/CORD19.jsonl")
    
    print("\nAll conversions completed!")


if __name__ == "__main__":
    main()

