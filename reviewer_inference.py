import os
# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"

import argparse
import json
import torch
from tqdm import tqdm
from unsloth import FastModel
from datasets import load_from_disk
from bert_score import score

from train_review_generator import STRENGTH_PROMPT, WEAKNESS_PROMPT

def load_model_for_inference(model_path, device="cuda"):
    """Load the saved model for inference with enhanced determinism"""
    print(f"Loading model from {model_path}")
    
    # Load model
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 16384,
        load_in_4bit = True,
    )
    model.eval()
    return model, tokenizer

def generate_review(sample, model, tokenizer, max_tokens, device="cuda"):
    """Generate a review for a sample"""
    # Determine the prompt type based on the review_type
    review_type = sample["review_type"].upper()
    
    if review_type == "STRENGTH":
        prompt_template = STRENGTH_PROMPT
    else:  # WEAKNESS
        prompt_template = WEAKNESS_PROMPT
    
    # Format the prompt with the paper text
    paper_text = sample["model_input"]
    prompt = prompt_template.format(paper_text=paper_text) + "\n<start_of_turn>model\n"
    
    # Tokenize the input
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    # Generate the review
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_tokens, # Increase for longer outputs!
            # Recommended Gemma-3 settings!
            temperature = 1.0, top_p = 0.95, top_k = 64,
        )

    # Extract only the generated part (excluding prompt)
    output_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return output_text

def calculate_bertscore(generated_text, reference_text):
    """Calculate BERTScore between generated and reference text"""
    precision, recall, f1 = score([generated_text], [reference_text], lang="en", verbose=False)
    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item()
    }

def main():
    parser = argparse.ArgumentParser(description="Generate reviews using a fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Maximum number of test samples to process")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for results")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    args = parser.parse_args()
    
    assert torch.cuda.is_available(), "CUDA is not available"
    device = "cuda"
    
    # Load model and dataset
    model, tokenizer = load_model_for_inference(args.model_path, device)
    dataset = load_from_disk(args.dataset_path)
    
    if "test" not in dataset:
        raise ValueError("Dataset must contain a 'test' split")
    
    test_dataset = dataset["test"]
    if args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(args.max_test_samples))
    
    results = []
    
    # Process each sample in the test dataset
    for sample in tqdm(test_dataset, desc="Generating reviews"):
        # Generate review
        generated_review = generate_review(sample, model, tokenizer, args.max_tokens, device)
        
        # Calculate BERTScore
        bertscore_results = calculate_bertscore(generated_review, sample["review"])
        bert_f1 = bertscore_results["f1"]
    
        # Add to results
        results.append({
            "paper_id": sample.get("id", "unknown"),
            "ground_truth": sample["review"],
            "generated_review": generated_review,
            "review_type": sample["review_type"],
            "bert_f1": bert_f1
        })
    
    # Save results to output file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    avg_f1 = sum(r["bert_f1"] for r in results) / len(results)
    print(f"Processed {len(results)} samples")
    print(f"Average BERTScore F1: {avg_f1:.4f}")

if __name__ == "__main__":
    main()
