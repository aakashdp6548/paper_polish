import os
import argparse
import wandb
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType

# for debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-90fa6d82-ba72-5026-b4de-fb27210cba83,MIG-6d25d029-31ac-5260-87fb-3e232ed96dad"

def preprocess_function_batched(examples, tokenizer, token_max: int):
    """
    Tokenizes batches of examples for Gemma-IT fine-tuning.
    Handles instruction/paper and review parts separately for truncation.
    Masks instruction/input labels. Pads to token_max.
    Assumes 'examples' is a dictionary of lists (batched input).
    """
    batch_size = len(examples["model_input"])

    # --- Part 1: Instructions + Paper ---
    part1_texts = []
    for i in range(batch_size):

        part1_texts.append(
            "<bos><start_of_turn>user\n"
            "### INSTRUCTIONS:\n"
            "You are an expert machine learning researcher. Given the content of a paper submitted"
            "to the International Conference on Learning Representations (ICLR), write a helpful "
            "review that highlights the paper's strengths and weaknesses.\n\n"
            "### PAPER:\n"
            f"{examples['model_input'][i]}\n"
            "<end_of_turn>\n"
        )

    # Tokenize part 1 for the whole batch, no padding here yet, just truncate if needed initially
    part1_tokens = tokenizer(
        part1_texts,
        truncation=True,
        max_length=token_max,
        add_special_tokens=False
    )

    # --- Part 2: Review (Strengths + Weaknesses) ---
    part2_texts = []
    for i in range(batch_size):

        strength = examples["strengths"][i]
        weakness = examples["weaknesses"][i]
        strength_text = strength[0] if len(strength) > 0 else ""
        weakness_text = weakness[0] if len(weakness) > 0 else ""

        part2_texts.append(
            "<start_of_turn>model\n"
            "### REVIEW:\n"
            "**Strengths:**\n"
            f"{strength_text}\n\n"
            "**Weaknesses:**\n"
            f"{weakness_text}\n"
            "<end_of_turn>"
        )

    final_input_ids = []
    final_attention_mask = []
    final_labels = []

    for i in range(batch_size):
        # Get tokens for part 1 for this specific example
        part1_ids = part1_tokens["input_ids"][i]
        part1_mask = part1_tokens["attention_mask"][i]
        prompt_len = len(part1_ids)

        # remaining tokens for part 2
        remaining_tokens = token_max - prompt_len

        part2_ids = []
        part2_mask = []
        if remaining_tokens > 0:
            # Tokenize part 2 *individually* because max_length depends on part 1
            part2_single_tokens = tokenizer(
                part2_texts[i],
                truncation=True,
                max_length=remaining_tokens,
                add_special_tokens=False
            )
            part2_ids = part2_single_tokens["input_ids"]
            part2_mask = part2_single_tokens["attention_mask"]

        combined_ids = part1_ids + part2_ids
        combined_mask = part1_mask + part2_mask

        labels = combined_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len

        # Manual padding
        pad_len = token_max - len(combined_ids)

        final_input_ids.append(combined_ids + [tokenizer.pad_token_id] * pad_len)
        final_attention_mask.append(combined_mask + [0] * pad_len)
        final_labels.append(labels + [-100] * pad_len) # Pad labels with -100
    
    print(f"input_ids[0] shape: {len(final_input_ids[0])}, total: {len(final_input_ids)}")
    print(f"attention_mask[0] shape: {len(final_attention_mask[0])}, total: {len(final_attention_mask)}")
    print(f"labels[0] shape: {len(final_labels[0])}, total: {len(final_labels)}")

    return {
        "input_ids": final_input_ids,
        "attention_mask": final_attention_mask,
        "labels": final_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma3-4B for ICLR-style paper review generation")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it", help="HuggingFace model name")
    parser.add_argument("--dataset_path", type=str, required=True, help="HuggingFace dataset path")
    parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, required=True, help="WandB run name")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_length", type=int, default=12000, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default="/gpfs/radev/home/tj372/scratch/paper_polish/review_models", help="Output directory")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    args = parser.parse_args()

    # Initialize wandb for tracking
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    
    # Load dataset and tokenizer
    dataset = load_from_disk(args.dataset_path)
    # # Select only the first example from each split
    # dataset = {
    #     "train": dataset["train"].select([0]),
    #     "val": dataset["val"].select([0]),
    #     "test": dataset["test"].select([0])
    # }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Starting dataset preprocessing...")
    tokenized_dataset = {
        split: dataset[split].map(
            lambda examples: preprocess_function_batched(examples, tokenizer, args.max_length),
            batched=True,
        )
        for split in dataset
    }
    print("Dataset preprocessing finished.")

    # Set format to ensure DataLoader gets tensors
    for split in tokenized_dataset:
        tokenized_dataset[split].set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
    
    # Inspect a sample using DataLoader with default_data_collator
    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=1,
        collate_fn=default_data_collator
    )
    sample = next(iter(train_loader))
    print("Sample batch from DataLoader:")
    for k, v in sample.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")\
    
    print(tokenized_dataset)
    

    # Load the causal language model (for text generation)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # use lora for peft
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "down_proj", "up_proj"],
    )
    model = get_peft_model(
        model,
        lora_config,
        )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=8,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=100,
        save_total_limit=3,
        bf16=True,
    )
    
    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        data_collator=default_data_collator,
    )

    # Train the model
    trainer.train()
    
    # Evaluate on the test set if available
    if "test" in tokenized_dataset:
        results = trainer.evaluate(tokenized_dataset["test"])
        print(f"Test results: {results}")
        wandb.log({"test": results})
    
    # Save the final model
    trainer.save_model(f"{args.output_dir}/final")
    wandb.finish()

if __name__ == "__main__":
    main()
