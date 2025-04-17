import argparse
import wandb
from unsloth import FastModel
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


# #################################################################
# NOTE: There's a bug in transformers that causes eval to fail when
# using bfloat16 with gemma-3. To get around this, find the model's
# config.json file by running
#   `find /path/to/huggingface/cache -name config.json 2>/dev/null`
# and change use_config to "false" in the file. Change it back
# before inference.
# #################################################################


### Prompts and constants
STRENGTH_PROMPT = """<bos><start_of_turn>user
### INSTRUCTIONS:
You are an expert machine learning researcher. Given the content of a paper submitted
to the a top machine learning conference, write a helpful review that highlights whether
the paper is a good fit for the conference.

Specifically, come up with a list of STRENGTHS of the paper. Be specific
and concise. Think about the paper's contributions, the novelty of the approach, and the
quality of the communication.

### PAPER:
{paper_text}
<end_of_turn>
"""
 
WEAKNESS_PROMPT = """<bos><start_of_turn>user
### INSTRUCTIONS:
You are an expert machine learning researcher. Given the content of a paper submitted
to the a top machine learning conference, write a helpful review that highlights whether
the paper is a good fit for the conference.

Specifically, come up with a list of WEAKNESSES of the paper. Be specific
and concise. Think about the paper's contributions, the novelty of the approach, and the
quality of the communication. Wherever possible, suggest specific actionable improvements to the paper.

### PAPER:
{paper_text}
<end_of_turn>
"""

MODEL_RESPONSE = """<start_of_turn>model\n
### REVIEW:
**{review_type}:**\n
{review_text}
<end_of_turn>
"""

INSTRUCTION_TEMPLATE = "### INSTRUCTIONS:"
RESPONSE_TEMPLATE = "### REVIEW:"


# Preprocessing functions

def preprocess_function_batched(examples, tokenizer, token_max: int):
    """
    Tokenizes batches of examples for Gemma-IT fine-tuning.
    Handles instruction/paper and review parts separately for truncation.
    Masks instruction/input labels. Pads to token_max.
    Assumes 'examples' is a dictionary of lists (batched input).
    """
    batch_size = len(examples["model_input"])
    inputs = []
    response_lengths = []

    for i in range(batch_size):
        review_type = examples["review_type"][i].upper()
        user_prompt_template = STRENGTH_PROMPT if review_type == "STRENGTH" else WEAKNESS_PROMPT
        model_response = MODEL_RESPONSE.format(review_type=review_type, review_text=examples["review"][i])

        # Tokenize the user prompt and model response, and calculate the length of the combined text
        user_prompt_tokens = tokenizer(user_prompt_template, add_special_tokens=False)["input_ids"]
        model_response_tokens = tokenizer(model_response, add_special_tokens=False)["input_ids"]
        combined_length = len(user_prompt_tokens) + len(model_response_tokens)
        tokens_remaining = token_max - combined_length
        response_lengths.append(len(model_response_tokens))
        # Tokenize the paper text and truncate if necessary
        paper_text = examples["model_input"][i]
        paper_tokens = tokenizer(paper_text, add_special_tokens=False)["input_ids"]
        if len(paper_tokens) > tokens_remaining:
            paper_tokens = paper_tokens[:tokens_remaining]
            paper_text = tokenizer.decode(paper_tokens)

        # Add the paper text to the combined text
        user_prompt = user_prompt_template.format(paper_text=paper_text)
        full_prompt = user_prompt + model_response

        inputs.append(full_prompt)

    return {
        "prompt": inputs
    }



# Main function

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma3-4B for ICLR-style paper review generation")
    
    # Data and model parameters
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it", help="HuggingFace model name")
    parser.add_argument("--dataset_path", type=str, required=True, help="HuggingFace dataset path")
    parser.add_argument("--output_dir", type=str, default="/gpfs/radev/home/tj372/scratch/paper_polish/review_models", help="Output directory")

    # Wandb parameters
    parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, required=True, help="WandB run name")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=12000, help="Max sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--lora_rank", type=int, default=8, help="Lora rank")
    parser.add_argument("--lora_alpha", type=int, default=8, help="Lora alpha")
    parser.add_argument("--lora_dropout", type=float, default=0, help="Lora dropout")

    # Training loop parameters
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of train epochs")
    parser.add_argument("--max_eval_samples", type=int, default=1000, help="Max eval samples")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save every N steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Eval every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Save total limit")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Initialize wandb for tracking
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    
    model, tokenizer = FastModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_length, # Choose any for long context!
        load_in_4bit = False,  # 4 bit quantization to reduce memory
        load_in_8bit = True, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!

        r = args.lora_rank,           # Larger = higher accuracy, but might overfit
        lora_alpha = args.lora_alpha,  # Recommended alpha == r at least
        lora_dropout = args.lora_dropout,
        bias = "none",
        random_state = args.seed,
    )
    
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    dataset = load_from_disk(args.dataset_path)
    for split in dataset:
        dataset[split] = dataset[split].shuffle(seed=args.seed)

    # Subset the validation set if max_eval_steps is set
    if args.max_eval_samples:
        dataset["val"] = dataset["val"].select(range(args.max_eval_samples))

    print("Starting dataset preprocessing...")
    processed_dataset = {
        split: dataset[split].map(
            lambda examples: preprocess_function_batched(examples, tokenizer, args.max_length),
            batched=True,
        )
        for split in ["train", "val"]
    }
    print("Dataset preprocessing finished.")

    collator = DataCollatorForCompletionOnlyLM(instruction_template=INSTRUCTION_TEMPLATE, response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer.tokenizer, mlm=False)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = processed_dataset["train"],
        eval_dataset = processed_dataset["val"], # Can set up evaluation!
        data_collator = collator,
        args = SFTConfig(
            dataset_num_proc = 4,
            dataset_text_field = "prompt",
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps, # Use GA to mimic batch size!
            warmup_ratio = args.warmup_ratio,
            num_train_epochs = args.num_train_epochs, # Set this for 1 full training run.
            learning_rate = args.learning_rate, # Reduce to 2e-5 for long training runs
            logging_steps = args.logging_steps,
            save_strategy = "steps",
            save_steps = args.save_steps,
            eval_strategy = "steps",
            eval_steps = args.eval_steps,
            optim = "adamw_8bit",
            weight_decay = args.weight_decay,
            lr_scheduler_type = "linear",
            seed = args.seed,
            report_to = "wandb", # Use this for WandB etc,
            remove_unused_columns = True,
            save_total_limit = args.save_total_limit,
        ),
    )

    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(f"{args.output_dir}/final")
    wandb.finish()

if __name__ == "__main__":
    main()
