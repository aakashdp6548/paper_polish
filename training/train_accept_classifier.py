import argparse
import wandb
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    TrainingArguments, 
    Trainer, 
    EvalPrediction,
    TrainerCallback
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR  # Import from the correct module
from peft import LoraConfig, PeftModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BINARY_LABELS = {"reject": 0, "poster": 1, "spotlight": 1, "oral": 1}
MULTI_LABELS = {"reject": 0, "poster": 1, "spotlight": 2, "oral": 3}

class Gemma3ForSequenceClassification(PreTrainedModel):
    def __init__(self, config, bfloat16=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bf16 = bfloat16
        
        # Load the base Gemma3 model
        self.gemma = AutoModel.from_pretrained(config._name_or_path)
        self.gemma = self.gemma.to(torch.bfloat16 if bfloat16 else torch.float32)
        
        lora_config = LoraConfig(
            r=8,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        self.gemma = PeftModel(self.gemma, lora_config)
        self.gemma = self.gemma.to(torch.bfloat16 if bfloat16 else torch.float32)

        # Create classifier with same dtype
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = self.classifier.to(torch.bfloat16 if bfloat16 else torch.float32)

        # Initialize weights
        self.post_init()

        print("Model parameters:")
        for name, param in self.gemma.named_parameters():
            print(name, param.dtype)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get the outputs from the base model
        outputs = self.gemma(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the sequence representation (last hidden state of [CLS] token or mean pooling)
        if hasattr(outputs, "last_hidden_state"):
            # Use mean pooling over the sequence
            pooled_output = outputs.last_hidden_state
            if attention_mask is not None:
                # Create a mask of shape [batch_size, seq_length, 1]
                # Ensure mask has the same dtype as pooled_output
                mask = attention_mask.unsqueeze(-1).expand(pooled_output.size()).to(pooled_output.dtype)
                # Apply mask and average
                pooled_output = torch.sum(pooled_output * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            else:
                pooled_output = pooled_output.mean(dim=1)
        else:
            # Fall back to the first token if the model doesn't output last_hidden_state
            pooled_output = outputs[0][:, 0]

        # Apply classification head
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    def save_model(self, save_directory):
        """Save the model, LoRA weights, and classifier in a way that guarantees consistent loading"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the config
        self.config.save_pretrained(save_directory)

        # 1. Explicitly save the LoRA adapter weights using torch.save
        adapter_weights = {}
        for name, param in self.gemma.named_parameters():
            if param.requires_grad:  # Only save trainable parameters (LoRA)
                adapter_weights[name] = param.data.cpu().clone()
        
        adapter_path = os.path.join(save_directory, "lora_weights.pt")
        torch.save(adapter_weights, adapter_path)
        
        # 2. Also save the adapter config
        if hasattr(self.gemma, "peft_config"):
            peft_config_path = os.path.join(save_directory, "peft_config.json")
            self.gemma.peft_config["default"].save_pretrained(peft_config_path)
        
        # 3. Save the classifier weights
        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save(self.classifier.state_dict(), classifier_path)
        
        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_path, *args, **kwargs):
        """Load the model with a reliable method to ensure consistent results"""
        # Initialize config
        config = kwargs.get('config', AutoConfig.from_pretrained(pretrained_path))
        
        # Set the base model path to the original model
        base_model_name = config._name_or_path
        print(f"Using base model: {base_model_name}")
        
        # Force deterministic behavior
        torch.manual_seed(42)
        
        # Create a new instance with the base model - EXPLICITLY load the base model first
        bf16 = kwargs.get('bf16', False)
        base_model = AutoModel.from_pretrained(base_model_name)
        
        # Convert base model to correct dtype if needed
        if bf16:
            base_model = base_model.to(torch.bfloat16)
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=8,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        
        # Create PeftModel
        peft_model = PeftModel(base_model, lora_config)
        
        # Create our model container
        model = cls.__new__(cls)
        PreTrainedModel.__init__(model, config)
        model.num_labels = config.num_labels
        model.config = config
        model.bf16 = bf16
        model.gemma = peft_model
        
        # Create classifier
        model.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Now load the LoRA weights if available
        lora_weights_path = os.path.join(pretrained_path, "lora_weights.pt")
        if os.path.exists(lora_weights_path):
            print(f"Loading custom LoRA weights from {lora_weights_path}")
            
            # Load the saved LoRA weights
            adapter_weights = torch.load(lora_weights_path, weights_only=True)
            
            # Apply weights to the model
            for name, param in model.gemma.named_parameters():
                if name in adapter_weights and param.requires_grad:
                    param.data.copy_(adapter_weights[name])
                    # print(f"Loaded parameter: {name}")

            # Freeze/unfreeze parameters as needed
            model.gemma.requires_grad_(False)
            for name, param in model.gemma.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
        
        # Load classifier weights
        classifier_path = os.path.join(pretrained_path, "classifier.pt")
        if os.path.exists(classifier_path):
            print(f"Loading classifier weights from {classifier_path}")
            model.classifier.load_state_dict(torch.load(classifier_path, weights_only=True))

        # Initialize weights for consistency
        model.post_init()

        model.gemma = model.gemma.to(torch.bfloat16 if bf16 else torch.float32)
        model.classifier = model.classifier.to(torch.bfloat16 if bf16 else torch.float32)
        
        return model

def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def preprocess_function(examples, tokenizer, max_length):
    pre_prompt = "You are a reviewer for a top machine learning conference. You are given a paper and your job is to determine whether the paper should be accepted or rejected. The paper is:\n"
    post_prompt = "\n\nPlease respond with a single word: 'accept' or 'reject'."
    
    # Process each example independently
    input_ids_list = []
    attention_mask_list = []
    
    for paper in examples["model_input"]:
        # First tokenize the prompts to know their length
        pre_tokens = tokenizer(pre_prompt, add_special_tokens=False)
        post_tokens = tokenizer(post_prompt, add_special_tokens=True)
        
        # Calculate remaining length for the paper content
        prompt_length = len(pre_tokens["input_ids"]) + len(post_tokens["input_ids"])
        max_paper_length = max_length - prompt_length
        
        # Tokenize and truncate the paper content
        paper_tokens = tokenizer(paper, truncation=True, max_length=max_paper_length, add_special_tokens=False)
        
        # Combine all parts
        combined_input_ids = pre_tokens["input_ids"] + paper_tokens["input_ids"] + post_tokens["input_ids"]
        combined_attention_mask = [1] * len(combined_input_ids)
        
        # Pad if necessary
        if len(combined_input_ids) < max_length:
            padding_length = max_length - len(combined_input_ids)
            combined_input_ids.extend([tokenizer.pad_token_id] * padding_length)
            combined_attention_mask.extend([0] * padding_length)
        
        input_ids_list.append(combined_input_ids)
        attention_mask_list.append(combined_attention_mask)
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list
    }

def prepare_labels(dataset, multi_class):
    label_mapping = MULTI_LABELS if multi_class else BINARY_LABELS
    for split in dataset:
        if split == "test":
            continue
        dataset[split] = dataset[split].map(lambda x: {"labels": label_mapping.get(x["result"], label_mapping["reject"])})

    num_labels = len(set(label_mapping.values()))
    return dataset, num_labels

def balance_dataset(dataset):
    for split in dataset:
        if split == "test":
            continue
        # Count samples per label
        label_counts = {}
        for example in dataset[split]:
            label = example["labels"]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

        # Find the majority class count
        majority_class_count = max(label_counts.values())
        print(f"Balancing {split} dataset. Label counts before balancing: {label_counts}")
        
        # Balance each minority class
        balanced_examples = []
        for label, count in label_counts.items():
            # Get all examples with this label
            label_examples = [ex for ex in dataset[split] if ex["labels"] == label]
            
            # If this is a minority class, sample additional examples
            if count < majority_class_count:
                # Calculate how many more examples we need
                samples_needed = majority_class_count - count
                
                # Sample with replacement from the existing examples
                indices = np.random.choice(len(label_examples), size=samples_needed, replace=True)
                additional_samples = [label_examples[i] for i in indices]
                
                # Add all examples for this class
                balanced_examples.extend(label_examples + additional_samples)
            else:
                # For majority class, just add all examples
                balanced_examples.extend(label_examples)
        
        # Replace the split with balanced data
        dataset[split] = dataset[split].from_list(balanced_examples)
    
        # Verify the balancing worked
        new_label_counts = {}
        for example in dataset[split]:
            label = example["labels"]
            if label not in new_label_counts:
                new_label_counts[label] = 0
            new_label_counts[label] += 1
        
        print(f"Label counts in {split} after balancing: {new_label_counts}")

    return dataset

class CustomSaveCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model
    
    def on_save(self, args, state, control, **kwargs):
        """Called when the trainer is saving a checkpoint."""
        checkpoint_folder = f"{args.output_dir}/checkpoint-{state.global_step}"
        self.model.save_model(checkpoint_folder)
        
        # Set should_save to False to prevent default saving
        control.should_save = False
        
        # Return the modified control object
        return control

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for paper acceptance prediction")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it", help="HuggingFace model name")
    parser.add_argument("--dataset_path", type=str, required=True, help="HuggingFace dataset path")
    parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, required=True, help="WandB run name")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=16384, help="Max sequence length")
    parser.add_argument("--multi_class", action="store_true", help="Use multi-class classification")
    parser.add_argument("--balance_dataset", action="store_true", help="Balance the dataset")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max number of validation samples")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    args = parser.parse_args()

    # Initialize wandb
    # wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # Process labels and get number of classes
    print(f"Processing labels and getting number of classes")
    dataset, num_labels = prepare_labels(dataset, args.multi_class)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Tokenize datasets
    tokenized_dataset = {}
    for split in dataset:
        if split == "test": # skip test set
            continue
        tokenized_dataset[split] = dataset[split].map(
            lambda examples: preprocess_function(examples, tokenizer, args.max_length),
            batched=True
        )

    if args.balance_dataset:
        print(f"Balancing dataset")
        dataset = balance_dataset(dataset)
    
    if args.max_val_samples is not None:
        print(f"Subsampling validation set to {args.max_val_samples} samples")
        dataset["val"] = dataset["val"].shuffle(seed=42).select(range(args.max_val_samples))

    return

    # Load model with the correct dtype from the beginning
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = num_labels
    
    # Set dtype before model creation
    model = Gemma3ForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        bf16=args.bf16
    )
    
    # Move to device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set up training arguments
    output_dir = f"{args.output_dir}/{args.wandb_run_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=10,
    )

    # Initialize Trainer with our custom callback
    custom_callback = CustomSaveCallback(model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"] if "val" in tokenized_dataset else None,
        compute_metrics=compute_metrics,
        callbacks=[custom_callback],
    )
    
    # Train model
    print("Training model...")
    trainer.train()

    # Save final model
    model.save_model(f"{args.output_dir}/final")
    
    # End wandb run
    wandb.finish()

if __name__ == "__main__":
    main()