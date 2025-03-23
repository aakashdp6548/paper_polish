import argparse
import wandb
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModel,
    PreTrainedModel,
    TrainingArguments, 
    Trainer, 
    EvalPrediction
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class Gemma3ForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # Load the base Gemma3 model
        self.gemma = AutoModel.from_pretrained(config._name_or_path)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.post_init()
        
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
                mask = attention_mask.unsqueeze(-1).expand(pooled_output.size()).float()
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
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.config.problem_type == "single_label_classification":
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

def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def preprocess_function(examples, tokenizer, max_length):
    return tokenizer(examples["model_input"], truncation=True, padding="max_length", max_length=max_length)

def prepare_labels(dataset, multi_class):
    if multi_class:
        # Create mapping for all unique status values
        unique_statuses = set(dataset["train"]["result"])
        for split in ["validation", "test"]:
            if split in dataset:
                unique_statuses.update(dataset[split]["result"])
        label_mapping = {status: idx for idx, status in enumerate(sorted(unique_statuses))}
    else:
        # Binary classification: accept vs reject
        accept_categories = ["poster", "spotlight", "oral"]
        label_mapping = lambda x: 1 if x.lower() in accept_categories else 0
    
    for split in dataset:
        if multi_class:
            dataset[split] = dataset[split].map(lambda x: {"labels": label_mapping[x["result"]]})
        else:
            dataset[split] = dataset[split].map(lambda x: {"labels": label_mapping(x["result"])})
    
    num_labels = len(label_mapping) if multi_class else 2
    return dataset, num_labels

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for paper acceptance prediction")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it", help="HuggingFace model name")
    parser.add_argument("--dataset_path", type=str, required=True, help="HuggingFace dataset path")
    parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, required=True, help="WandB run name")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--multi_class", action="store_true", help="Use multi-class classification")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--use_gemma3", action="store_true", default=True, help="Use Gemma3 model")
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Process labels and get number of classes
    dataset, num_labels = prepare_labels(dataset, args.multi_class)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Tokenize datasets
    tokenized_dataset = {
        split: dataset[split].map(
            lambda examples: preprocess_function(examples, tokenizer, args.max_length),
            batched=True
        )
        for split in dataset
    }
    
    # Load model
    if args.use_gemma3:
        # For Gemma3, use our custom class
        # First load the config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_name)
        config.num_labels = num_labels
        
        # Then create our custom model
        model = Gemma3ForSequenceClassification.from_pretrained(
            args.model_name,
            config=config
        )
    else:
        # For other models, use standard sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=num_labels
        )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=10,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    trainer.train()
    
    # Evaluate on test set if available
    if "test" in tokenized_dataset:
        results = trainer.evaluate(tokenized_dataset["test"])
        print(f"Test results: {results}")
        wandb.log({"test": results})
    
    # Save model
    trainer.save_model(f"{args.output_dir}/final")
    
    # End wandb run
    wandb.finish()

if __name__ == "__main__":
    main()