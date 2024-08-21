from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
import json
import os
import time
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

start_time = time.time()

# Enable CUDA Launch Blocking for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i)}")
        print(f"Memory Cached: {torch.cuda.memory_reserved(i)}")
else:
    print("CUDA is not available. Exiting.")
    exit()

# Define the model name
model_name = "t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# Load data
data_path = 'cleaned_train_cm1.json'
with open(data_path, 'r') as f:
    data = json.load(f)

data_df = pd.json_normalize(data)

# Prepare dataset
def process_data_to_dataset(dataframe):
    dataset = Dataset.from_pandas(dataframe)
    return dataset

full_dataset = process_data_to_dataset(data_df)

# Define output keys
output_keys = ["Task Achievement", "Coherence and Cohesion", "Vocabulary", "Grammar", "Overall", "Feedback"]

# Initialize tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Define the function to tokenize and encode examples
# Define the function to tokenize and encode examples
def tokenize_and_encode(examples):
    combined_texts = ["Generate structured feedback for the following essay: " + inst + " " + inp for inst, inp in zip(examples['instruction'], examples['input'])]
    tokenized_inputs = tokenizer(combined_texts, truncation=True, padding='max_length', max_length=512)
    
    output_data = ["[STRUCTURED FEEDBACK]: " + output for output in examples['output']]
    output_data = [output.split(';') for output in output_data]  # Split by ';' to correctly segment each part
    labels = []
    for outputs in output_data:
        example_labels = []
        for key in output_keys:
            found = False
            for comment in outputs:
                if comment.strip().startswith(key):
                    try:
                        # Extract the comment after the key
                        extracted_comment = comment.split(":", 1)[1].strip()  # split only at the first occurrence of ":"
                        example_labels.append(f"[{key}]: {extracted_comment}")
                        found = True
                        break
                    except (ValueError, IndexError):
                        continue
            if not found:
                example_labels.append(f"[{key}]: ")  # Default or placeholder value for missing items
        labels.append(" ".join(example_labels))  # Join all labels into a single string

    # Tokenize the labels using the __call__ method for fast tokenization and padding
    tokenized_labels = tokenizer(labels, truncation=True, padding='max_length', max_length=512)

    # Combine tokenized inputs and labels
    tokenized_inputs['labels'] = tokenized_labels['input_ids']
    tokenized_inputs['text'] = combined_texts

    return tokenized_inputs

# Tokenize and encode the dataset
print("Tokenizing and encoding the dataset...")
tokenized_dataset = full_dataset.map(tokenize_and_encode, batched=True)



# Split the dataset into training, validation, and testing sets if not already done
print("Splitting the dataset...")
train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)  # 10% for testing
train_val_split = train_test_split['train'].train_test_split(test_size=0.1, seed=42)  # 10% of the remaining 90% for validation

# Create a DatasetDict to hold the splits
dataset_dict = DatasetDict({
    'train': train_val_split['train'],
    'validation': train_val_split['test'],
    'test': train_test_split['test']
})

# Verify data splits
print("Verifying data splits...")
for split in ['train', 'validation', 'test']:
    print(f"{split} dataset: {len(dataset_dict[split])} samples")
    print(f"Sample data from {split} dataset:")
    print(dataset_dict[split][0])

# Ensure there are no empty batches
def check_empty_batches(dataset):
    for split in ['train', 'validation', 'test']:
        for i, batch in enumerate(dataset[split]):
            if len(batch['input_ids']) == 0:
                print(f"Empty batch found in {split} at index {i}")
                return False
    return True

if not check_empty_batches(dataset_dict):
    raise ValueError("Empty batches found in the dataset.")




import torch
import torch.nn.functional as F

def custom_loss_fn(logits, labels, ignore_index=-100):
    # Define your structural tokens (e.g., category headers)
    structural_tokens = tokenizer.encode(
        "Task Achievement: Coherence and Cohesion: Vocabulary: Grammar: Overall: Feedback:",
        add_special_tokens=False
    )
    
    # Create a weight tensor
    weights = torch.ones_like(labels, dtype=torch.float)
    
    # Increase weight for structural tokens
    for token in structural_tokens:
        weights[labels == token] = 5.0  # You can adjust this weight
    
    # Calculate loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction='none'
    )
    
    # Apply weights
    weighted_loss = loss * weights.view(-1)
    
    return weighted_loss.mean()

from transformers import T5ForConditionalGeneration

class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
        
        if labels is not None:
            loss = custom_loss_fn(outputs.logits, labels)
            outputs.loss = loss

        return outputs

# Load model
print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(model_name)

# PEFT Configuration
#lora_alpha = 16
#lora_dropout = 0.1
#lora_r = 64

#peft_config = LoraConfig(
#    lora_alpha=lora_alpha,
#    lora_dropout=lora_dropout,
#    r=lora_r,
#    bias="none",
#    task_type="SEQ2SEQ_LM",  # Use SEQ2SEQ_LM for T5
#)

# Define a configuration for T5 (assuming similar structure to PrefixTuningConfig)
#class PrefixTuningConfig(T5Config):
#    def __init__(self, prefix_length=10, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        self.prefix_length = prefix_length
#        self.decoder_start_token_id = T5Tokenizer.from_pretrained("t5-base").pad_token_id  # Typically set to pad_token_id
#        self.bos_token_id = T5Tokenizer.from_pretrained("t5-base").bos_token_id  # Typically set to bos_token_id

from peft import PrefixTuningConfig
peft_config = PrefixTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    prefix_projection=True,
)
# Wrap the model with PEFT configuration
print("Applying PEFT configuration...")
model = PeftModel(model, peft_config)

# Data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir="./results_t5",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # Increase effective batch size
    fp16=True,  # Enable mixed precision training
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_t5",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataloader_pin_memory=True,
    group_by_length=True,
)

# Initialize Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['validation'],
    data_collator=data_collator,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model and tokenizer
print("Saving the model and tokenizer...")
model.save_pretrained("./model_t5/")
tokenizer.save_pretrained("./model_t5/")


end_time = time.time()

# Calculate the duration
duration = end_time - start_time

# Print the duration in a readable format
print(f"Training completed in {duration // 3600} hours, {(duration % 3600) // 60} minutes, and {duration % 60:.2f} seconds.")


