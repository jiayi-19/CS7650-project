import json
import time
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

# Load data (prompt+response format)
def load_jsonl_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

def tokenize_prompt_response(example):
    full_text = example["prompt"] + example["response"]
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

data_path = "data/train_prompt_response.jsonl"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
output_dir = "output_llama1b_bf16_subset"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_jsonl_dataset(data_path)
dataset = dataset.train_test_split(test_size=0.27, seed=42)["train"] 
dataset = dataset.map(tokenize_prompt_response, remove_columns=["prompt", "response"])

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    warmup_steps=200,
    bf16=True,
    logging_steps=200,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
# trainer.train(resume_from_checkpoint=True)

print(f"\nComplete.")

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
