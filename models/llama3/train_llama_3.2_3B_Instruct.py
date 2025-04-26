import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

import os
print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

def load_jsonl_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

def build_prompt(example):
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()
    output = example["output"].strip()

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

data_path = "data/train_medqa_raw.jsonl"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
output_dir = "output"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_jsonl_dataset(data_path).map(build_prompt, remove_columns=["instruction", "input", "output"])

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)
model.config.torch_dtype = torch.float16

peft_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    # max_steps=20000,
    num_train_epochs=4,
    learning_rate=4e-5,
    lr_scheduler_type="linear",
    warmup_steps=100,
    fp16=True,
    logging_steps=20,
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

trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
