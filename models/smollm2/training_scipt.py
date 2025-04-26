from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch
from peft import LoraConfig, TaskType
from datasets import load_dataset, Dataset
import json
import shutil

def tokenize(sample):
    full_prompt = sample["prompt"]
    full_output = sample["output"]

    full_text = full_prompt + "\n\n" + full_output

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

USE_4BIT = True
COMPUTE_DTYPE = "float16"
QUANTIZATION_TYPE = "nf4"
USE_NESTED_QUANTIZATION = False

bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_quant_type=QUANTIZATION_TYPE,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=USE_NESTED_QUANTIZATION,
)

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
).to("cuda")

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("Malikeh1375/medical-question-answering-datasets", name="all-processed", split="train")
formatted_data = []
for item in dataset:
    prompt = f"{item['instruction']}\n\n{item['input']}"
    output = item['output']
    formatted_data.append({
        "prompt": prompt,
        "output": output
    })

train_dataset = Dataset.from_list(formatted_data)
tokenized_dataset = train_dataset.map(tokenize, remove_columns=["prompt", "output"])
subset_dataset = tokenized_dataset.shuffle(seed=42).select(range(30000))

training_args = TrainingArguments(
    output_dir="./smol_lora_ckpt",
    per_device_train_batch_size=12,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=5,
    save_steps=4000,
    logging_steps=400,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=subset_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./final_smol_model")