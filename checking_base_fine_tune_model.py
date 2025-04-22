import os
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from openai import OpenAI

# --- Configuration ---
# OpenAI API key (set this in your environment or replace directly)
client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key="os.environ['OPENAI_API_KEY']"
)

# Paths & model IDs\BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

NUM_EXAMPLES = 5
RESULT_CSV = "comparison_results.csv"
BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# --- Load Tokenizer ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    use_fast=True,
    trust_remote_code=True
)

# --- Load Pure Base Model ---
print("Loading pure base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
base_model.eval()
device = next(base_model.parameters()).device

# --- Load Fine-Tuned Model (Separate Instance) ---
print("Loading fine-tuned model with LoRA adapters...")
# Load a fresh copy of the base model
finetuned_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
# Apply LoRA adapters
config = PeftConfig.from_pretrained(LORA_DIR)
finetuned_model = PeftModel.from_pretrained(
    finetuned_model,
    LORA_DIR,
    device_map="auto",
    torch_dtype=torch.float16
)
finetuned_model.eval()


# --- Helper Functions ---
def generate_medical_question():
    """Use GPT-4 to generate a medical question."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical educator. Generate a concise, challenging medical question. One example is What is the description of the medical code 89.43 in ICD9PROC?"},
            {"role": "user", "content": "Please provide one medical question."}
        ],
        temperature=0.7,
        max_tokens=100
    )
    return resp.choices[0].message.content.strip()


def get_model_answer(model, question):
    """Generate an answer from a local HF model."""
    # Simple prompt formatting
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip prompt
    return text[len(prompt):].strip()


def vote_on_answers(question, ans_a, ans_b):
    """Use GPT-4 to pick the better answer A or B."""
    vote_prompt = (
        f"Question: {question}\n"
        f"Answer A: {ans_a}\n\n"
        f"Answer B: {ans_b}\n\n"
        "Which answer is more accurate and complete? Reply with 'A', 'B', or 'Tie', then a brief justification."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert evaluator comparing two model answers."},
            {"role": "user", "content": vote_prompt}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# --- Main Loop ---
results = []
for i in tqdm(range(NUM_EXAMPLES), desc="Running evaluations"):
    q = generate_medical_question()
    a_base = get_model_answer(base_model, q)
    a_fine = get_model_answer(finetuned_model, q)
    vote = vote_on_answers(q, a_base, a_fine)
    results.append({
        "question": q,
        "base_answer": a_base,
        "fine_tuned_answer": a_fine,
        "vote": vote
    })
    time.sleep(1)  # avoid hitting rate limits

# --- Save Results ---
df = pd.DataFrame(results)
df.to_csv(RESULT_CSV, index=False)
print(f"Completed {NUM_EXAMPLES} evaluations. Results saved to {RESULT_CSV}")
