import os
import json
import time
import random
import pandas as pd
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from tqdm import tqdm


BASE_URL = "https://api.openai.com/v1"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_NUM_EXAMPLES = 50
RESULT_CSV = "comparison_results.csv"
LORA_DIR = "./drive/MyDrive/Research/cs7650/final_deepseek_model"
BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

def setup_openai_client(api_key_env=OPENAI_KEY_ENV, base_url=BASE_URL):
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env} not set.")
    return OpenAI(base_url=base_url, api_key=api_key)


def evaluate_consistency(client, prompt, response_a, response_b, evaluator_model="gpt-4o-mini"):
    system_msg = """
You are a medical expert evaluator. Do not output any extra fields.
"""
    user_msg = f"""
Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Please reply with **only** a JSON object matching this schema:

{{
  "consistent": "<Yes or No>",
  "consistent_reason": "<brief explanation>",
  // If consistent == Yes, ALSO include:
  "better_model": "<A or B>",
  "better_model_reason": "<brief explanation>"
}}
"""
    comp = client.chat.completions.create(
        model=evaluator_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0.0
    )
    text = comp.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"consistent": None, "consistent_reason": text}

# --- Question Generation ---
FUNCTIONS = [{
  "name": "generate_questions",
  "description": "Generate medical questions.",
  "parameters": {
    "type": "object",
    "properties": {
      "questions": { "type": "array", "minItems": DEFAULT_NUM_EXAMPLES, "maxItems": DEFAULT_NUM_EXAMPLES,
        "items": {"type": "object","properties": {"question": {"type": "string"},"topic": {"type": "string"}},"required": ["question"]}
      }
    },
    "required": ["questions"]
  }
}]

def generate_medical_questions(client, num_questions=DEFAULT_NUM_EXAMPLES):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical educator."},
            {"role": "user",   "content": f"Generate exactly {num_questions} concise, challenging medical questions.Return exactly 50 questions. Do not return more or fewer. One example is What is the description of the medical code 89.43 in ICD9PROC?"}
        ],
        functions=FUNCTIONS,
        function_call={"name": "generate_questions"},
        temperature=0.7
    )
    func_args = json.loads(response.choices[0].message.function_call.arguments)
    return func_args.get("questions", [])


def load_tokenizer_and_model(model_id, device_map="auto", torch_dtype=torch.float16, lora_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True)
    if lora_dir:
        peft_cfg = PeftConfig.from_pretrained(lora_dir)
        model = PeftModel.from_pretrained(model, lora_dir, device_map=device_map, torch_dtype=torch_dtype)
    model.eval()
    return tokenizer, model

def get_model_answer(tokenizer, model, device, question, max_new_tokens=200):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


def vote_on_answers(client, question, ans_a, ans_b, evaluator_model="gpt-4o-mini"):
    vote_prompt = (
        f"Question: {question}\nAnswer A: {ans_a}\nAnswer B: {ans_b}\n"
        "Which answer is more accurate and complete? Reply with 'A', 'B', or 'Tie', then a brief justification."
    )
    resp = client.chat.completions.create(
        model=evaluator_model,
        messages=[{"role": "system", "content": "You are a medical expert evaluator."},
                  {"role": "user",   "content": vote_prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()


def count_votes(df, vote_col='vote'):
    counts = df[vote_col].str.strip().str[0].str.upper().value_counts()
    return {k: int(counts.get(k, 0)) for k in ['A','B','T']}

# --- Main Workflow ---
def main():
    client = setup_openai_client()

    examples = [
        {
            "prompt": "What is the description of the medical code F19.139 in ICD10CM?",
            "response_a": "Other psychoactive substance abuse with withdrawal, unspecified.",
            "response_b": "Other psychoactive substance abuse with withdrawal, unspecified."
        }
    ]
    for ex in examples:
        result = evaluate_consistency(client, ex['prompt'], ex['response_a'], ex['response_b'])
        print("Consistency result:", result)

    questions = generate_medical_questions(client)
    questions = questions[:DEFAULT_NUM_EXAMPLES]

    tokenizer_base, model_base = load_tokenizer_and_model(BASE_MODEL_ID)
    tokenizer_finetuned, model_finetuned = load_tokenizer_and_model(BASE_MODEL_ID, lora_dir=LORA_DIR)
    device = next(model_base.parameters()).device

    results = []
    for q in tqdm(questions, desc="Answering questions"):
        q_text = q['question']
        ans_base = get_model_answer(tokenizer_base, model_base, device, q_text)
        ans_fine = get_model_answer(tokenizer_finetuned, model_finetuned, device, q_text)
        vote = vote_on_answers(client, q_text, ans_base, ans_fine)
        entry = {"question": q_text, "base_answer": ans_base, "fine_tuned_answer": ans_fine, "vote": vote}
        if 'topic' in q:
            entry['topic'] = q['topic']
        results.append(entry)
        time.sleep(1)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULT_CSV, index=False)
    print(f"Saved comparison results to {RESULT_CSV}")
    print("Vote summary:", count_votes(df))

if __name__ == "__main__":
    main()
