"""
SCRIPT 2: HaluEval QA Evaluation
==================================
Loads HaluEval QA subset, prompts each model, checks whether it produces
the hallucinated answer or the correct one. Saves per-sample results + summary.

Usage:
    python 02_halueval.py

Requirements:
    pip install transformers datasets accelerate torch
"""

import json
import torch
import random
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "llama3.1-8b":    "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b":     "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

NUM_SAMPLES   = 300      # how many HaluEval QA examples to evaluate
MAX_NEW_TOKENS = 100
SEED          = 42
OUTPUT_DIR    = Path("results/halueval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

# ── Prompt template ────────────────────────────────────────────────────────────

def build_prompt(question: str, context: str) -> str:
    """
    Ask the model to answer directly. We then check if the answer
    string-matches the hallucinated answer vs. the correct answer.
    """
    return (
        "Answer the following question based on the given context. "
        "Be factual and concise.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

# ── Load dataset ───────────────────────────────────────────────────────────────

print("Loading HaluEval QA dataset...")
# HaluEval QA: each sample has: question, answer, hallucinated_answer, right_answer
ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
samples = list(ds)
random.shuffle(samples)
samples = samples[:NUM_SAMPLES]
print(f"Using {len(samples)} samples")

# ── Evaluate each model ────────────────────────────────────────────────────────

all_results = {}

for model_name, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    per_sample = []
    correct_count = 0
    hallucinated_count = 0
    neither_count = 0

    for i, sample in enumerate(samples):
        question          = sample["question"]
        right_answer      = sample["right_answer"].strip().lower()
        hallucinated_ans  = sample["hallucinated_answer"].strip().lower()
        # HaluEval QA samples contain a 'knowledge' field as context
        context           = sample.get("knowledge", "")

        prompt = build_prompt(question, context)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip().lower()

        # Simple string-match scoring
        is_correct      = right_answer in generated
        is_hallucinated = hallucinated_ans in generated and not is_correct

        if is_correct:
            label = "correct"
            correct_count += 1
        elif is_hallucinated:
            label = "hallucinated"
            hallucinated_count += 1
        else:
            label = "neither"
            neither_count += 1

        per_sample.append({
            "index":             i,
            "question":          question,
            "right_answer":      right_answer,
            "hallucinated_answer": hallucinated_ans,
            "model_output":      generated,
            "label":             label,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{NUM_SAMPLES}] correct={correct_count} "
                  f"hallucinated={hallucinated_count} neither={neither_count}")

    accuracy     = correct_count / NUM_SAMPLES
    halluc_rate  = hallucinated_count / NUM_SAMPLES

    summary = {
        "model":             model_name,
        "num_samples":       NUM_SAMPLES,
        "correct":           correct_count,
        "hallucinated":      hallucinated_count,
        "neither":           neither_count,
        "accuracy":          round(accuracy, 4),
        "hallucination_rate": round(halluc_rate, 4),
    }
    all_results[model_name] = summary

    print(f"\n✓ {model_name}: accuracy={accuracy:.4f}, hallucination_rate={halluc_rate:.4f}")

    # Save per-sample results
    out_path = OUTPUT_DIR / f"{model_name}_samples.json"
    with open(out_path, "w") as f:
        json.dump(per_sample, f, indent=2)

    # Free VRAM before loading next model
    del model
    torch.cuda.empty_cache()

# ── Final summary ──────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("HALUEVAL RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<25} {'Accuracy':>10} {'Halluc. Rate':>14}")
print("-" * 51)
for model_name, s in all_results.items():
    print(f"{model_name:<25} {s['accuracy']:>10.4f} {s['hallucination_rate']:>14.4f}")

summary_path = OUTPUT_DIR / "summary.json"
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSummary saved to {summary_path}")