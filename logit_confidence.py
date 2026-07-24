"""
SCRIPT 3: Token-Level Confidence Analysis via Logits
======================================================
For each model, generates answers on HaluEval QA samples and extracts
token-level softmax probabilities. Computes:
  - Mean token confidence for CORRECT answers
  - Mean token confidence for HALLUCINATED answers
  - Entropy of token distributions (lower entropy = more confident)

This tests the overconfidence hypothesis: do models assign high confidence
even when generating hallucinated content?

Usage:
    python 03_logit_confidence.py

Requirements:
    pip install transformers datasets torch matplotlib seaborn
"""

import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "llama3.1-8b":    "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b":     "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

NUM_SAMPLES    = 200
MAX_NEW_TOKENS = 80
SEED           = 42
OUTPUT_DIR     = Path("results/logits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


# ── Helpers ────────────────────────────────────────────────────────────────────

def token_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy per token position. Lower = more confident."""
    # probs: [seq_len, vocab_size]
    eps = 1e-9
    return -(probs * (probs + eps).log()).sum(dim=-1)


def build_prompt(question: str, context: str) -> str:
    return (
        "Answer the following question based on the given context. "
        "Be factual and concise.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


# ── Load dataset ───────────────────────────────────────────────────────────────

print("Loading HaluEval QA dataset...")
ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
samples = list(ds)
random.shuffle(samples)
samples = samples[:NUM_SAMPLES]

# ── Evaluate ───────────────────────────────────────────────────────────────────

all_model_stats = {}

for model_name, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Extracting logits: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Per-sample stats
    correct_confidences    = []  # list of mean top-1 prob for correct samples
    hallucinated_confidences = []
    correct_entropies      = []
    hallucinated_entropies = []

    per_sample_records = []

    for i, sample in enumerate(samples):
        question         = sample["question"]
        right_answer     = sample["right_answer"].strip().lower()
        hallucinated_ans = sample["hallucinated_answer"].strip().lower()
        context          = sample.get("knowledge", "")

        prompt = build_prompt(question, context)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,          # ← get logits at each generation step
            )

        # output.scores: tuple of (vocab_size,) tensors, one per generated token
        scores = torch.stack(output.scores, dim=0)          # [gen_len, vocab]
        probs  = torch.softmax(scores.float(), dim=-1)       # [gen_len, vocab]

        # Top-1 probability at each token (model's "confidence")
        top1_probs = probs.max(dim=-1).values               # [gen_len]
        entropies  = token_entropy(probs)                    # [gen_len]

        mean_conf    = top1_probs.mean().item()
        mean_entropy = entropies.mean().item()

        # Decode generated text
        gen_ids = output.sequences[0][prompt_len:]
        generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

        # Label
        is_correct      = right_answer in generated
        is_hallucinated = hallucinated_ans in generated and not is_correct

        if is_correct:
            label = "correct"
            correct_confidences.append(mean_conf)
            correct_entropies.append(mean_entropy)
        elif is_hallucinated:
            label = "hallucinated"
            hallucinated_confidences.append(mean_conf)
            hallucinated_entropies.append(mean_entropy)
        else:
            label = "neither"

        per_sample_records.append({
            "index":          i,
            "label":          label,
            "mean_confidence": round(mean_conf, 6),
            "mean_entropy":    round(mean_entropy, 6),
            "generated":       generated,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{NUM_SAMPLES}] "
                  f"correct_conf={np.mean(correct_confidences):.4f} "
                  f"halluc_conf={np.mean(hallucinated_confidences) if hallucinated_confidences else 0:.4f}")

    # Summary stats
    stats = {
        "model": model_name,
        "correct_samples":       len(correct_confidences),
        "hallucinated_samples":  len(hallucinated_confidences),
        "correct_mean_conf":     round(float(np.mean(correct_confidences)), 6)    if correct_confidences else None,
        "halluc_mean_conf":      round(float(np.mean(hallucinated_confidences)), 6) if hallucinated_confidences else None,
        "correct_mean_entropy":  round(float(np.mean(correct_entropies)), 6)      if correct_entropies else None,
        "halluc_mean_entropy":   round(float(np.mean(hallucinated_entropies)), 6) if hallucinated_entropies else None,
        # Overconfidence gap: positive = model MORE confident when hallucinating
        "overconfidence_gap":    round(
            float(np.mean(hallucinated_confidences)) - float(np.mean(correct_confidences)), 6
        ) if (correct_confidences and hallucinated_confidences) else None,
    }
    all_model_stats[model_name] = {
        "summary": stats,
        "correct_confidences":     correct_confidences,
        "hallucinated_confidences": hallucinated_confidences,
    }

    print(f"\n✓ {model_name}")
    print(f"  Correct mean conf:     {stats['correct_mean_conf']}")
    print(f"  Hallucinated mean conf:{stats['halluc_mean_conf']}")
    print(f"  Overconfidence gap:    {stats['overconfidence_gap']}")

    # Save per-sample records
    out_path = OUTPUT_DIR / f"{model_name}_logits.json"
    with open(out_path, "w") as f:
        json.dump(per_sample_records, f, indent=2)

    del model
    torch.cuda.empty_cache()


# ── Save summary ───────────────────────────────────────────────────────────────

summary_data = {k: v["summary"] for k, v in all_model_stats.items()}
with open(OUTPUT_DIR / "confidence_summary.json", "w") as f:
    json.dump(summary_data, f, indent=2)


# ── Plot: confidence distribution by label, per model ─────────────────────────

fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 5), sharey=False)
if len(MODELS) == 1:
    axes = [axes]

for ax, (model_name, data) in zip(axes, all_model_stats.items()):
    c_conf = data["correct_confidences"]
    h_conf = data["hallucinated_confidences"]

    if c_conf:
        ax.hist(c_conf, bins=20, alpha=0.6, color="steelblue", label="Correct")
    if h_conf:
        ax.hist(h_conf, bins=20, alpha=0.6, color="tomato",    label="Hallucinated")

    ax.set_title(model_name, fontsize=11)
    ax.set_xlabel("Mean Token Confidence (top-1 prob)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.axvline(np.mean(c_conf) if c_conf else 0,  color="steelblue", linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(h_conf) if h_conf else 0,  color="tomato",    linestyle="--", linewidth=1.5)

plt.suptitle("Token-Level Confidence: Correct vs Hallucinated Outputs", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confidence_distribution.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_DIR}/confidence_distribution.png")


# ── Plot: overconfidence gap bar chart ────────────────────────────────────────

model_names = list(summary_data.keys())
gaps = [summary_data[m]["overconfidence_gap"] or 0 for m in model_names]
colors = ["tomato" if g > 0 else "steelblue" for g in gaps]

fig2, ax2 = plt.subplots(figsize=(7, 4))
bars = ax2.bar(model_names, gaps, color=colors, edgecolor="black", width=0.5)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("Overconfidence Gap\n(Hallucinated conf − Correct conf)")
ax2.set_title("Overconfidence Gap per Model\n(positive = more confident when hallucinating)")
for bar, gap in zip(bars, gaps):
    ax2.text(bar.get_x() + bar.get_width() / 2, gap + 0.001,
             f"{gap:+.4f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "overconfidence_gap.png", dpi=150, bbox_inches="tight")
print(f"Plot saved to {OUTPUT_DIR}/overconfidence_gap.png")

print("\nDone. All results in results/logits/")