"""
SCRIPT 4: CrossLingual Hallucination Analysis Qwen English vs Chinese
=========================================================================
Takes factual QA pairs, translates them to Chinese (using Helsinki-NLP/opusmt),
then compares Qwen2.5's accuracy + token confidence in both languages.

Key findings to look for:
  - Accuracy gap (EN vs ZH): is Qwen more/less accurate in Chinese?
  - Confidence gap: is it more confident in one language even if less accurate?
  - Overconfidence in ZH: high confidence + wrong answer (dangerous hallucinations)

Usage:
    python 04_crosslingual.py

Requirements:
    pip install transformers datasets torch sentencepiece sacrebleu matplotlib
"""

import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

# -- Config --------------------------------------------------------------------

QWEN_MODEL     = "Qwen/Qwen2.5-7B-Instruct"
TRANS_MODEL    = "Helsinki-NLP/opus-mt-en-zh"
NUM_SAMPLES    = 150
MAX_NEW_TOKENS = 80
SEED           = 42
OUTPUT_DIR     = Path("results/crosslingual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

# FIX: determine device safely (guards against broken CUDA init)
def safe_device():
    try:
        if torch.cuda.is_available():
            _ = torch.zeros(1).cuda()   # probe  raises if driver mismatch
            return "cuda"
    except Exception:
        pass
    return "cpu"

DEVICE = safe_device()
print(f"Using device: {DEVICE}")

# -- Helpers -------------------------------------------------------------------

def extract_answers(sample):
    """
    Safely extract (right_answer, hallucinated_answer) from a HaluEval QA sample.
    Guards against None values that crash .strip().
    """
    right = (
        sample.get("right_answer")
        or sample.get("answer")
        or ""
    )
    halluc = (
        sample.get("hallucinated_answer")
        or sample.get("hallucinated_ans")
        or ""
    )
    return right.strip().lower(), halluc.strip().lower()


def has_valid_answers(sample):
    r, h = extract_answers(sample)
    return bool(r) and bool(h)


def en_prompt(question: str, context: str) -> str:
    return (
        "Answer the following question based on the given context. "
        "Be factual and concise.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def zh_prompt(question_zh: str, context_zh: str) -> str:
    return (
        "??????????????????????\n\n"
        f"??:{context_zh}\n\n"
        f"??:{question_zh}\n\n"
        "??:"
    )

# -- Load & filter dataset -----------------------------------------------------

print("Loading HaluEval QA dataset...")
ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
samples = list(ds)
random.shuffle(samples)

# FIX: filter out rows where either answer field is None/empty
samples = [s for s in samples if has_valid_answers(s)]
samples = samples[:NUM_SAMPLES]
print(f"Using {len(samples)} valid samples (after filtering None answers)")

# -- Load translation model ----------------------------------------------------

print(f"\nLoading translation model: {TRANS_MODEL}")
trans_tokenizer = AutoTokenizer.from_pretrained(TRANS_MODEL)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(TRANS_MODEL).to(DEVICE)  # FIX: use safe DEVICE


def translate(text: str) -> str:
    """Translate English text to Chinese using Helsinki NLP Marian model."""
    if not text or not text.strip():
        return ""
    words = text.split()
    if len(words) > 200:
        text = " ".join(words[:200])
    try:
        inputs = trans_tokenizer(text, return_tensors="pt").to(trans_model.device)
        with torch.no_grad():
            generated_ids = trans_model.generate(**inputs, max_length=512)
        return trans_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"  [WARN] Translation failed: {e}")
        return text  # fallback to original English

# -- Pretranslate all samples -------------------------------------------------

print(f"\nTranslating {len(samples)} samples to Chinese...")
for i, sample in enumerate(samples):
    right_answer, hallucinated_ans = extract_answers(sample)   # FIX: safe extraction
    sample["question_zh"]         = translate(sample["question"])
    sample["context_zh"]          = translate(sample.get("knowledge", ""))
    sample["right_answer_zh"]     = translate(right_answer)
    sample["hallucinated_ans_zh"] = translate(hallucinated_ans)
    if (i + 1) % 30 == 0:
        print(f"  Translated {i+1}/{len(samples)}")

# Free translation model
del trans_model, trans_tokenizer
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# -- Load Qwen -----------------------------------------------------------------

print(f"\nLoading Qwen model: {QWEN_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    dtype=torch.bfloat16,       # FIX: use dtype= not torch_dtype=
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# -- Inference helper ----------------------------------------------------------

def run_inference(prompt: str):
    """Returns (generated_text, mean_top1_confidence, mean_entropy)."""
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    scores     = torch.stack(output.scores, dim=0)
    probs      = torch.softmax(scores.float(), dim=-1)
    top1_probs = probs.max(dim=-1).values.mean().item()
    entropy    = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean().item()

    gen_ids   = output.sequences[0][prompt_len:]
    generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
    return generated, top1_probs, entropy

# -- Run evaluation ------------------------------------------------------------

per_sample = []

en_correct = 0;  en_halluc = 0
zh_correct = 0;  zh_halluc = 0

en_conf_correct = [];  en_conf_halluc = []
zh_conf_correct = [];  zh_conf_halluc = []

zh_high_conf_wrong = 0
en_high_conf_wrong = 0
CONF_THRESHOLD     = 0.75

print(f"\nRunning inference ({len(samples)} samples × 2 languages)...")

for i, sample in enumerate(samples):
    right_answer, hallucinated_ans = extract_answers(sample)   # FIX: safe extraction
    right_answer_zh = sample["right_answer_zh"].strip().lower()
    halluc_ans_zh   = sample["hallucinated_ans_zh"].strip().lower()

    # -- English inference
    p_en = en_prompt(sample["question"], sample.get("knowledge", ""))
    gen_en, conf_en, ent_en = run_inference(p_en)

    is_correct_en = right_answer in gen_en
    is_halluc_en  = hallucinated_ans in gen_en and not is_correct_en

    if is_correct_en:
        en_correct += 1;  en_conf_correct.append(conf_en)
    elif is_halluc_en:
        en_halluc += 1;   en_conf_halluc.append(conf_en)
        if conf_en >= CONF_THRESHOLD:
            en_high_conf_wrong += 1

    # -- Chinese inference
    p_zh = zh_prompt(sample["question_zh"], sample["context_zh"])
    gen_zh, conf_zh, ent_zh = run_inference(p_zh)

    is_correct_zh = right_answer_zh in gen_zh or right_answer in gen_zh
    is_halluc_zh  = (halluc_ans_zh in gen_zh or hallucinated_ans in gen_zh) and not is_correct_zh

    if is_correct_zh:
        zh_correct += 1;  zh_conf_correct.append(conf_zh)
    elif is_halluc_zh:
        zh_halluc += 1;   zh_conf_halluc.append(conf_zh)
        if conf_zh >= CONF_THRESHOLD:
            zh_high_conf_wrong += 1

    per_sample.append({
        "index": i,
        "en": {"generated": gen_en, "confidence": round(conf_en, 6),
               "entropy": round(ent_en, 6),
               "is_correct": is_correct_en, "is_hallucinated": is_halluc_en},
        "zh": {"generated": gen_zh, "confidence": round(conf_zh, 6),
               "entropy": round(ent_zh, 6),
               "is_correct": is_correct_zh, "is_hallucinated": is_halluc_zh},
    })

    if (i + 1) % 30 == 0:
        n = i + 1
        print(f"  [{n}/{len(samples)}] EN acc={en_correct/n:.3f} ZH acc={zh_correct/n:.3f}")

# -- Summary -------------------------------------------------------------------

n = len(samples)
summary = {
    "model":       QWEN_MODEL,
    "num_samples": n,
    "english": {
        "accuracy":           round(en_correct / n, 4),
        "hallucination_rate": round(en_halluc  / n, 4),
        "mean_conf_correct":  round(float(np.mean(en_conf_correct)), 6) if en_conf_correct else None,
        "mean_conf_halluc":   round(float(np.mean(en_conf_halluc)),  6) if en_conf_halluc  else None,
        "high_conf_wrong":    en_high_conf_wrong,
    },
    "chinese": {
        "accuracy":           round(zh_correct / n, 4),
        "hallucination_rate": round(zh_halluc  / n, 4),
        "mean_conf_correct":  round(float(np.mean(zh_conf_correct)), 6) if zh_conf_correct else None,
        "mean_conf_halluc":   round(float(np.mean(zh_conf_halluc)),  6) if zh_conf_halluc  else None,
        "high_conf_wrong":    zh_high_conf_wrong,
    },
    "gaps": {
        "accuracy_gap_en_minus_zh": round(en_correct/n - zh_correct/n, 4),
        "conf_halluc_gap_zh_minus_en": round(
            (float(np.mean(zh_conf_halluc)) if zh_conf_halluc else 0) -
            (float(np.mean(en_conf_halluc)) if en_conf_halluc else 0), 6
        ),
    },
}

print(f"\n{'='*60}")
print("CROSS-LINGUAL RESULTS (Qwen2.5-7B)")
print(f"{'='*60}")
print(f"{'Metric':<35} {'English':>10} {'Chinese':>10}")
print("-" * 57)
en, zh = summary["english"], summary["chinese"]
print(f"{'Accuracy':<35} {en['accuracy']:>10.4f} {zh['accuracy']:>10.4f}")
print(f"{'Hallucination Rate':<35} {en['hallucination_rate']:>10.4f} {zh['hallucination_rate']:>10.4f}")
print(f"{'Mean Conf (correct)':<35} {str(en['mean_conf_correct']):>10} {str(zh['mean_conf_correct']):>10}")
print(f"{'Mean Conf (hallucinated)':<35} {str(en['mean_conf_halluc']):>10} {str(zh['mean_conf_halluc']):>10}")
print(f"{'High-conf wrong answers':<35} {en['high_conf_wrong']:>10} {zh['high_conf_wrong']:>10}")
print(f"\nAccuracy gap (EN-ZH):         {summary['gaps']['accuracy_gap_en_minus_zh']:+.4f}")
print(f"Conf halluc gap (ZH-EN):      {summary['gaps']['conf_halluc_gap_zh_minus_en']:+.6f}")

with open(OUTPUT_DIR / "crosslingual_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(OUTPUT_DIR / "crosslingual_samples.json", "w") as f:
    json.dump(per_sample, f, indent=2, ensure_ascii=False)

# -- Plots ---------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
w = 0.35

# Plot 1: Accuracy & Hallucination Rate
metrics = ["Accuracy", "Hallucination Rate"]
en_vals = [en["accuracy"], en["hallucination_rate"]]
zh_vals = [zh["accuracy"], zh["hallucination_rate"]]
x = np.arange(len(metrics))

axes[0].bar(x - w/2, en_vals, w, label="English", color="steelblue", edgecolor="black")
axes[0].bar(x + w/2, zh_vals, w, label="Chinese",  color="coral",     edgecolor="black")
axes[0].set_xticks(x); axes[0].set_xticklabels(metrics)
axes[0].set_ylim(0, 1); axes[0].set_ylabel("Rate")
axes[0].set_title("Qwen2.5: Accuracy & Hallucination Rate\nEnglish vs Chinese")
axes[0].legend()
for xi, (ev, zv) in zip(x, zip(en_vals, zh_vals)):
    axes[0].text(xi - w/2, ev + 0.01, f"{ev:.3f}", ha="center", fontsize=9)
    axes[0].text(xi + w/2, zv + 0.01, f"{zv:.3f}", ha="center", fontsize=9)

# Plot 2: Mean confidence by answer type
labels  = ["Correct", "Hallucinated"]
en_confs = [en["mean_conf_correct"] or 0, en["mean_conf_halluc"] or 0]
zh_confs = [zh["mean_conf_correct"] or 0, zh["mean_conf_halluc"] or 0]
x2 = np.arange(len(labels))

axes[1].bar(x2 - w/2, en_confs, w, label="English", color="steelblue", edgecolor="black")
axes[1].bar(x2 + w/2, zh_confs, w, label="Chinese",  color="coral",     edgecolor="black")
axes[1].set_xticks(x2); axes[1].set_xticklabels(labels)
axes[1].set_ylim(0, 1); axes[1].set_ylabel("Mean Token Confidence (top-1 prob)")
axes[1].set_title("Qwen2.5: Token Confidence by Answer Type\nEnglish vs Chinese")
axes[1].legend()
for xi, (ev, zv) in zip(x2, zip(en_confs, zh_confs)):
    axes[1].text(xi - w/2, ev + 0.01, f"{ev:.3f}", ha="center", fontsize=9)
    axes[1].text(xi + w/2, zv + 0.01, f"{zv:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "crosslingual_comparison.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_DIR}/crosslingual_comparison.png")
print("\nDone. All results in results/crosslingual/")
