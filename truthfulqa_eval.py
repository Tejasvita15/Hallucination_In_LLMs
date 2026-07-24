"""
SCRIPT 1: TruthfulQA Baseline Evaluation
=========================================
Runs TruthfulQA (MC1 + MC2) on all 3 models using lm-evaluation-harness.
Saves results to results/truthfulqa/

Usage:
    python 01_truthfulqa_eval.py

Requirements:
    pip install lm-eval
    pip install accelerate
"""

import subprocess
import json
import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "llama3.1-8b":   "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b":    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

OUTPUT_DIR = Path("results/truthfulqa")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = "truthfulqa_mc1,truthfulqa_mc2"
BATCH_SIZE = 8
DEVICE = "cuda"

# ── Run evaluation ─────────────────────────────────────────────────────────────

summary = {}

for model_name, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    output_path = OUTPUT_DIR / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16",
        "--tasks", TASKS,
        "--device", DEVICE,
        "--batch_size", str(BATCH_SIZE),
        "--output_path", str(output_path),
        "--log_samples",                        # saves per-sample outputs
        "--trust_remote_code",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"[ERROR] {model_name} failed. Check logs above.")
        continue

    # Parse the results JSON written by lm_eval
    result_files = list(output_path.glob("**/results.json"))
    if result_files:
        with open(result_files[0]) as f:
            data = json.load(f)
        results = data.get("results", {})
        mc1 = results.get("truthfulqa_mc1", {}).get("acc,none", None)
        mc2 = results.get("truthfulqa_mc2", {}).get("acc,none", None)
        summary[model_name] = {"mc1_acc": mc1, "mc2_acc": mc2}
        print(f"\n✓ {model_name}: MC1={mc1:.4f}, MC2={mc2:.4f}")
    else:
        print(f"[WARN] Could not find results.json for {model_name}")

# ── Print comparison table ─────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("TRUTHFULQA RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<25} {'MC1 Acc':>10} {'MC2 Acc':>10}")
print("-" * 47)
for model_name, scores in summary.items():
    mc1 = f"{scores['mc1_acc']:.4f}" if scores['mc1_acc'] is not None else "N/A"
    mc2 = f"{scores['mc2_acc']:.4f}" if scores['mc2_acc'] is not None else "N/A"
    print(f"{model_name:<25} {mc1:>10} {mc2:>10}")

# Save summary
summary_path = OUTPUT_DIR / "summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {summary_path}")