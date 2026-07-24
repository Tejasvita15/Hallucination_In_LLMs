"""
SCRIPT 5: Aggregate Results & Generate Thesis Figures
======================================================
Loads all saved results from scripts 1–4 and produces:
  1. Combined benchmark table (TruthfulQA + HaluEval)
  2. Overconfidence gap bar chart across models
  3. Cross-lingual comparison chart (Qwen)
  4. Combined summary JSON for your thesis writeup

Run this AFTER scripts 1–4 have completed.

Usage:
    python 05_aggregate_results.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load all results ───────────────────────────────────────────────────────────

def load_json(path):
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Not found: {path}")
        return None
    with open(p) as f:
        return json.load(f)

tqa_summary     = load_json("results/truthfulqa/summary.json")       or {}
halu_summary    = load_json("results/halueval/summary.json")          or {}
logit_summary   = load_json("results/logits/confidence_summary.json") or {}
xling_summary   = load_json("results/crosslingual/crosslingual_summary.json") or {}

MODELS = ["llama3.1-8b", "qwen2.5-7b", "deepseek-r1-7b"]

# ── Figure 1: Combined Benchmark Table ────────────────────────────────────────

print("\nGenerating Figure 1: Combined Benchmark Table")

fig1, ax1 = plt.subplots(figsize=(10, 3.5))
ax1.axis("off")

col_labels = ["Model", "TruthfulQA MC1↑", "TruthfulQA MC2↑",
              "HaluEval Acc↑", "HaluEval Halluc%↓"]
rows = []
for m in MODELS:
    tqa  = tqa_summary.get(m, {})
    halu = halu_summary.get(m, {})
    rows.append([
        m,
        f"{tqa.get('mc1_acc', 'N/A'):.4f}" if tqa.get('mc1_acc') else "N/A",
        f"{tqa.get('mc2_acc', 'N/A'):.4f}" if tqa.get('mc2_acc') else "N/A",
        f"{halu.get('accuracy', 'N/A'):.4f}" if halu.get('accuracy') else "N/A",
        f"{halu.get('hallucination_rate', 'N/A'):.4f}" if halu.get('hallucination_rate') else "N/A",
    ])

table = ax1.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(11)

# Header styling
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Row alternating colors
for i, row in enumerate(rows):
    color = "#eaf0fb" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(color)

ax1.set_title("Table 1: Benchmark Results Across Models", fontsize=13, pad=12, fontweight="bold")
plt.tight_layout()
fig1.savefig(OUTPUT_DIR / "fig1_benchmark_table.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {OUTPUT_DIR}/fig1_benchmark_table.png")


# ── Figure 2: Overconfidence Gap ──────────────────────────────────────────────

print("Generating Figure 2: Overconfidence Gap")

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Left: mean confidence correct vs hallucinated per model
x    = np.arange(len(MODELS))
w    = 0.35
conf_correct = [logit_summary.get(m, {}).get("correct_mean_conf") or 0 for m in MODELS]
conf_halluc  = [logit_summary.get(m, {}).get("halluc_mean_conf")  or 0 for m in MODELS]

axes2[0].bar(x - w/2, conf_correct, w, label="Correct",      color="steelblue", edgecolor="black")
axes2[0].bar(x + w/2, conf_halluc,  w, label="Hallucinated", color="tomato",    edgecolor="black")
axes2[0].set_xticks(x)
axes2[0].set_xticklabels([m.replace(".", "\n") for m in MODELS], fontsize=9)
axes2[0].set_ylabel("Mean Token Confidence (top-1 prob)")
axes2[0].set_ylim(0, 1)
axes2[0].set_title("Mean Token Confidence\nCorrect vs Hallucinated Answers")
axes2[0].legend()

# Right: overconfidence gap
gaps   = [logit_summary.get(m, {}).get("overconfidence_gap") or 0 for m in MODELS]
colors = ["tomato" if g > 0 else "steelblue" for g in gaps]
axes2[1].bar(MODELS, gaps, color=colors, edgecolor="black", width=0.5)
axes2[1].axhline(0, color="black", linewidth=0.8)
axes2[1].set_ylabel("Overconfidence Gap\n(Halluc conf − Correct conf)")
axes2[1].set_title("Overconfidence Gap per Model\n(positive = more confident when wrong)")
axes2[1].set_xticklabels([m.replace(".", "\n") for m in MODELS], fontsize=9)
for i, (bar, g) in enumerate(zip(axes2[1].patches, gaps)):
    axes2[1].text(bar.get_x() + bar.get_width()/2, g + 0.0005,
                  f"{g:+.4f}", ha="center", va="bottom", fontsize=10)

plt.suptitle("Figure 2: Token-Level Confidence Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
fig2.savefig(OUTPUT_DIR / "fig2_overconfidence.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {OUTPUT_DIR}/fig2_overconfidence.png")


# ── Figure 3: Cross-lingual ───────────────────────────────────────────────────

print("Generating Figure 3: Cross-lingual Qwen Analysis")

if xling_summary:
    en = xling_summary.get("english", {})
    zh = xling_summary.get("chinese", {})

    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Accuracy & Hallucination Rate
    metrics = ["Accuracy", "Hallucination\nRate"]
    en_vals = [en.get("accuracy", 0), en.get("hallucination_rate", 0)]
    zh_vals = [zh.get("accuracy", 0), zh.get("hallucination_rate", 0)]
    xp = np.arange(2)
    axes3[0].bar(xp - w/2, en_vals, w, label="English", color="steelblue", edgecolor="black")
    axes3[0].bar(xp + w/2, zh_vals, w, label="Chinese", color="coral",     edgecolor="black")
    axes3[0].set_xticks(xp); axes3[0].set_xticklabels(metrics)
    axes3[0].set_ylim(0, 1); axes3[0].set_ylabel("Rate")
    axes3[0].set_title("(A) Accuracy & Hallucination Rate")
    axes3[0].legend()
    for xi, (ev, zv) in zip(xp, zip(en_vals, zh_vals)):
        axes3[0].text(xi - w/2, ev + 0.01, f"{ev:.3f}", ha="center", fontsize=9)
        axes3[0].text(xi + w/2, zv + 0.01, f"{zv:.3f}", ha="center", fontsize=9)

    # Panel B: Token confidence by answer type
    labels = ["Correct", "Hallucinated"]
    en_c = [en.get("mean_conf_correct") or 0, en.get("mean_conf_halluc") or 0]
    zh_c = [zh.get("mean_conf_correct") or 0, zh.get("mean_conf_halluc") or 0]
    xq = np.arange(2)
    axes3[1].bar(xq - w/2, en_c, w, label="English", color="steelblue", edgecolor="black")
    axes3[1].bar(xq + w/2, zh_c, w, label="Chinese", color="coral",     edgecolor="black")
    axes3[1].set_xticks(xq); axes3[1].set_xticklabels(labels)
    axes3[1].set_ylim(0, 1); axes3[1].set_ylabel("Mean Confidence")
    axes3[1].set_title("(B) Token Confidence by Answer Type")
    axes3[1].legend()

    # Panel C: High-confidence wrong answers (dangerous hallucinations)
    hcw_en = en.get("high_conf_wrong", 0)
    hcw_zh = zh.get("high_conf_wrong", 0)
    axes3[2].bar(["English", "Chinese"], [hcw_en, hcw_zh],
                 color=["steelblue", "coral"], edgecolor="black", width=0.4)
    axes3[2].set_ylabel("Count")
    axes3[2].set_title("(C) High-Confidence Wrong Answers\n(conf ≥ 0.75, incorrect)")
    for i, v in enumerate([hcw_en, hcw_zh]):
        axes3[2].text(i, v + 0.3, str(v), ha="center", fontsize=11)

    plt.suptitle("Figure 3: Qwen2.5 Cross-Lingual Hallucination Analysis\n(English vs Chinese)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig3.savefig(OUTPUT_DIR / "fig3_crosslingual.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/fig3_crosslingual.png")
else:
    print("  [SKIP] No cross-lingual data found.")


# ── Save combined summary ─────────────────────────────────────────────────────

combined = {
    "truthfulqa":     tqa_summary,
    "halueval":       halu_summary,
    "logit_analysis": logit_summary,
    "crosslingual":   xling_summary,
}
with open(OUTPUT_DIR / "combined_results.json", "w") as f:
    json.dump(combined, f, indent=2)

print(f"\n{'='*60}")
print("ALL FIGURES GENERATED")
print(f"{'='*60}")
print(f"  {OUTPUT_DIR}/fig1_benchmark_table.png")
print(f"  {OUTPUT_DIR}/fig2_overconfidence.png")
print(f"  {OUTPUT_DIR}/fig3_crosslingual.png")
print(f"  {OUTPUT_DIR}/combined_results.json")
print("\nYou're ready to drop these into your thesis. Good luck!")