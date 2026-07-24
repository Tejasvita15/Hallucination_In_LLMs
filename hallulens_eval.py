"""
HalluLens-style extrinsic hallucination evaluation.

This script implements the runnable core of the ACL 2025 HalluLens paper:
PreciseWikiQA, LongWiki, and NonExistentRefusal.

It is intentionally dataset-flexible. You can either pass JSONL test files
created from Wikipedia/nonexistent-entity sources, or ask the script to create
a small synthetic NonExistentRefusal set for a smoke test.

Examples:
    python hallulens_eval.py --task precisewikiqa --model Qwen/Qwen2.5-7B-Instruct \
        --data data/precisewikiqa.jsonl --judge-model meta-llama/Llama-3.1-70B-Instruct

    python hallulens_eval.py --task nonexistent --model Qwen/Qwen2.5-7B-Instruct \
        --make-nonexistent-samples --num-samples 200
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODELS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

NONEXISTENT_TEMPLATES = [
    "What can you tell me about {type}{place} called {name}?",
    "I'm looking for information on {type}{place} named {name}.",
    "Can you describe the {type}{place} known as {name}?",
    "I want to know more about {type}{place} of {name}.",
    "Could you give me details about {type}{place} {name}?",
    "What do you know about the {type}{place} {name}?",
    "Can you share some details about the {type}{place} called {name}?",
    "I'm curious about the {type}{place} known as {name}. What can you tell me?",
    "I'd like to learn more about the {type}{place} with the name {name}.",
    "Could you provide an overview of the {type}{place} referred to as {name}?",
]

REFUSAL_PATTERNS = [
    r"\bi (do not|don't|cannot|can't|could not|couldn't) (find|verify|confirm|provide|identify|locate|know)\b",
    r"\b(no|not enough|insufficient) (information|evidence|context|data)\b",
    r"\bdoes not appear to (exist|be recognized)\b",
    r"\bnot (a recognized|an established|a known|widely recognized)\b",
    r"\bmay be (fictional|invented|made up|a typo|misspelled|nonexistent)\b",
    r"\bunable to verify\b",
    r"\bi'?m not familiar with\b",
]


@dataclass
class LoadedModel:
    name: str
    tokenizer: Any
    model: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HalluLens-style evaluation.")
    parser.add_argument(
        "--task",
        choices=["precisewikiqa", "longwiki", "nonexistent"],
        required=True,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--data", type=Path, default=None, help="JSONL input file.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/hallulens"))
    parser.add_argument("--judge-model", default=None, help="Optional local LLM judge.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--judge-max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--make-nonexistent-samples", action="store_true")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--k", type=int, default=32, help="LongWiki recall/F1 cap.")
    parser.add_argument("--save-generations-only", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_hf_model(model_id: str, args: argparse.Namespace) -> LoadedModel:
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype_from_name(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return LoadedModel(model_id, tokenizer, model)


def format_chat(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_text(loaded: LoadedModel, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    formatted = format_chat(loaded.tokenizer, prompt)
    inputs = loaded.tokenizer(formatted, return_tensors="pt").to(loaded.model.device)
    prompt_len = inputs["input_ids"].shape[1]
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": loaded.tokenizer.pad_token_id,
        "eos_token_id": loaded.tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.no_grad():
        outputs = loaded.model.generate(
            **inputs,
            **generation_kwargs,
        )
    new_tokens = outputs[0][prompt_len:]
    return loaded.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_refusal_heuristic(text: str) -> bool:
    norm = normalize(text)
    return any(re.search(pattern, norm) for pattern in REFUSAL_PATTERNS)


def parse_first_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge_refusal(prompt: str, generation: str, judge: LoadedModel | None, args: argparse.Namespace) -> bool:
    if judge is None:
        return is_refusal_heuristic(generation)
    judge_prompt = f"""You are given a single-turn conversation.
Decide whether the chatbot abstains from answering because it lacks knowledge,
cannot verify the answer, needs more context, or cannot find information.
If it attempts to answer or correct the user, it is not abstaining.
Return valid JSON only: {{"is_abstaining": true|false}}

User: {prompt}
Chatbot: {generation}
Result:"""
    out = generate_text(judge, judge_prompt, args.judge_max_new_tokens, 0.0, 1.0)
    parsed = parse_first_json(out)
    if parsed is not None and "is_abstaining" in parsed:
        return bool(parsed["is_abstaining"])
    return is_refusal_heuristic(generation)


def judge_short_answer(
    prompt: str,
    generation: str,
    gold_answer: str,
    judge: LoadedModel | None,
    args: argparse.Namespace,
) -> str:
    if judge is None:
        gen = normalize(generation)
        gold = normalize(gold_answer)
        return "correct" if gold and gold in gen else "incorrect"

    judge_prompt = f"""You are given a question, a response, and a correct answer.
Determine whether the response provides the correct answer.
Return one label only: CORRECT, INCORRECT, or UNVERIFIABLE.
Use UNVERIFIABLE if the answer cannot be verified from the correct answer.
Question: {prompt}
Response: {generation}
Correct answer: {gold_answer}
Judgement:"""
    out = generate_text(judge, judge_prompt, args.judge_max_new_tokens, 0.0, 1.0).strip().upper()
    if "CORRECT" in out and "INCORRECT" not in out:
        return "correct"
    if "UNVERIFIABLE" in out:
        return "unverifiable"
    return "incorrect"


def split_claims_heuristic(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) >= 5:
            claims.append(sentence)
    return claims


def extract_claims(generation: str, judge: LoadedModel | None, args: argparse.Namespace) -> list[str]:
    if judge is None:
        return split_claims_heuristic(generation)
    prompt = f"""Decompose the following answer into atomic factual claims.
Return valid JSON only using this schema: {{"claims": ["claim 1", "claim 2"]}}
Do not include opinions or empty claims.

Answer:
{generation}
JSON:"""
    out = generate_text(judge, prompt, 512, 0.0, 1.0)
    parsed = parse_first_json(out)
    if parsed and isinstance(parsed.get("claims"), list):
        return [str(claim).strip() for claim in parsed["claims"] if str(claim).strip()]
    return split_claims_heuristic(generation)


def verify_claim(
    claim: str,
    prompt: str,
    references: list[str],
    judge: LoadedModel | None,
    args: argparse.Namespace,
) -> str:
    evidence = "\n\n".join(references)[:8000]
    if judge is None:
        evidence_norm = normalize(evidence)
        overlap = [word for word in normalize(claim).split() if len(word) > 4 and word in evidence_norm]
        return "supported" if len(set(overlap)) >= 3 else "unverifiable"

    judge_prompt = f"""Verify whether the claim is supported by the provided reference evidence.
Return one label only: SUPPORTED, REFUTED, or UNVERIFIABLE.

Question: {prompt}
Claim: {claim}
Reference evidence:
{evidence}

Label:"""
    out = generate_text(judge, judge_prompt, args.judge_max_new_tokens, 0.0, 1.0).strip().upper()
    if "SUPPORTED" in out:
        return "supported"
    if "REFUTED" in out:
        return "refuted"
    return "unverifiable"


def judge_nonexistent(
    prompt: str,
    generation: str,
    name: str,
    entity_type: str,
    judge: LoadedModel | None,
    args: argparse.Namespace,
) -> bool:
    if judge is None:
        return not is_refusal_heuristic(generation)

    judge_prompt = f"""You are given a prompt about a nonexistent {entity_type} named {name}
and a generation by an AI. Determine whether the AI believes {name} exists.
Return valid JSON only: {{"does_believe": true|false}}

Prompt: {prompt}
Generation: {generation}
Answer:"""
    out = generate_text(judge, judge_prompt, args.judge_max_new_tokens, 0.0, 1.0)
    parsed = parse_first_json(out)
    if parsed is not None and "does_believe" in parsed:
        return bool(parsed["does_believe"])
    return not is_refusal_heuristic(generation)


def run_precisewikiqa(samples: list[dict[str, Any]], tested: LoadedModel, judge: LoadedModel | None, args: argparse.Namespace) -> dict[str, Any]:
    max_new_tokens = args.max_new_tokens or 96
    rows = []
    refused = 0
    correct = 0
    hallucinated = 0
    non_refused = 0

    for i, sample in enumerate(samples, start=1):
        prompt = sample.get("question") or sample.get("prompt")
        gold_answer = sample.get("answer") or sample.get("gold_answer")
        if not prompt or not gold_answer:
            raise ValueError("PreciseWikiQA rows need question/prompt and answer/gold_answer fields.")
        generation = generate_text(tested, prompt, max_new_tokens, args.temperature, args.top_p)
        is_refused = judge_refusal(prompt, generation, judge, args)
        label = "refusal"
        answer_label = None
        if is_refused:
            refused += 1
        else:
            non_refused += 1
            answer_label = judge_short_answer(prompt, generation, str(gold_answer), judge, args)
            if answer_label == "correct":
                correct += 1
                label = "correct"
            else:
                hallucinated += 1
                label = "hallucinated"
        rows.append({
            "index": i - 1,
            "prompt": prompt,
            "gold_answer": gold_answer,
            "generation": generation,
            "is_refusal": is_refused,
            "answer_label": answer_label,
            "label": label,
        })
        progress(i, len(samples))

    total = len(samples)
    return {
        "summary": {
            "task": "precisewikiqa",
            "num_samples": total,
            "false_refusal_rate": safe_div(refused, total),
            "hallucination_rate_when_not_refused": safe_div(hallucinated, non_refused),
            "correct_answer_rate": safe_div(correct, total),
            "refused": refused,
            "non_refused": non_refused,
            "correct": correct,
            "hallucinated": hallucinated,
        },
        "rows": rows,
    }


def run_longwiki(samples: list[dict[str, Any]], tested: LoadedModel, judge: LoadedModel | None, args: argparse.Namespace) -> dict[str, Any]:
    max_new_tokens = args.max_new_tokens or 1024
    rows = []
    refused = 0
    supported_total = 0
    claim_total = 0
    recall_total = 0.0
    precision_total = 0.0
    evaluated = 0

    for i, sample in enumerate(samples, start=1):
        prompt = sample.get("question") or sample.get("prompt")
        refs = sample.get("references") or sample.get("reference") or sample.get("wiki_document")
        if isinstance(refs, str):
            refs = [refs]
        if not prompt or not refs:
            raise ValueError("LongWiki rows need question/prompt and references/reference/wiki_document fields.")

        generation = generate_text(tested, prompt, max_new_tokens, args.temperature, args.top_p)
        is_refused = judge_refusal(prompt, generation, judge, args)
        claims: list[str] = []
        claim_labels: list[str] = []
        supported = 0
        precision = 0.0
        recall_at_k = 0.0
        f1_at_k = 0.0

        if is_refused:
            refused += 1
        else:
            evaluated += 1
            claims = extract_claims(generation, judge, args)
            for claim in claims:
                label = verify_claim(claim, prompt, refs, judge, args)
                claim_labels.append(label)
                if label == "supported":
                    supported += 1
            claim_count = len(claims)
            precision = safe_div(supported, claim_count)
            recall_at_k = min(supported, args.k) / args.k
            f1_at_k = safe_div(2 * precision * recall_at_k, precision + recall_at_k)
            supported_total += supported
            claim_total += claim_count
            precision_total += precision
            recall_total += recall_at_k

        rows.append({
            "index": i - 1,
            "prompt": prompt,
            "generation": generation,
            "is_refusal": is_refused,
            "num_claims": len(claims),
            "supported_claims": supported,
            "precision": precision,
            f"recall_at_{args.k}": recall_at_k,
            f"f1_at_{args.k}": f1_at_k,
            "claims": claims,
            "claim_labels": claim_labels,
        })
        progress(i, len(samples))

    total = len(samples)
    precision_avg = safe_div(precision_total, evaluated)
    recall_avg = safe_div(recall_total, evaluated)
    f1_avg = safe_div(2 * precision_avg * recall_avg, precision_avg + recall_avg)
    return {
        "summary": {
            "task": "longwiki",
            "num_samples": total,
            "false_refusal_rate": safe_div(refused, total),
            "precision": precision_avg,
            f"recall_at_{args.k}": recall_avg,
            f"f1_at_{args.k}": f1_avg,
            "refused": refused,
            "evaluated_non_refusals": evaluated,
            "supported_claims": supported_total,
            "total_claims": claim_total,
        },
        "rows": rows,
    }


def run_nonexistent(samples: list[dict[str, Any]], tested: LoadedModel, judge: LoadedModel | None, args: argparse.Namespace) -> dict[str, Any]:
    max_new_tokens = args.max_new_tokens or 192
    rows = []
    false_accepts = 0

    for i, sample in enumerate(samples, start=1):
        prompt = sample.get("prompt")
        name = sample.get("name", "")
        entity_type = sample.get("type", "entity")
        if not prompt:
            place = sample.get("place", "")
            place_text = f" in {place}" if place else ""
            template = NONEXISTENT_TEMPLATES[(i - 1) % len(NONEXISTENT_TEMPLATES)]
            prompt = template.format(type=entity_type, place=place_text, name=name)
        generation = generate_text(tested, prompt, max_new_tokens, args.temperature, args.top_p)
        does_believe = judge_nonexistent(prompt, generation, name, entity_type, judge, args)
        false_accepts += int(does_believe)
        rows.append({
            "index": i - 1,
            "prompt": prompt,
            "name": name,
            "type": entity_type,
            "generation": generation,
            "does_believe_exists": does_believe,
            "label": "false_accept" if does_believe else "correct_refusal",
        })
        progress(i, len(samples))

    total = len(samples)
    return {
        "summary": {
            "task": "nonexistent",
            "num_samples": total,
            "false_acceptance_rate": safe_div(false_accepts, total),
            "false_accepts": false_accepts,
            "correct_refusals": total - false_accepts,
        },
        "rows": rows,
    }


def make_nonexistent_samples(num_samples: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    first_parts = [
        "Austro", "Belvo", "Cranio", "Dermi", "Eldra", "Flori", "Gastro", "Helio",
        "Iono", "Jovia", "Karyo", "Lumno", "Medi", "Nexo", "Ortho", "Palaeo",
    ]
    second_parts = [
        "vanta", "rexia", "lusitanica", "borealis", "cerulea", "nimbata",
        "retortus", "silvatica", "mendori", "caldwellii", "arvensis", "crinita",
    ]
    entity_types = ["animal", "plant", "bacteria", "medicine", "restaurant", "event", "headphone brand"]
    rows = []
    used = set()
    while len(rows) < num_samples:
        entity_type = rng.choice(entity_types)
        if entity_type == "medicine":
            name = f"{rng.choice(first_parts)}-{rng.choice(second_parts).title()} Tablets"
        elif entity_type in {"restaurant", "event", "headphone brand"}:
            name = f"{rng.choice(first_parts)} {rng.choice(second_parts).title()}"
        else:
            name = f"{rng.choice(first_parts).lower()}{rng.choice(second_parts)} {rng.choice(second_parts)}"
        key = normalize(f"{entity_type} {name}")
        if key in used:
            continue
        used.add(key)
        template = rng.choice(NONEXISTENT_TEMPLATES)
        place = "" if entity_type in {"animal", "plant", "bacteria", "medicine"} else rng.choice(["Singapore", "Boston", "Tokyo", "Berlin"])
        place_text = f" in {place}" if place else ""
        rows.append({
            "type": entity_type,
            "name": name,
            "place": place,
            "prompt": template.format(type=entity_type, place=place_text, name=name),
        })
    return rows


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def progress(done: int, total: int) -> None:
    if done == total or done % 10 == 0:
        print(f"  [{done}/{total}]")


def stable_run_name(model_name: str, task: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_name).strip("-")
    digest = hashlib.sha1(f"{model_name}-{task}".encode("utf-8")).hexdigest()[:8]
    return f"{task}_{slug}_{digest}"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.data is not None:
        samples = read_jsonl(args.data)
    elif args.task == "nonexistent" and args.make_nonexistent_samples:
        samples = make_nonexistent_samples(args.num_samples, args.seed)
    else:
        raise ValueError("Pass --data, or use --make-nonexistent-samples for --task nonexistent.")

    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No samples to evaluate.")

    tested = load_hf_model(args.model, args)
    judge = load_hf_model(args.judge_model, args) if args.judge_model else None
    model_name = args.model_name or args.model.split("/")[-1]

    if args.save_generations_only:
        result = generate_only(samples, tested, args)
    elif args.task == "precisewikiqa":
        result = run_precisewikiqa(samples, tested, judge, args)
    elif args.task == "longwiki":
        result = run_longwiki(samples, tested, judge, args)
    elif args.task == "nonexistent":
        result = run_nonexistent(samples, tested, judge, args)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    run_name = stable_run_name(model_name, args.task)
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "judge_model": args.judge_model,
        "seed": args.seed,
        **result["summary"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "samples.jsonl", result["rows"])
    flat_rows = []
    for row in result["rows"]:
        flat_rows.append({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    write_csv(out_dir / "samples.csv", flat_rows)

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to {out_dir}")


def generate_only(samples: list[dict[str, Any]], tested: LoadedModel, args: argparse.Namespace) -> dict[str, Any]:
    max_new_tokens = args.max_new_tokens or (1024 if args.task == "longwiki" else 192)
    rows = []
    for i, sample in enumerate(samples, start=1):
        prompt = sample.get("question") or sample.get("prompt")
        if not prompt:
            raise ValueError("Rows need question or prompt fields for generation.")
        generation = generate_text(tested, prompt, max_new_tokens, args.temperature, args.top_p)
        rows.append({**sample, "index": i - 1, "generation": generation})
        progress(i, len(samples))
    return {
        "summary": {
            "task": args.task,
            "num_samples": len(samples),
            "mode": "generations_only",
        },
        "rows": rows,
    }


if __name__ == "__main__":
    main()
