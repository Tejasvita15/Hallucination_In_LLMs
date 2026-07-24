from __future__ import annotations

from .data import QASample
from .metrics import RunningStats, classify_generation, summarize
from .models import HFGenerator
from .prompts import qa_prompt


def evaluate_qa_samples(
    generator: HFGenerator,
    samples: list[QASample],
    model_name: str,
    language_code: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    high_confidence_threshold: float,
    prompt_prefix: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    stats = RunningStats()
    rows: list[dict[str, object]] = []

    for sample in samples:
        prompt = qa_prompt(sample.question, sample.context, language_code, prompt_prefix)
        generation = generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        label = classify_generation(generation.text, sample.right_answer, sample.hallucinated_answer)
        stats.total += 1

        if label == "correct":
            stats.correct += 1
            stats.conf_correct.append(generation.mean_confidence)
            stats.entropy_correct.append(generation.mean_entropy)
        elif label == "hallucinated":
            stats.hallucinated += 1
            stats.conf_hallucinated.append(generation.mean_confidence)
            stats.entropy_hallucinated.append(generation.mean_entropy)
            if generation.mean_confidence >= high_confidence_threshold:
                stats.high_conf_wrong += 1
        elif label == "refused":
            stats.refused += 1
        else:
            stats.neither += 1
            stats.conf_neither.append(generation.mean_confidence)
            stats.entropy_neither.append(generation.mean_entropy)
            if generation.mean_confidence >= high_confidence_threshold:
                stats.high_conf_wrong += 1

        rows.append(
            {
                "sample_index": sample.index,
                "model": model_name,
                "language": language_code,
                "question": sample.question,
                "context": sample.context,
                "right_answer": sample.right_answer,
                "hallucinated_answer": sample.hallucinated_answer,
                "prompt": prompt,
                "generation": generation.text,
                "label": label,
                "mean_confidence": generation.mean_confidence,
                "mean_entropy": generation.mean_entropy,
            }
        )

    return summarize(stats, language_code, model_name, high_confidence_threshold), rows

