from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.data import QASample, load_halueval_qa
from hallu_llm.detection import (
    build_paired_audit_rows,
    build_detection_instances,
    evaluate_detection_instances,
    stable_sample_id,
)
from hallu_llm.models import HFGenerator
from hallu_llm.reviews import (
    load_review_manifest,
    review_manifest_metadata,
    translation_review,
)
from hallu_llm.text import is_language_neutral_number
from hallu_llm.translation import TranslationResult, Translator
from hallu_llm.translation_quality import (
    grounded_answer_alignment_status,
    missing_numbers,
    question_intent_lost,
    summarize_translation_quality,
    translation_quality_warnings,
)
from hallu_llm.utils import (
    ensure_dir,
    read_json,
    set_seed,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired, cross-lingual HaluEval hallucination detection."
    )
    parser.add_argument("--config", required=True, help="Path to detection experiment JSON.")
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Override num_samples without modifying the tracked config.",
    )
    parser.add_argument(
        "--experiment-name",
        help="Override experiment_name so smoke and full runs remain separate.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Use every valid row in the configured dataset split.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and validate translation caches without loading detector models.",
    )
    parser.add_argument(
        "--fail-on-translation-quality",
        action="store_true",
        help="Abort before model inference when a translation quality gate fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = read_json(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    configured_num_samples = cfg.get("num_samples", 20)
    if args.num_samples is not None and args.all_samples:
        raise ValueError("--num-samples and --all-samples cannot be used together.")
    if args.all_samples:
        num_samples = None
    elif args.num_samples is not None:
        num_samples = args.num_samples
    elif configured_num_samples is None:
        num_samples = None
    else:
        num_samples = int(configured_num_samples)
    experiment_name = args.experiment_name or cfg["experiment_name"]
    review_path = _resolve_optional_path(
        cfg.get("review_manifest"),
        base_dir=config_path.parent,
    )
    reviews = load_review_manifest(review_path)
    reviews_metadata = review_manifest_metadata(reviews)
    if args.fail_on_translation_quality:
        for language_cfg in cfg.get("languages", []):
            if language_cfg.get("translator") is None:
                continue
            quality_cfg = dict(language_cfg.get("translation_quality", {}))
            quality_cfg["fail_on_quality_gate"] = True
            language_cfg["translation_quality"] = quality_cfg
    effective_cfg = dict(cfg)
    effective_cfg["num_samples"] = num_samples
    effective_cfg["experiment_name"] = experiment_name
    effective_cfg["review_manifest_metadata"] = reviews_metadata

    output_root = ensure_dir(
        Path(cfg.get("output_dir", "results/detection")) / experiment_name
    )
    source_samples = load_halueval_qa(
        num_samples=num_samples,
        seed=seed,
        dataset_cfg=cfg.get("dataset", {}),
    )
    source_fingerprint = _samples_fingerprint(source_samples)
    write_json(output_root / "config.json", effective_cfg)
    print(f"Loaded {len(source_samples)} HaluEval source samples")
    print(f"Source fingerprint: {source_fingerprint}")
    print(
        "Review manifest: "
        f"{review_path if review_path is not None else 'none'} "
        f"({reviews_metadata['sample_reviews']} sample reviews)"
    )

    prepared_languages: dict[
        str,
        tuple[list[QASample], dict[int, str], dict[int, list[str]]],
    ] = {}
    for language_cfg in cfg["languages"]:
        language_code = language_cfg["code"]
        print(f"\n=== Preparing dataset: {language_code} ===")
        prepared_languages[language_code] = prepare_language_dataset(
            source_samples=source_samples,
            source_fingerprint=source_fingerprint,
            language_cfg=language_cfg,
            cache_dir=ensure_dir(output_root / "datasets"),
            review_manifest=reviews,
            review_metadata=reviews_metadata,
        )

    if args.prepare_only:
        print(
            "\nPreparation complete. Translation caches and quality audits "
            f"saved to {output_root / 'datasets'}"
        )
        return

    model_cfg_defaults = cfg.get("model_loading", cfg.get("generation", {}))
    detection_cfg = cfg.get("detection", {})
    max_new_tokens = int(detection_cfg.get("max_new_tokens", 4))
    confidence_threshold = float(detection_cfg.get("high_confidence_threshold", 0.75))
    all_summaries: list[dict[str, Any]] = []

    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        model_path = model_cfg["path"]
        print(f"\n=== Loading model: {model_name} ({model_path}) ===")
        generator = HFGenerator.load(
            model_path,
            dtype=model_cfg_defaults.get("dtype", "bfloat16"),
            device_map=model_cfg_defaults.get("device_map", "auto"),
        )

        for language_cfg in cfg["languages"]:
            language_code = language_cfg["code"]
            samples, sample_ids, translation_warnings = prepared_languages[language_code]
            instances = build_detection_instances(
                samples,
                seed=seed,
                sample_ids=sample_ids,
                source_reviews=reviews.get("samples", {}),
            )
            print(
                f"\n--- Detecting hallucinations: {model_name} / {language_code} "
                f"({len(instances)} balanced instances) ---"
            )
            summary, rows = evaluate_detection_instances(
                generator=generator,
                instances=instances,
                model_name=model_name,
                language_code=language_code,
                max_new_tokens=max_new_tokens,
                high_confidence_threshold=confidence_threshold,
                prompt_instruction=language_cfg.get("detection_instruction"),
                translation_warnings=translation_warnings,
                length_match_max_relative_difference=float(
                    detection_cfg.get(
                        "length_match_max_relative_difference",
                        0.25,
                    )
                ),
            )
            all_summaries.append(summary)
            run_dir = ensure_dir(output_root / model_name / language_code)
            write_json(run_dir / "summary.json", summary)
            write_jsonl(run_dir / "predictions.jsonl", rows)
            write_csv(run_dir / "paired_audit.csv", build_paired_audit_rows(rows))
            print(json.dumps(summary, indent=2, ensure_ascii=False))

        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(output_root / "summary_all.json", all_summaries)
    print(f"\nDone. Detection results saved to {output_root}")


def prepare_language_dataset(
    source_samples: list[QASample],
    source_fingerprint: str,
    language_cfg: dict[str, Any],
    cache_dir: Path,
    review_manifest: dict[str, Any] | None = None,
    review_metadata: dict[str, Any] | None = None,
) -> tuple[list[QASample], dict[int, str], dict[int, list[str]]]:
    language_code = language_cfg["code"]
    translator_spec = _normalize_translator_spec(language_cfg.get("translator"))
    quality_cfg = dict(language_cfg.get("translation_quality", {}))
    preserve_numeric_answers = bool(
        language_cfg.get("preserve_numeric_answers", True)
    )
    retry_failed_questions = bool(
        language_cfg.get("retry_failed_questions", True)
    )
    sentence_split_fields = sorted(
        {
            str(field)
            for field in language_cfg.get("sentence_split_fields", ["knowledge"])
        }
    )
    supported_fields = {
        "question",
        "knowledge",
        "right_answer",
        "hallucinated_answer",
    }
    unsupported_fields = set(sentence_split_fields) - supported_fields
    if unsupported_fields:
        raise ValueError(
            "Unsupported sentence_split_fields: "
            f"{sorted(unsupported_fields)}"
        )
    cache_path = cache_dir / f"{language_code}.jsonl"
    metadata_path = cache_dir / f"{language_code}.meta.json"
    expected_metadata = {
        "schema_version": 5,
        "language": language_code,
        "source_fingerprint": source_fingerprint,
        "num_samples": len(source_samples),
        "translator": translator_spec,
        "translation_max_length": int(language_cfg.get("translation_max_length", 512)),
        "sentence_split_fields": sentence_split_fields,
        "translation_policy": {
            "preserve_numeric_answers": preserve_numeric_answers,
            "retry_failed_questions": retry_failed_questions,
        },
        "translation_quality_rules": {
            "require_grounded_answer_alignment": bool(
                quality_cfg.get("require_grounded_answer_alignment", True)
            ),
            "min_segment_output_ratio": float(
                quality_cfg.get("min_segment_output_ratio", 0.2)
            ),
            "min_segment_source_tokens": int(
                quality_cfg.get("min_segment_source_tokens", 8)
            ),
        },
        "review_manifest": review_metadata
        or review_manifest_metadata(
            review_manifest
            or {"schema_version": 1, "samples": {}}
        ),
    }

    if cache_path.exists() and metadata_path.exists():
        cached_metadata = read_json(metadata_path)
        if cached_metadata == expected_metadata:
            rows = _read_jsonl(cache_path)
            if len(rows) == len(source_samples):
                print(f"Reusing validated translation cache: {cache_path}")
                _validate_translation_quality(
                    rows,
                    language_code=language_code,
                    quality_cfg=quality_cfg,
                    cache_dir=cache_dir,
                )
                return _samples_from_cache_rows(rows)
            print("Translation cache row count changed; rebuilding it.")
        else:
            print("Translation cache metadata changed; rebuilding it.")

    if language_code != "en" and translator_spec is None:
        raise ValueError(
            f"Language {language_code!r} requires a translator in the detection config."
        )

    rows: list[dict[str, Any]] = []
    translator = _load_translator(translator_spec) if translator_spec else None
    max_length = int(language_cfg.get("translation_max_length", 512))
    try:
        for position, source in enumerate(source_samples, start=1):
            sample_id = stable_sample_id(source)
            automatic_translated, automatic_diagnostics = _translate_sample(
                source,
                translator,
                max_length,
                language_code=language_code,
                sentence_split_fields=set(sentence_split_fields),
                preserve_numeric_answers=preserve_numeric_answers,
                retry_failed_questions=retry_failed_questions,
            )
            automatic_alignment = grounded_answer_alignment_status(
                source,
                automatic_translated,
            )
            automatic_warnings = translation_quality_warnings(
                source=source,
                translated=automatic_translated,
                diagnostics=automatic_diagnostics,
                language_code=language_code,
                require_grounded_answer_alignment=bool(
                    quality_cfg.get("require_grounded_answer_alignment", True)
                ),
                min_segment_output_ratio=float(
                    quality_cfg.get("min_segment_output_ratio", 0.2)
                ),
                min_segment_source_tokens=int(
                    quality_cfg.get("min_segment_source_tokens", 8)
                ),
            )
            applied_review = translation_review(
                review_manifest or {},
                sample_id,
                language_code,
            )
            translated, diagnostics = _apply_translation_review(
                automatic_translated,
                automatic_diagnostics,
                applied_review,
            )
            automatic_reviewed_alignment = grounded_answer_alignment_status(
                source,
                translated,
            )
            warnings = translation_quality_warnings(
                source=source,
                translated=translated,
                diagnostics=diagnostics,
                language_code=language_code,
                require_grounded_answer_alignment=bool(
                    quality_cfg.get("require_grounded_answer_alignment", True)
                ),
                min_segment_output_ratio=float(
                    quality_cfg.get("min_segment_output_ratio", 0.2)
                ),
                min_segment_source_tokens=int(
                    quality_cfg.get("min_segment_source_tokens", 8)
                ),
            )
            alignment = automatic_reviewed_alignment
            if (
                applied_review.get("accept_grounded_alignment", False)
                and automatic_reviewed_alignment == "lost"
            ):
                warnings = [
                    warning
                    for warning in warnings
                    if warning != "right_answer:grounded_alignment_lost"
                ]
                alignment = "reviewed_semantic_alignment"
            rows.append(
                {
                    "sample_index": source.index,
                    "sample_id": sample_id,
                    "language": language_code,
                    "source": {
                        "question": source.question,
                        "knowledge": source.context,
                        "right_answer": source.right_answer,
                        "hallucinated_answer": source.hallucinated_answer,
                    },
                    "translated": {
                        "question": translated.question,
                        "knowledge": translated.context,
                        "right_answer": translated.right_answer,
                        "hallucinated_answer": translated.hallucinated_answer,
                    },
                    "automatic_translated": (
                        {
                            "question": automatic_translated.question,
                            "knowledge": automatic_translated.context,
                            "right_answer": automatic_translated.right_answer,
                            "hallucinated_answer": (
                                automatic_translated.hallucinated_answer
                            ),
                        }
                        if applied_review
                        else None
                    ),
                    "translation_diagnostics": {
                        field: {
                            "source_tokens": result.source_tokens,
                            "output_tokens": result.output_tokens,
                            "input_truncated": result.input_truncated,
                            "output_truncated": result.output_truncated,
                            "segment_count": len(result.segments),
                            "strategy": result.strategy,
                            "initial_text": result.initial_text,
                            "segments": [
                                {
                                    "source_text": segment.source_text,
                                    "translated_text": segment.translated_text,
                                    "source_tokens": segment.source_tokens,
                                    "output_tokens": segment.output_tokens,
                                    "output_source_token_ratio": (
                                        segment.output_tokens
                                        / segment.source_tokens
                                        if segment.source_tokens
                                        else None
                                    ),
                                    "input_truncated": segment.input_truncated,
                                    "output_truncated": segment.output_truncated,
                                }
                                for segment in result.segments
                            ],
                        }
                        for field, result in diagnostics.items()
                    },
                    "automatic_grounded_answer_alignment": automatic_alignment,
                    "grounded_answer_alignment": alignment,
                    "automatic_translation_warnings": automatic_warnings,
                    "translation_warnings": warnings,
                    "translation_review": applied_review or None,
                }
            )
            if translator is not None and (position % 10 == 0 or position == len(source_samples)):
                print(f"Translated {position}/{len(source_samples)} samples")
    finally:
        if translator is not None:
            del translator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_jsonl(cache_path, rows)
    write_json(metadata_path, expected_metadata)
    print(f"Wrote fixed dataset cache: {cache_path}")
    _validate_translation_quality(
        rows,
        language_code=language_code,
        quality_cfg=quality_cfg,
        cache_dir=cache_dir,
    )
    return _samples_from_cache_rows(rows)


def _translate_sample(
    source: QASample,
    translator: Translator | None,
    max_length: int,
    language_code: str,
    sentence_split_fields: set[str] | None = None,
    preserve_numeric_answers: bool = True,
    retry_failed_questions: bool = True,
) -> tuple[QASample, dict[str, TranslationResult]]:
    source_fields = {
        "question": source.question,
        "knowledge": source.context,
        "right_answer": source.right_answer,
        "hallucinated_answer": source.hallucinated_answer,
    }
    if translator is None:
        diagnostics = {
            field: TranslationResult(
                text=text,
                source_tokens=0,
                output_tokens=0,
                input_truncated=False,
                output_truncated=False,
                strategy="identity",
            )
            for field, text in source_fields.items()
        }
    else:
        split_fields = sentence_split_fields or set()
        diagnostics = {}
        for field, text in source_fields.items():
            if (
                preserve_numeric_answers
                and field in {"right_answer", "hallucinated_answer"}
                and is_language_neutral_number(text)
            ):
                diagnostics[field] = TranslationResult(
                    text=text.strip(),
                    source_tokens=0,
                    output_tokens=0,
                    input_truncated=False,
                    output_truncated=False,
                    strategy="preserved_numeric",
                )
                continue
            diagnostics[field] = translator.translate_with_metadata(
                text,
                max_length=max_length,
                split_sentences=field in split_fields,
            )

        if retry_failed_questions:
            initial_question = diagnostics["question"]
            initial_score = _question_issue_score(
                source.question,
                initial_question.text,
                language_code,
            )
            retry = (
                translator.translate_question_clauses_with_metadata(
                    source.question,
                    max_length=max_length,
                    initial_text=initial_question.text,
                )
                if initial_score
                else None
            )
            if (
                retry is not None
                and _question_issue_score(
                    source.question,
                    retry.text,
                    language_code,
                )
                < initial_score
            ):
                diagnostics["question"] = retry

    translated = QASample(
        index=source.index,
        question=diagnostics["question"].text,
        context=diagnostics["knowledge"].text,
        right_answer=diagnostics["right_answer"].text,
        hallucinated_answer=diagnostics["hallucinated_answer"].text,
    )
    return translated, diagnostics


def _apply_translation_review(
    translated: QASample,
    diagnostics: dict[str, TranslationResult],
    review: dict[str, Any],
) -> tuple[QASample, dict[str, TranslationResult]]:
    fields = dict(review.get("fields", {}))
    if not fields:
        return translated, diagnostics

    reviewed_diagnostics = dict(diagnostics)
    for field, replacement_text in fields.items():
        previous = reviewed_diagnostics[field]
        reviewed_diagnostics[field] = TranslationResult(
            text=replacement_text.strip(),
            source_tokens=previous.source_tokens,
            output_tokens=previous.output_tokens,
            input_truncated=previous.input_truncated,
            output_truncated=previous.output_truncated,
            segments=(),
            strategy="review_override",
            initial_text=previous.text,
        )

    reviewed = QASample(
        index=translated.index,
        question=reviewed_diagnostics["question"].text,
        context=reviewed_diagnostics["knowledge"].text,
        right_answer=reviewed_diagnostics["right_answer"].text,
        hallucinated_answer=reviewed_diagnostics["hallucinated_answer"].text,
    )
    return reviewed, reviewed_diagnostics


def _question_issue_score(
    source_question: str,
    translated_question: str,
    language_code: str,
) -> int:
    return (
        10
        * int(
            question_intent_lost(
                source_question,
                translated_question,
                language_code,
            )
        )
        + len(missing_numbers(source_question, translated_question))
    )


def _validate_translation_quality(
    rows: list[dict[str, Any]],
    language_code: str,
    quality_cfg: dict[str, Any],
    cache_dir: Path,
) -> None:
    summary = {
        "language": language_code,
        **summarize_translation_quality(rows, quality_cfg),
    }
    quality_path = cache_dir / f"{language_code}.quality.json"
    write_json(quality_path, summary)
    write_csv(
        cache_dir / f"{language_code}.audit.csv",
        _translation_audit_rows(rows),
    )
    print(
        f"Translation quality ({language_code}): "
        f"critical={summary['critical_warning_rate']:.1%}, "
        f"grounded-alignment-loss={summary['grounded_alignment_loss_rate']:.1%}, "
        f"passed={summary['quality_gate_passed']}"
    )
    if (
        quality_cfg.get("fail_on_quality_gate", False)
        and not summary["quality_gate_passed"]
    ):
        raise RuntimeError(
            f"Translation quality gate failed for {language_code}; "
            f"inspect {quality_path} and {cache_dir / f'{language_code}.jsonl'}."
        )


def _translation_audit_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        diagnostics = row.get("translation_diagnostics", {})
        knowledge_diagnostics = diagnostics.get(
            "knowledge",
            {},
        )
        segments = list(knowledge_diagnostics.get("segments", []))
        segment_ratios = [
            float(segment["output_source_token_ratio"])
            for segment in segments
            if segment.get("output_source_token_ratio") is not None
        ]
        review = row.get("translation_review") or {}
        automatic_translated = row.get("automatic_translated") or {}
        audit_rows.append(
            {
                "sample_index": row["sample_index"],
                "sample_id": row["sample_id"],
                "language": row["language"],
                "automatic_translation_warnings": "|".join(
                    row.get("automatic_translation_warnings", [])
                ),
                "translation_warnings": "|".join(row["translation_warnings"]),
                "automatic_grounded_answer_alignment": row.get(
                    "automatic_grounded_answer_alignment",
                    "unknown",
                ),
                "grounded_answer_alignment": row.get(
                    "grounded_answer_alignment",
                    "unknown",
                ),
                "translation_review_reason": review.get("reason"),
                "reviewed_fields": "|".join(
                    sorted(review.get("fields", {}))
                ),
                "accept_grounded_alignment": review.get(
                    "accept_grounded_alignment",
                    False,
                ),
                "knowledge_segment_count": len(segments),
                "knowledge_translation_strategy": knowledge_diagnostics.get(
                    "strategy",
                    "unknown",
                ),
                "initial_knowledge_translation": knowledge_diagnostics.get(
                    "initial_text"
                ),
                "question_translation_strategy": diagnostics["question"].get(
                    "strategy",
                    "unknown",
                ),
                "initial_question_translation": diagnostics["question"].get(
                    "initial_text"
                ),
                "right_answer_translation_strategy": diagnostics[
                    "right_answer"
                ].get("strategy", "unknown"),
                "initial_right_answer_translation": diagnostics[
                    "right_answer"
                ].get("initial_text"),
                "minimum_segment_output_ratio": (
                    min(segment_ratios) if segment_ratios else None
                ),
                "source_question": row["source"]["question"],
                "translated_question": row["translated"]["question"],
                "source_knowledge": row["source"]["knowledge"],
                "translated_knowledge": row["translated"]["knowledge"],
                "source_right_answer": row["source"]["right_answer"],
                "translated_right_answer": row["translated"]["right_answer"],
                "automatic_translated_knowledge": automatic_translated.get(
                    "knowledge"
                ),
                "automatic_translated_right_answer": automatic_translated.get(
                    "right_answer"
                ),
                "source_hallucinated_answer": row["source"][
                    "hallucinated_answer"
                ],
                "translated_hallucinated_answer": row["translated"][
                    "hallucinated_answer"
                ],
                "knowledge_segments_json": json.dumps(
                    segments,
                    ensure_ascii=False,
                ),
            }
        )
    return audit_rows


def _samples_from_cache_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[QASample], dict[int, str], dict[int, list[str]]]:
    samples: list[QASample] = []
    sample_ids: dict[int, str] = {}
    warnings: dict[int, list[str]] = {}
    for row in rows:
        translated = row["translated"]
        sample_index = int(row["sample_index"])
        samples.append(
            QASample(
                index=sample_index,
                question=translated["question"],
                context=translated["knowledge"],
                right_answer=translated["right_answer"],
                hallucinated_answer=translated["hallucinated_answer"],
            )
        )
        sample_ids[sample_index] = row["sample_id"]
        warnings[sample_index] = list(row.get("translation_warnings", []))
    return samples, sample_ids, warnings


def _normalize_translator_spec(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"model": value}
    if isinstance(value, dict) and value.get("model"):
        return {
            key: value[key]
            for key in (
                "model",
                "source_lang",
                "target_lang",
                "device",
                "dtype",
                "num_beams",
            )
            if value.get(key) is not None
        }
    raise ValueError(
        "translator must be null, a model-name string, "
        "or an object with a model field"
    )


def _load_translator(spec: dict[str, Any]) -> Translator:
    print(f"Loading translator: {spec['model']}")
    return Translator(
        model_name=spec["model"],
        device=spec.get("device"),
        source_lang=spec.get("source_lang"),
        target_lang=spec.get("target_lang"),
        dtype=spec.get("dtype", "float32"),
        num_beams=int(spec.get("num_beams", 1)),
    )


def _samples_fingerprint(samples: list[QASample]) -> str:
    payload = [
        {
            "index": sample.index,
            "question": sample.question,
            "knowledge": sample.context,
            "right_answer": sample.right_answer,
            "hallucinated_answer": sample.hallucinated_answer,
        }
        for sample in samples
    ]
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_optional_path(
    value: Any,
    base_dir: Path,
) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


if __name__ == "__main__":
    main()
