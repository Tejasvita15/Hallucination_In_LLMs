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
    build_detection_instances,
    evaluate_detection_instances,
    stable_sample_id,
)
from hallu_llm.models import HFGenerator
from hallu_llm.translation import TranslationResult, Translator
from hallu_llm.utils import ensure_dir, read_json, set_seed, write_json, write_jsonl


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    num_samples = args.num_samples if args.num_samples is not None else int(cfg.get("num_samples", 20))
    experiment_name = args.experiment_name or cfg["experiment_name"]
    effective_cfg = dict(cfg)
    effective_cfg["num_samples"] = num_samples
    effective_cfg["experiment_name"] = experiment_name

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

    prepared_languages: dict[str, tuple[list[QASample], dict[int, str], dict[int, list[str]]]] = {}
    for language_cfg in cfg["languages"]:
        language_code = language_cfg["code"]
        print(f"\n=== Preparing dataset: {language_code} ===")
        prepared_languages[language_code] = prepare_language_dataset(
            source_samples=source_samples,
            source_fingerprint=source_fingerprint,
            language_cfg=language_cfg,
            cache_dir=ensure_dir(output_root / "datasets"),
        )

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
            )
            all_summaries.append(summary)
            run_dir = ensure_dir(output_root / model_name / language_code)
            write_json(run_dir / "summary.json", summary)
            write_jsonl(run_dir / "predictions.jsonl", rows)
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
) -> tuple[list[QASample], dict[int, str], dict[int, list[str]]]:
    language_code = language_cfg["code"]
    translator_spec = _normalize_translator_spec(language_cfg.get("translator"))
    cache_path = cache_dir / f"{language_code}.jsonl"
    metadata_path = cache_dir / f"{language_code}.meta.json"
    expected_metadata = {
        "schema_version": 1,
        "language": language_code,
        "source_fingerprint": source_fingerprint,
        "num_samples": len(source_samples),
        "translator": translator_spec,
        "translation_max_length": int(language_cfg.get("translation_max_length", 512)),
    }

    if cache_path.exists() and metadata_path.exists():
        cached_metadata = read_json(metadata_path)
        if cached_metadata == expected_metadata:
            rows = _read_jsonl(cache_path)
            if len(rows) == len(source_samples):
                print(f"Reusing validated translation cache: {cache_path}")
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
            translated, diagnostics = _translate_sample(source, translator, max_length)
            sample_id = stable_sample_id(source)
            warnings = _translation_warnings(diagnostics)
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
                    "translation_diagnostics": {
                        field: {
                            "source_tokens": result.source_tokens,
                            "output_tokens": result.output_tokens,
                            "input_truncated": result.input_truncated,
                            "output_truncated": result.output_truncated,
                        }
                        for field, result in diagnostics.items()
                    },
                    "translation_warnings": warnings,
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
    return _samples_from_cache_rows(rows)


def _translate_sample(
    source: QASample,
    translator: Translator | None,
    max_length: int,
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
            )
            for field, text in source_fields.items()
        }
    else:
        diagnostics = {
            field: translator.translate_with_metadata(text, max_length=max_length)
            for field, text in source_fields.items()
        }

    translated = QASample(
        index=source.index,
        question=diagnostics["question"].text,
        context=diagnostics["knowledge"].text,
        right_answer=diagnostics["right_answer"].text,
        hallucinated_answer=diagnostics["hallucinated_answer"].text,
    )
    return translated, diagnostics


def _translation_warnings(diagnostics: dict[str, TranslationResult]) -> list[str]:
    warnings: list[str] = []
    for field, result in diagnostics.items():
        if result.input_truncated:
            warnings.append(f"{field}:input_truncated")
        if result.output_truncated:
            warnings.append(f"{field}:output_truncated")
        if result.source_tokens and not result.text:
            warnings.append(f"{field}:empty_output")
    return warnings


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
            for key in ("model", "source_lang", "target_lang", "device")
            if value.get(key) is not None
        }
    raise ValueError("translator must be null, a model-name string, or an object with a model field")


def _load_translator(spec: dict[str, Any]) -> Translator:
    print(f"Loading translator: {spec['model']}")
    return Translator(
        model_name=spec["model"],
        device=spec.get("device"),
        source_lang=spec.get("source_lang"),
        target_lang=spec.get("target_lang"),
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


if __name__ == "__main__":
    main()
