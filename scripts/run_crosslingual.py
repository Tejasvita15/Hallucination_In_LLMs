from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hallu_llm.data import load_halueval_qa
from hallu_llm.evaluation import evaluate_qa_samples
from hallu_llm.models import HFGenerator
from hallu_llm.translation import Translator, translate_samples
from hallu_llm.utils import ensure_dir, read_json, set_seed, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured cross-lingual hallucination experiments.")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    set_seed(int(cfg.get("seed", 42)))

    output_root = ensure_dir(Path(cfg.get("output_dir", "results/structured")) / cfg["experiment_name"])
    samples_en = load_halueval_qa(
        num_samples=int(cfg.get("num_samples", 150)),
        seed=int(cfg.get("seed", 42)),
        dataset_cfg=cfg.get("dataset", {}),
    )
    write_json(output_root / "config.json", cfg)
    print(f"Loaded {len(samples_en)} valid HaluEval QA samples")

    gen_cfg = cfg.get("generation", {})
    all_summaries = []

    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        model_path = model_cfg["path"]
        print(f"\n=== Loading model: {model_name} ({model_path}) ===")
        generator = HFGenerator.load(
            model_path,
            dtype=gen_cfg.get("dtype", "bfloat16"),
            device_map=gen_cfg.get("device_map", "auto"),
        )

        for lang_cfg in cfg["languages"]:
            lang_code = lang_cfg["code"]
            print(f"\n--- Evaluating {model_name} on {lang_code} ---")
            translator_name = lang_cfg.get("translator")
            translator = Translator(translator_name) if translator_name else None
            samples = translate_samples(samples_en, translator)

            summary, rows = evaluate_qa_samples(
                generator=generator,
                samples=samples,
                model_name=model_name,
                language_code=lang_code,
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 96)),
                temperature=float(gen_cfg.get("temperature", 0.0)),
                top_p=float(gen_cfg.get("top_p", 1.0)),
                high_confidence_threshold=float(cfg.get("confidence", {}).get("high_confidence_threshold", 0.75)),
                prompt_prefix=lang_cfg.get("prompt_prefix"),
            )
            all_summaries.append(summary)
            run_dir = ensure_dir(output_root / model_name / lang_code)
            write_json(run_dir / "summary.json", summary)
            write_jsonl(run_dir / "samples.jsonl", rows)
            print(summary)

            if translator is not None:
                del translator

        del generator

    write_json(output_root / "summary_all.json", all_summaries)
    print(f"\nDone. Results saved to {output_root}")


if __name__ == "__main__":
    main()

