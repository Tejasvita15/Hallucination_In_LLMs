"""Official harness MC1/MC2; full validation split, saved per-question scores."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hallu_llm.benchmark_suite import file_sha256
from hallu_llm.detection_checkpoint import DetectionCheckpoint
from hallu_llm.truthfulqa_tasks import DATASET_PATH, TASK_NAMES, task_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_suite.thesis.json")
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()
    if importlib.metadata.version("lm_eval") != "0.4.9.1":
        raise SystemExit("Install requirements-thesis.txt in the ASPIRE environment first")
    if importlib.metadata.version("transformers") != "4.51.3":
        raise SystemExit("Use the dedicated TruthfulQA environment: run bash jobs/nscc_2ap/prepare_truthfulqa_login.sh, then submit truthfulqa_mc_full.pbs")
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    model = next(m for m in config["models"] if m["name"] == args.model_name)
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    from lm_eval import evaluator
    from lm_eval.utils import handle_non_serializable
    snapshot = snapshot_download(model["path"], local_files_only=True)
    dataset = load_dataset(DATASET_PATH, "multiple_choice", split="validation")
    out = ROOT / "results/truthfulqa_mc_v1" / args.model_name
    context = {"model": model, "snapshot": snapshot, "dataset_fingerprint": dataset._fingerprint,
               "num_questions": len(dataset), "tasks": list(TASK_NAMES),
               "task_overrides": task_configs(),
               "apply_chat_template": False, "seed": config["seed"], "dtype": "bfloat16",
               "code_sha256": file_sha256(Path(__file__)),
               "versions": {p: importlib.metadata.version(p) for p in ["lm_eval", "torch", "transformers", "datasets", "peft", "accelerate", "huggingface-hub", "tokenizers"]}}
    with DetectionCheckpoint(out / "checkpoint", context):
        metadata = out / "run_metadata.json"
        metadata.write_text(json.dumps(context, indent=2), encoding="utf-8")
        if (out / "summary.json").exists():
            print("This compatible run is already complete.")
            return
        result = evaluator.simple_evaluate(model="hf", model_args={"pretrained": snapshot, "dtype": "bfloat16"},
            tasks=task_configs(), batch_size=1, device="cuda", num_fewshot=0,
            apply_chat_template=False, limit=None, log_samples=True,
            use_cache=str(out / "likelihood_cache"), bootstrap_iters=1000,
            random_seed=config["seed"], numpy_random_seed=config["seed"],
            torch_random_seed=config["seed"], fewshot_random_seed=config["seed"])
        for task in context["tasks"]:
            samples = result["samples"][task]
            if len(samples) != len(dataset):
                raise RuntimeError(f"Incomplete {task}: {len(samples)}/{len(dataset)}")
            temp = out / f"{task}.tmp"
            with temp.open("w", encoding="utf-8") as handle:
                for row in samples:
                    handle.write(json.dumps(row, ensure_ascii=False, default=handle_non_serializable) + "\n")
            temp.replace(out / f"{task}.jsonl")
        result.pop("samples")
        temp = out / "results.tmp"
        temp.write_text(json.dumps(result, indent=2, default=handle_non_serializable), encoding="utf-8")
        temp.replace(out / "results.json")
        summary = {"model": args.model_name, "run_complete": True, "num_questions": len(dataset),
                   "protocol": "official_harness_mc_no_chat_template", "metrics": result["results"]}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
