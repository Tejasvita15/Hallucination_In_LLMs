"""Dataset-location override for the pinned harness's official MC tasks.

lm-eval 0.4.9.1 merges task dictionaries into its registered task definitions.
Override only the obsolete dataset alias; retain prompts and MC1/MC2 scoring.
Return fresh dictionaries because the harness mutates them while loading tasks.
"""

DATASET_PATH = "truthfulqa/truthful_qa"
TASK_NAMES = ("truthfulqa_mc1", "truthfulqa_mc2")


def task_configs():
    return [{"task": name, "dataset_path": DATASET_PATH} for name in TASK_NAMES]
