"""Compatibility entrypoint; pass --model-name. See jobs/nscc_2ap/THESIS_RUNS.md."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "run_truthfulqa_mc.py"), run_name="__main__")
