"""Cache isolation regression tests; no model downloads or inference."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ThesisCacheTests(unittest.TestCase):
    def test_cache_selection_preserves_old_cache_and_shared_models(self):
        windows_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(windows_bash) if windows_bash.exists() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash unavailable")
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "datasets").mkdir()
            old = home / "datasets" / "existing.txt"
            old.write_text("keep")
            shell_home = home.as_posix()
            if os.name == "nt":
                shell_home = "/" + shell_home[0].lower() + shell_home[2:]
            for name in ('common.sh', 'prepare_thesis_login.sh', 'prepare_truthfulqa_login.sh'):
                text = (ROOT / 'jobs/aspire2a' / name).read_text()
                start = text.index(': "${HF_HOME:?Set HF_HOME before selecting the thesis cache}"')
                end = text.index('mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"', start)
                block = text[start:end] + 'mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"'
                result = subprocess.run([bash, "-c",
                    'PATH=/usr/bin:/bin:$PATH; ' + block + '\nprintf "%s\\n%s\\n" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"'],
                    cwd=ROOT, env={**os.environ, "HF_HOME": shell_home},
                    check=False, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(),
                                 [shell_home + "/datasets-thesis-3.6.0", shell_home + "/hub"])
            self.assertEqual(old.read_text(), "keep")
            self.assertTrue((home / "datasets-thesis-3.6.0").is_dir())

    def test_all_thesis_pbs_jobs_select_the_cache_before_common(self):
        for name in ["token_confidence_full", "truthfulqa_mc_full", "ragtruth_full", "mushroom_full", "qwen_crosslingual_full"]:
            text = (ROOT / f"jobs/aspire2a/{name}.pbs").read_text()
            self.assertLess(text.index("export THESIS_DATA_CACHE=1"), text.index("/common.sh"))

    def test_preparation_and_submission_use_versioned_marker(self):
        prep = (ROOT / "jobs/aspire2a/prepare_thesis_login.sh").read_text()
        submit = (ROOT / "jobs/aspire2a/submit_thesis_full.sh").read_text()
        self.assertIn('export HF_DATASETS_CACHE="$HF_HOME/datasets-thesis-3.6.0"', prep)
        self.assertIn("Path(os.environ['HF_DATASETS_CACHE'], 'thesis_prepared.json')", prep)
        self.assertIn("$HF_HOME/datasets-thesis-3.6.0/thesis_prepared.json", submit)


if __name__ == "__main__":
    unittest.main()
