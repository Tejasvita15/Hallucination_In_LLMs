"""Atomic per-instance checkpoints with strict provenance matching."""
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class DetectionCheckpoint:
    def __init__(self, directory, context):
        self.directory = Path(directory)
        self.context = context

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = self.directory / "writer.lock"
        try:
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"Checkpoint locked: {self.lock}. If its job has ended, remove only this stale lock before resuming.") from error
        with os.fdopen(fd, "w") as handle:
            handle.write(f"pid={os.getpid()} job={os.environ.get('PBS_JOBID', 'local')}\n")
        try:
            manifest = self.directory / "manifest.json"
            expected = {"schema": 1, "context_sha256": digest(self.context)}
            if manifest.exists():
                if json.loads(manifest.read_text(encoding="utf-8")) != expected:
                    raise ValueError("Checkpoint provenance changed; use a new experiment name.")
            else:
                self._write(manifest, expected)
        except BaseException:
            self.lock.unlink()
            raise
        return self

    def __exit__(self, *args):
        self.lock.unlink()

    def _write(self, path, value):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def load(self, key):
        path = self.directory / (digest(key) + ".json")
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["key"] != key or digest(record["row"]) != record["row_sha256"]:
            raise ValueError(f"Corrupt checkpoint: {path}")
        return record["row"]

    def save(self, key, row):
        self._write(self.directory / (digest(key) + ".json"),
                    {"key": key, "row": row, "row_sha256": digest(row)})
