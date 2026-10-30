"""Build a code-and-analysis submission ZIP without local working documents."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    tracked = subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    roots = {'src', 'scripts', 'configs', 'tests', 'jobs', 'data'}
    top = {'README.md', 'DATASETS.md', 'requirements.txt',
           'requirements-thesis.txt', 'requirements-truthfulqa.txt',
           '.gitattributes'}
    paths = set()
    for name in tracked:
        if name.startswith('jobs/nscc_2ap/'):
            name = name.replace('jobs/nscc_2ap/', 'jobs/aspire2a/', 1)
        p = Path(name)
        if not name or not (ROOT / p).is_file():
            continue
        if name in top or (p.parts[0] in roots and p.suffix.lower() not in {'.md', '.pbs'}):
            paths.add(p)
    # Explicitly include the final analysis inputs and figures, not local drafts.
    for directory in ('analysis/inputs', 'analysis/assets'):
        paths.update(p.relative_to(ROOT) for p in (ROOT / directory).rglob('*')
                     if p.is_file())
    paths.add(Path('scripts/package_submission.py'))
    required = ['README.md', 'scripts/build_thesis_assets.py',
                'analysis/inputs/human_review.json', 'requirements.txt']
    if any(Path(p) not in paths for p in required):
        raise RuntimeError('Missing required submission files')
    manifest = {'scope': 'Code and audited analysis; excludes local thesis drafts, planning notes, caches and raw execution outputs.',
                'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                'note': 'Includes working-tree changes; source_head is not an experiment-wide revision claim.',
                'files_sha256': {p.as_posix(): hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                 for p in sorted(paths)}}
    out = ROOT / 'submission'
    out.mkdir(exist_ok=True)
    target = out / 'hallucination_research_submission.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            z.write(ROOT / p, p.as_posix())
        z.writestr('SUBMISSION_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise RuntimeError('ZIP integrity verification failed')
    print(f'Verified {len(paths)} files in {target}')


if __name__ == '__main__':
    main()
