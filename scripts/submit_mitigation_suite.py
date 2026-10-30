"""Plan or submit the full English suite. No inference, smoke jobs or downloads."""
import argparse
import json
import math
import re
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hallu_llm.detection_checkpoint import digest
from hallu_llm.mitigation_suite import methods_for


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--submit',action='store_true')
    p.add_argument('--model',default='all');p.add_argument('--dataset',default='all');a=p.parse_args()
    c=json.loads((ROOT/'configs/mitigation.suite.en.v2.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/c['data_dir']/'manifest.json').read_text(encoding='utf-8'))
    models=list(c['models']) if a.model=='all' else [a.model]
    datasets=list(manifest['datasets']) if a.dataset=='all' else [a.dataset]
    if any(m not in c['models'] for m in models) or any(d not in manifest['datasets'] for d in datasets): p.error('Unknown model/dataset or dataset not prepared')
    ledger=ROOT/c['output_dir']/'submissions.jsonl';previous={}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            entry=json.loads(line);previous[(entry['dataset'],entry['model'],entry['shard'])]=entry
    count=outputs=0
    for dataset in datasets:
        entry=manifest['datasets'][dataset];n=entry['n'];shards=math.ceil(n/c['shard_size'])
        for model in models:
            for shard in range(shards):
                expected=min(c['shard_size'],n-shard*c['shard_size'])*len(methods_for(entry['task']))
                count+=1;outputs+=expected;key=(dataset,model,shard)
                directory=ROOT/c['output_dir']/dataset/model/f'shard_{shard:03d}'
                summary=directory/'summary.json';metadata=directory/'run_metadata.json'
                if summary.exists() and json.loads(summary.read_text())['run_complete']:
                    meta=json.loads(metadata.read_text())
                    if meta['config']!=c or meta['manifest_entry']!=entry: raise ValueError('Completed run provenance differs; version output directory')
                    print('Already complete:',key);continue
                if not a.submit: print(f'{dataset} {model} shard={shard} outputs={expected}');continue
                if key in previous:
                    old=previous[key]
                    if old['config_sha256']!=digest(c): raise SystemExit('Submission ledger config differs; version the output directory.')
                    status=subprocess.run(['qstat','-f',old['job_id']],capture_output=True,text=True)
                    if status.returncode==0:
                        state=re.search(r'job_state\s*=\s*([A-Z])',status.stdout)
                        if state is None: raise SystemExit('Cannot parse PBS state; refusing duplicate submission.')
                        if state.group(1) not in {'F','C'}:
                            print('Already queued/running:',key,old['job_id']);continue
                    elif 'unknown job' not in (status.stdout+status.stderr).lower():
                        raise SystemExit('Cannot establish previous job status; refusing duplicate submission: '+status.stderr)
                lock=directory/'checkpoint/writer.lock'
                if lock.exists():
                    raise SystemExit(f'Writer lock exists: {lock}. Verify its PBS job has ended before removing only the stale lock.')
                env=f'MODEL={model},DATASET={dataset},SHARD={shard}'
                result=subprocess.run(['qsub','-v',env,'jobs/aspire2a/mitigation_suite_v2.pbs'],cwd=ROOT,check=True,capture_output=True,text=True)
                job=result.stdout.strip().splitlines()[-1]
                ledger.parent.mkdir(parents=True,exist_ok=True)
                with ledger.open('a',encoding='utf-8') as f: f.write(json.dumps(dict(dataset=dataset,model=model,shard=shard,job_id=job,config_sha256=digest(c)))+'\n')
                print('Submitted:',key,job,flush=True)
    print(f'Selected scope: {count} jobs, {outputs} outputs. Completed or active jobs are skipped when submitting.')
    for name,reason in manifest.get('blocked',{}).items(): print(f'NOT INCLUDED: {name}: {reason}')


if __name__=='__main__': main()
