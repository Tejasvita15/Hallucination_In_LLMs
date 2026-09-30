"""Merge complete shards and recompute dataset-level metrics; never average shard scores."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hallu_llm.benchmark_suite import read_jsonl
from hallu_llm.detection_checkpoint import digest
from hallu_llm.mitigation_suite import methods_for,summarize,score_row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--dataset',required=True);a=p.parse_args()
    c=json.loads((ROOT/'configs/mitigation.suite.en.v2.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/c['data_dir']/'manifest.json').read_text(encoding='utf-8'));entry=manifest['datasets'][a.dataset]
    sources=read_jsonl(ROOT/c['data_dir']/entry['file']);expected_ids={r['sample_id'] for r in sources}
    source_map={r['sample_id']:r for r in sources}
    if digest(sources)!=entry['content_sha256']: raise ValueError('Source hash changed')
    folder=ROOT/c['output_dir']/a.dataset/a.model;methods=methods_for(entry['task']);counts={m:[] for m in methods};seen=set()
    identity=None
    for shard in range(math.ceil(entry['n']/c['shard_size'])):
        path=folder/f'shard_{shard:03d}'
        if not json.loads((path/'summary.json').read_text())['run_complete']: raise ValueError('Incomplete shard '+str(shard))
        meta=json.loads((path/'run_metadata.json').read_text())
        stable={k:v for k,v in meta.items() if k!='shard'}
        if identity is None: identity=stable
        elif identity!=stable: raise ValueError('Shards differ in code/model/config provenance')
        if meta['config']!=c or meta['manifest_entry']!=entry or meta['model']!=a.model or meta['dataset']!=a.dataset: raise ValueError('Run identity mismatch')
        shard_ids={r['sample_id'] for r in sources[shard*c['shard_size']:(shard+1)*c['shard_size']]}
        rows=read_jsonl(path/'predictions.jsonl');local=set()
        for r in rows:
            key=(r['sample_id'],r['method'])
            if key in seen or r['sample_id'] not in shard_ids or r['method'] not in methods: raise ValueError('Duplicate/unexpected row')
            if r['source']!=source_map[r['sample_id']] or digest(r['source'])!=r['source_sha256']: raise ValueError('Source hash mismatch')
            expected_hash=digest(dict(sample_id=r['sample_id'],method=r['method'],text=r['generated_text'],source_sha256=r['source_sha256']))
            if expected_hash!=r['generation_sha256']: raise ValueError('Generation hash mismatch')
            recalculated=score_row(r['source'],dict(r))
            for field in ('predicted_choice','choice_valid','correct','predicted_spans','span_output_parseable','gold_spans'):
                if field in recalculated and r.get(field)!=recalculated[field]: raise ValueError('Stored scoring differs from recomputation')
            local.add(key);seen.add(key)
            counts[r['method']].append({k:v for k,v in r.items() if k not in ['source','token_trace','generated_text','prompt','without_context_prompt','evidence_text']})
        if local!={(s,m) for s in shard_ids for m in methods}: raise ValueError('Incomplete source-method coverage in shard')
    if seen!={(s,m) for s in expected_ids for m in methods}: raise ValueError('Incomplete dataset')
    result={'dataset':a.dataset,'model':a.model,'run_complete':True,
            'methods':{m:summarize(v,entry['task'],entry['n']) for m,v in counts.items()}}
    (folder/'aggregate_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
