"""Full English task-aware comparisons, sharded and strictly resumable."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hallu_llm.benchmark_suite import read_jsonl,file_sha256
from hallu_llm.detection_checkpoint import DetectionCheckpoint,digest
from hallu_llm.mitigation_suite import methods_for,generate_row,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/mitigation.suite.en.v2.json');p.add_argument('--model',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args();config=json.loads((ROOT/a.config).read_text(encoding='utf-8'))
    if a.model not in config['models']: p.error('Unknown model')
    folder=ROOT/config['data_dir'];manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if a.dataset not in manifest['datasets']: p.error(manifest['blocked'].get(a.dataset,'Unknown dataset'))
    entry=manifest['datasets'][a.dataset];all_rows=read_jsonl(folder/entry['file'])
    if len(all_rows)!=entry['n'] or digest(all_rows)!=entry['content_sha256']: raise ValueError('Dataset provenance mismatch')
    size=config['shard_size'];shards=math.ceil(len(all_rows)/size)
    if not 0<=a.shard<shards: p.error('Invalid shard index')
    rows=all_rows[a.shard*size:(a.shard+1)*size];methods=methods_for(entry['task'])
    from hallu_llm.models import HFGenerator
    from hallu_llm.utils import set_seed
    set_seed(config['seed']);generator=HFGenerator.load(config['models'][a.model],**config['model_loading'])
    context=dict(config=config,dataset=a.dataset,manifest_entry=entry,shard=a.shard,shards=shards,model=a.model,
        model_commit=getattr(generator.model.config,'_commit_hash',None),tokenizer_commit=generator.tokenizer.init_kwargs.get('_commit_hash'),
        chat_template=generator.tokenizer.chat_template,generation_config=generator.model.generation_config.to_dict(),
        versions={k:importlib.metadata.version(k) for k in ['torch','transformers','accelerate']},
        code={k:file_sha256(ROOT/k) for k in ['scripts/run_mitigation_suite.py','src/hallu_llm/mitigation_suite.py','src/hallu_llm/mitigation.py','src/hallu_llm/models.py','src/hallu_llm/benchmark_suite.py']})
    out=ROOT/config['output_dir']/a.dataset/a.model/f'shard_{a.shard:03d}'
    started=time.monotonic();summaries={m:[] for m in methods};stop=False
    with DetectionCheckpoint(out/'checkpoint',context) as checkpoint:
        (out/'run_metadata.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
        temp=out/'predictions.tmp'
        with temp.open('w',encoding='utf-8') as f:
            for i,row in enumerate(rows):
                order=methods[i%len(methods):]+methods[:i%len(methods)]
                for method in order:
                    key=[row['sample_id'],method];result=checkpoint.load(key)
                    if result is None:
                        if time.monotonic()-started>=config['max_runtime_seconds']: stop=True;break
                        try: trace=generate_row(generator,row,method,config['max_new_tokens'],config['cad_alpha'])
                        except ValueError as error:
                            if 'exceeds context' not in str(error): raise
                            from hallu_llm.mitigation_suite import score_row
                            trace=score_row(row,dict(final_answer=None,valid_for_semantic_review=False,truncated=False,
                                parse_error='input_context_limit_exceeded',generated_text='',token_trace=[],generated_tokens=0,
                                generation_seconds=0.0,semantic_label=None))
                        result=dict(sample_id=row['sample_id'],source_group_id=row['source_group_id'],dataset=a.dataset,
                            model=a.model,method=method,task=row['task'],source=row,source_sha256=digest(row),**trace)
                        result['generation_sha256']=digest(dict(sample_id=row['sample_id'],method=method,text=result['generated_text'],source_sha256=result['source_sha256']))
                        checkpoint.save(key,result)
                    f.write(json.dumps(result,ensure_ascii=False)+'\n')
                    summaries[method].append({k:v for k,v in result.items() if k not in ['token_trace','source','prompt','without_context_prompt','generated_text','evidence_text']})
                if stop: break
                if (i+1)%25==0: print(f'{a.dataset}/{a.model}/shard {a.shard}: {i+1}/{len(rows)} sources',flush=True)
        temp.replace(out/'predictions.jsonl')
        summary={m:summarize(v,entry['task'],len(rows)) for m,v in summaries.items()}
        payload=dict(dataset=a.dataset,model=a.model,shard=a.shard,shards=shards,
            run_complete=all(s['run_complete'] for s in summary.values()),methods=summary,
            stopped_for_runtime=stop,all_dataset_rows=len(all_rows))
        (out/'summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        print(json.dumps(payload,indent=2))
    if stop: print('Partial shard saved. Resubmit the same model/dataset/shard to resume.',flush=True)


if __name__=='__main__': main()
