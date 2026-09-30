"""Run the complete paired English mitigation study; no smoke/limit mode."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hallu_llm.detection_checkpoint import DetectionCheckpoint, digest
from hallu_llm.benchmark_suite import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/mitigation.english.v1.json')
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    data=ROOT/config['sources']
    source_rows=json.loads(data.read_text(encoding='utf-8'))
    rows=[r for r in source_rows if r['split']=='evaluation']
    if len(rows)!=150 or len({r['sample_id'] for r in rows})!=150:
        raise ValueError('Expected the frozen set of 150 unique evaluation sources')
    if digest(source_rows)!=config['sources_sha256']:
        raise ValueError('Source file changed; version the experiment rather than silently resuming')
    for r in rows:
        if digest(r['source'])!=r['source_sha256']:
            raise ValueError('Source content hash mismatch')
    if config['methods']!=['baseline','evidence_first','cad'] or config['cad_alpha']!=1.0:
        raise ValueError('v1 uses exactly three methods and prespecified alpha=1.0')
    from hallu_llm.models import HFGenerator
    from hallu_llm.mitigation import generate_mitigated
    from hallu_llm.utils import set_seed
    set_seed(config['seed'])
    generator=HFGenerator.load(config['model'],**config['model_loading'])
    context=dict(config=config, source_file_sha256=file_sha256(data), model_commit=getattr(generator.model.config,'_commit_hash',None),
        tokenizer_commit=generator.tokenizer.init_kwargs.get('_commit_hash'),
        chat_template=generator.tokenizer.chat_template,
        generation_config=generator.model.generation_config.to_dict(),
        decoder='manual greedy, EOS stopping; no logits processors, penalties, temperature scaling or candidate filter',
        versions={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate']},
        code={p:file_sha256(ROOT/p) for p in ['scripts/run_english_mitigation.py','src/hallu_llm/mitigation.py','src/hallu_llm/models.py']})
    directory=ROOT/config['output_dir']
    with DetectionCheckpoint(directory/'checkpoint',context) as checkpoint:
        (directory/'run_metadata.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
        temporary=directory/'predictions.tmp'
        counts={m:{'completed':0,'truncated':0,'invalid':0} for m in config['methods']}
        with temporary.open('w',encoding='utf-8') as handle:
            for i,source in enumerate(rows):
                # Rotate method order to reduce systematic warm-up/timing order effects.
                methods=config['methods'][i%3:]+config['methods'][:i%3]
                for method in methods:
                    key=[source['sample_id'],method]
                    row=checkpoint.load(key)
                    if row is None:
                        result=generate_mitigated(generator,source['source'],method,config['max_new_tokens'],config['cad_alpha'])
                        row=dict(sample_id=source['sample_id'],source_sha256=source['source_sha256'],
                            source=source['source'],model=config['model'],language='en',method=method,
                            alpha=config['cad_alpha'] if method=='cad' else None,
                            semantic_label=None,semantic_review_status='pending_researcher_annotation',**result)
                        row['generation_sha256']=digest({'sample_id':row['sample_id'],'method':method,'prompt':row['prompt'],'text':row['generated_text']})
                        checkpoint.save(key,row)
                    counts[method]['completed']+=1
                    counts[method]['truncated']+=int(row['truncated'])
                    counts[method]['invalid']+=int(not row['valid_for_semantic_review'])
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(f'Checkpointed {i+1}/150 source questions ({3*(i+1)}/450 answers)',flush=True)
        os.replace(temporary,directory/'predictions.jsonl')
        (directory/'summary.json').write_text(json.dumps(dict(expected=450,completed=450,run_complete=True,
            counts=counts,semantic_review_still_required=True,mitigation_effect_established=False),indent=2),encoding='utf-8')


if __name__=='__main__': main()
