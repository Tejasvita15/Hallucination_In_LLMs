"""Freeze full English task inputs; never use labels as CAD evidence."""
import argparse
import json
from pathlib import Path
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hallu_llm.benchmark_suite import read_jsonl, file_sha256
from hallu_llm.detection_checkpoint import digest


def prepare(prepared,out,hallulens=None):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): raise ValueError('Existing frozen suite: use another output directory to reprepare')
    datasets={};provenance={}
    path=prepared/'halueval_qa.jsonl'
    raw=read_jsonl(path);provenance['halueval_qa']=file_sha256(path)
    datasets['halueval_qa']=[dict(sample_id=r['sample_id'],task='grounded_generation',question=r['question'],
        knowledge=r['knowledge'],reference_answer=r['right_answer'],source_group_id=r['sample_id']) for r in raw]
    path=prepared/'ragtruth.jsonl';raw=read_jsonl(path);provenance['ragtruth']=file_sha256(path)
    groups={}
    for r in raw:
        gid=r['source_group_id']
        if gid in groups:
            old=groups[gid]
            if any(old[k]!=r[k] for k in ['context','model_input','task']): raise ValueError('Inconsistent RAGTruth source group')
        else: groups[gid]=r
    datasets['ragtruth']=[]
    for gid,r in sorted(groups.items()):
        context=r['context']
        if r['task']=='QA':
            fields=json.loads(context);question=fields['question'];knowledge=fields['passages']
        elif r['task']=='Summary':
            match=re.match(r'^Summarize the following news within (\d+) words:',r['model_input'])
            if not match: raise ValueError('Unrecognized RAGTruth summary instruction')
            question=f'Summarize the supplied news within {match.group(1)} words.';knowledge=context
        elif r['task']=='Data2txt':
            question="Write an objective overview of the supplied local business in 100–200 words. Include information from the supplied customer reviews and attribute their opinions to the reviewers. Do not invent facts."
            knowledge=context
        else: raise ValueError('Unknown RAGTruth task')
        datasets['ragtruth'].append(dict(sample_id='ragtruth_source:'+gid,source_group_id=gid,task='grounded_generation',
            question=question,knowledge=knowledge,original_task=r['task']))
    path=ROOT/'data/mitigation/truthfulqa_official817_docs.json';docs=json.loads(path.read_text(encoding='utf-8'))
    if len(docs)!=817: raise ValueError('Use all 817 official saved TruthfulQA documents, not the older 790-row CSV')
    provenance['truthfulqa']=file_sha256(path);datasets['truthfulqa']=[]
    for i,doc in enumerate(docs):
        targets=doc['mc1_targets'];order=list(range(len(targets['choices'])))
        random.Random(f'20260930:truthfulqa:{i}').shuffle(order)
        if sum(targets['labels'])!=1: raise ValueError('MC1 must have one best answer')
        datasets['truthfulqa'].append(dict(sample_id=f'truthfulqa_official:{i}',task='prompted_choice',question=doc['question'],
            choices=[targets['choices'][j] for j in order],correct_index=next(k for k,j in enumerate(order) if targets['labels'][j]==1),
            original_choice_indices=order,source_group_id=f'truthfulqa_official:{i}'))
    path=prepared/'mu_shroom.jsonl';raw=read_jsonl(path);provenance['mu_shroom']=file_sha256(path)
    datasets['mu_shroom']=[dict(sample_id=r['sample_id'],task='closed_book_span_detection',question=r['model_input'],
        candidate=r['model_output_text'],gold_spans=r['gold_spans'],source_group_id=r['sample_id']) for r in raw if r['language'].lower()=='en']
    blocked={'hallulens':'Frozen generated task prompts are absent locally. Supply --hallulens with sample_id/task/prompt JSONL. No evaluation or paid API call is launched to invent a new benchmark.'}
    if hallulens:
        raw=read_jsonl(hallulens);provenance['hallulens']=file_sha256(hallulens)
        datasets['hallulens']=[]
        for r in raw:
            if not all(r.get(k) for k in ('sample_id','task','prompt')): raise ValueError('HalluLens requires sample_id, task, prompt')
            datasets['hallulens'].append(dict(sample_id=r['sample_id'],task='closed_book_generation',question=r['prompt'],
                original_task=r['task'],reference=r.get('reference'),source_group_id=r['sample_id']))
        blocked={}
    manifest={'protocol':'mitigation_suite_en_v2','language':'en','datasets':{},'blocked':blocked,'provenance':provenance}
    for name,rows in datasets.items():
        if not rows or len({r['sample_id'] for r in rows})!=len(rows): raise ValueError('Empty or duplicated '+name)
        if len({r['task'] for r in rows})!=1: raise ValueError('Mixed task schema')
        target=out/(name+'.jsonl')
        if target.exists(): raise ValueError('Do not overwrite '+str(target))
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        manifest['datasets'][name]={'n':len(rows),'task':rows[0]['task'],'content_sha256':digest(rows),'file':target.name}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,default=ROOT/'data/benchmarks/prepared')
    p.add_argument('--output',type=Path,default=ROOT/'data/mitigation_suite_en_v2');p.add_argument('--hallulens',type=Path)
    a=p.parse_args();prepare(a.prepared,a.output,a.hallulens)
