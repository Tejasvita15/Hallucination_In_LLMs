"""Stream the full suite archive; preserve strict metrics and label post-hoc recovery."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hallu_llm.detection_checkpoint import digest
from hallu_llm.benchmark_suite import read_jsonl,span_metrics,span_iou
from hallu_llm.mitigation_suite import make_prompt,score_row,methods_for


def recover(row):
    """Never select using gold labels; reject truncation and ambiguous alternatives."""
    if row['truncated']: return None
    text=row['generated_text'].strip()
    if 'deepseek-r1' in row['model'].lower():
        if text.count('</think>')!=1: return None
        text=text.split('</think>')[1].strip()
    if row['task']=='prompted_choice':
        if re.fullmatch(r'[1-9][0-9]*',text): return text
        # One terminal standalone answer declaration; no competing declarations.
        matches=list(re.finditer(r'(?im)^\s*(?:FINAL|ANSWER)\s*:\s*([1-9][0-9]*)\s*[.]?\s*$',text))
        if len(matches)==1 and not text[matches[0].end():].strip(): return matches[0].group(1)
        return None
    if row['task']=='closed_book_span_detection':
        start=text.find('{')
        if start<0:return None
        try:obj,end=json.JSONDecoder().raw_decode(text[start:])
        except ValueError:return None
        if any(c in text[start+end:] for c in '{}'):return None
        if not isinstance(obj,dict) or set(obj)!={'hallucinated_spans'}:return None
        return json.dumps(obj)
    return None


def paired_ci(values,seed=20261001,reps=2000):
    rng=random.Random(seed);n=len(values)
    means=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(reps))
    return [means[int(.025*(reps-1))],means[int(.975*(reps-1))]]


def run(archive,out):
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/mitigation_suite_en_v2/manifest.json').read_text())
    sources={d:{r['sample_id']:r for r in read_jsonl(ROOT/'data/mitigation_suite_en_v2'/v['file'])} for d,v in manifest['datasets'].items()}
    groups=defaultdict(list);errors=Counter();warnings=Counter();counts=Counter();reasons=defaultdict(Counter);costs=defaultdict(lambda:Counter());examples=[]
    def check(ok,message):
        if not ok:errors[message]+=1
    def close(a,b):return a is b if a is None or b is None else math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if not info.is_dir() and info.file_size==0:
                warnings['empty_checkpoint_manifest' if info.filename.endswith('/checkpoint/manifest.json') else 'empty_checkpoint_record' if '/checkpoint/' in info.filename else 'empty_other_file']+=1
        summaries=sorted(n for n in z.namelist() if n.endswith('/summary.json'))
        check(len(summaries)==39,'expected_39_shards');seen=set();identities={}
        for name in summaries:
            directory=name.rsplit('/',1)[0];summary=json.loads(z.read(name));meta=json.loads(z.read(directory+'/run_metadata.json'))
            d,m,shard=summary['dataset'],summary['model'],summary['shard']
            check(summary['run_complete'],'incomplete_shard')
            check(meta['manifest_entry']==manifest['datasets'][d],'dataset_manifest')
            cp_bytes=z.read(directory+'/checkpoint/manifest.json')
            if cp_bytes:
                cp=json.loads(cp_bytes)
                check(cp['context_sha256']==digest(meta),'checkpoint_metadata_hash')
            identity={k:v for k,v in meta.items() if k!='shard'}
            if (d,m) in identities: check(identity==identities[d,m],'inconsistent_shard_provenance')
            identities[d,m]=identity
            for path,sha in meta['code'].items():check(hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==sha,'code_hash:'+path)
            shard_counts=Counter();strict_correct=Counter();strict_invalid=Counter();trunc=Counter();span_rows=defaultdict(list)
            expected_ids=set(list(sources[d])[shard*meta['config']['shard_size']:(shard+1)*meta['config']['shard_size']])
            with z.open(directory+'/predictions.jsonl') as f:
                for line in f:
                    r=json.loads(line);method=r['method'];key=(d,m,method);pair=(d,m,r['sample_id'],method)
                    check(pair not in seen,'duplicate_row');seen.add(pair);shard_counts[method]+=1
                    check(r['sample_id'] in expected_ids,'wrong_shard');check(r['source']==sources[d][r['sample_id']],'source_content')
                    check(digest(r['source'])==r['source_sha256'],'source_hash')
                    check(digest(dict(sample_id=r['sample_id'],method=method,text=r['generated_text'],source_sha256=r['source_sha256']))==r['generation_sha256'],'generation_hash')
                    if r.get('prompt') is not None:
                        check(r['prompt']==make_prompt(r['source'],method),'prompt')
                        check(r['without_context_prompt']==(make_prompt(r['source'],method,False) if method=='cad' else None),'counterfactual_prompt')
                    scored=score_row(r['source'],dict(r))
                    for field in ('correct','choice_valid','predicted_choice','predicted_spans','span_output_parseable','gold_spans'):
                        if field in scored:check(scored[field]==r[field],'score:'+field)
                    tokens=r['token_trace'];check(len(tokens)==r['generated_tokens'],'token_count')
                    selected=[t for t in tokens if t['segment']=='final' and not t['is_special']]
                    if 'final_tokens' in r:
                        check(len(selected)==r['final_tokens'],'final_token_count')
                        for field in ('raw_context_probability','raw_context_entropy_nats','decoding_probability','decoding_entropy_nats'):
                            val=sum(t[field] for t in selected)/len(selected) if selected else None
                            check(close(val,r['mean_final_'+field]),'token_mean:'+field)
                    counts[key]+=1;costs[key]['seconds']+=r['generation_seconds'];costs[key]['tokens']+=r['generated_tokens']
                    costs[key]['invalid']+=not r['valid_for_semantic_review'];costs[key]['truncated']+=r['truncated']
                    reasons[key][r['parse_error'] or 'valid']+=1;strict_invalid[method]+=not r['valid_for_semantic_review'];trunc[method]+=r['truncated']
                    compact={k:r[k] for k in ['sample_id','method','task','valid_for_semantic_review','truncated']}
                    if d=='truthfulqa':
                        compact.update(correct=r['correct'],choice_valid=r['choice_valid'])
                        strict_correct[method]+=r['correct']
                    if d=='mu_shroom':
                        compact.update(gold_spans=r['gold_spans'],predicted_spans=r['predicted_spans'],span_output_parseable=r['span_output_parseable'])
                        span_rows[method].append(compact)
                    if d in ('truthfulqa','mu_shroom'):
                        recovered=recover(r)
                        alt=score_row(r['source'],dict(r,final_answer=recovered,valid_for_semantic_review=bool(recovered))) if recovered is not None else dict(r)
                        # Never replace an already valid strict decision with a recovered one.
                        strict_valid=r.get('choice_valid',r.get('span_output_parseable',False))
                        if strict_valid:alt=r
                        compact['recovered_correct']=alt.get('correct')
                        compact['recovered_valid']=alt.get('choice_valid',alt.get('span_output_parseable',False))
                        compact['recovered_spans']=alt.get('predicted_spans')
                        if not strict_valid and compact['recovered_valid'] and len(examples)<24:
                            examples.append(dict(dataset=d,model=m,method=method,sample_id=r['sample_id'],recovered=recovered,original_parse_error=r['parse_error']))
                    groups[key].append(compact)
            for method,expected in summary['methods'].items():
                check(shard_counts[method]==expected['expected']==expected['completed'],'shard_coverage')
                check(strict_invalid[method]==expected['invalid_generation'] and trunc[method]==expected['truncated'],'summary_invalid')
                if d=='truthfulqa':check(close(strict_correct[method]/shard_counts[method],expected['prompted_choice_accuracy_all']),'summary_accuracy')
                if d=='mu_shroom':
                    metrics=span_metrics(span_rows[method])
                    for k in ['hallucinated_character_f1','mean_hard_span_iou','invalid_instances']:check(close(metrics[k],expected['span_metrics'][k]),'summary_span:'+k)
            print(f'Checked {d}/{m}/shard {shard}: {sum(shard_counts.values())} rows',flush=True)
        for d,entry in manifest['datasets'].items():
            for m in ['qwen2.5-7b','llama3.1-8b','deepseek-r1-7b']:
                for method in methods_for(entry['task']):check(counts[d,m,method]==entry['n'],'dataset_coverage')
    stats=[];comparisons=[]
    for (d,m,method),rows in sorted(groups.items()):
        c=costs[d,m,method];s=dict(dataset=d,model=m,method=method,n=len(rows),invalid_generation=c['invalid'],truncated=c['truncated'],mean_tokens=c['tokens']/len(rows),mean_seconds=c['seconds']/len(rows),parse_reasons=dict(reasons[d,m,method]))
        if d=='truthfulqa':s.update(correct=sum(r['correct'] for r in rows),strict_accuracy=mean(r['correct'] for r in rows),strict_valid=sum(r['choice_valid'] for r in rows),recovered_accuracy=mean(r['recovered_correct'] for r in rows),recovered_valid=sum(r['recovered_valid'] for r in rows))
        if d=='mu_shroom':
            s['strict_metrics']=span_metrics(rows)
            alternative=[dict(r,predicted_spans=r['recovered_spans'],span_output_parseable=r['recovered_valid']) for r in rows]
            s['recovered_metrics']=span_metrics(alternative)
            positive=[r for r in rows if r['gold_spans']]
            s['positive_only_strict_metrics']=span_metrics(positive) if positive else None
        stats.append(s)
    for model in ['deepseek-r1-7b','llama3.1-8b','qwen2.5-7b']:
        b={r['sample_id']:r for r in groups['truthfulqa',model,'baseline']};c={r['sample_id']:r for r in groups['truthfulqa',model,'claim_check']}
        for mode,field in [('strict','correct'),('posthoc_recovery','recovered_correct')]:
            differences=[int(c[s][field])-int(b[s][field]) for s in sorted(b)]
            comparisons.append(dict(dataset='truthfulqa',model=model,analysis=mode,delta=mean(differences),paired_bootstrap_95_ci=paired_ci(differences),claim_only_correct=differences.count(1),baseline_only_correct=differences.count(-1)))
    result=dict(rows_checked=sum(counts.values()),shards_checked=len(summaries),integrity_errors=dict(errors),archive_warnings=dict(warnings),stats=stats,comparisons=comparisons,
        recovery_rule='Post-hoc: nontruncated final response only; unique terminal line FINAL:/ANSWER: integer, or sole JSON object with hallucinated_spans. Strict valid decisions retained. No gold-guided extraction.',
        bootstrap='2000 paired question resamples; seed 20261001; marginal exploratory intervals, no multiplicity correction.',
        checks_limitations='Only nonempty checkpoint manifests verified; 33 manifests and 91770 checkpoint records are empty in the received archive. Duplicate per-row checkpoints were not reread. Complete vocabulary distributions and tokenizer replay unavailable; entropy/CAD distributions not independently reconstructed.')
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'recovery_examples.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
    lines=['# Full English mitigation suite audit','',f"Checked {result['rows_checked']:,} prediction rows across {len(summaries)} shards. Integrity discrepancies: {sum(errors.values())}.",'',
        'Coverage, original source content, prompts, source/generation hashes, saved code hashes (LF-normalized), available nonempty checkpoint manifests, token counts/means and directly scoreable labels were checked. Full token-vocabulary distributions and per-row duplicate checkpoint files were not independently replayed. Original archive and metrics were not changed.',
        '', 'Archive limitation: 91,770 checkpoint records and 33 checkpoint manifests are zero bytes. None of the 39 prediction files, summaries or run-metadata files are empty. The predictions remain analyzable, but this ZIP is not a complete resumable checkpoint backup. Preserve the original cluster directory and verify it before any cleanup. The cause of empty checkpoint files is unknown.', '',
        '## TruthfulQA: prompted-choice accuracy','',
        'These are generated-choice results, not official likelihood-based MC1/MC2. Invalid outputs count as incorrect. Claim checking requires a CHECK preface and FINAL heading in the strict protocol. Missing structure can dominate results, especially for DeepSeek.','',
        '| Model | Baseline strict | Claim-check strict | Baseline recovery | Claim-check recovery |','|---|---:|---:|---:|---:|']
    for m in ['deepseek-r1-7b','llama3.1-8b','qwen2.5-7b']:
        pair={s['method']:s for s in stats if s['dataset']=='truthfulqa' and s['model']==m};b,c=pair['baseline'],pair['claim_check']
        lines.append(f"| {m} | {b['strict_accuracy']:.2%} | {c['strict_accuracy']:.2%} | {b['recovered_accuracy']:.2%} | {c['recovered_accuracy']:.2%} |")
    lines+=['','Recovery is a separately reported post-hoc sensitivity analysis. It rejects truncated outputs and ambiguous declarations and does not consult correct labels. It cannot be presented as prespecified or substituted silently for strict metrics.','', '| Model / analysis | Claim-check minus baseline | Paired 95% interval |','|---|---:|---:|']
    for c in comparisons:lines.append(f"| {c['model']} / {c['analysis']} | {100*c['delta']:+.2f} pp | [{100*c['paired_bootstrap_95_ci'][0]:+.2f}, {100*c['paired_bootstrap_95_ci'][1]:+.2f}] pp |")
    lines+=['','Intervals use 2,000 paired question resamples and are exploratory, without multiple-comparison correction.','', '## Mu-SHROOM English: span detection','',
        'These measure detector quality, not fewer hallucinations in newly generated answers. Strict JSON and offset failures count as invalid. Scores are low in absolute terms.','',
        '| Model / method | Strict character F1 | Recovery character F1 | Strict valid / 154 |','|---|---:|---:|---:|']
    for s in stats:
        if s['dataset']=='mu_shroom':lines.append(f"| {s['model']} / {s['method']} | {s['strict_metrics']['hallucinated_character_f1']:.4f} | {s['recovered_metrics']['hallucinated_character_f1']:.4f} | {154-s['strict_metrics']['invalid_instances']} |")
    lines+=['','## Source-grounded generation: factuality pending','',
        'HaluEval and RAGTruth outputs still require semantic annotation. Existing dataset labels refer to old candidate responses and cannot label these new generations. No generation-level hallucination reduction is established by this audit.','',
        '| Dataset / model / method | Generated | Format/reasoning invalid | Truncated | Mean output tokens |','|---|---:|---:|---:|---:|']
    for s in stats:
        if s['dataset'] in ('halueval_qa','ragtruth'):lines.append(f"| {s['dataset']} / {s['model']} / {s['method']} | {s['n']} | {s['invalid_generation']} | {s['truncated']} | {s['mean_tokens']:.1f} |")
    lines+=['','## Next steps','',
        'Preserve strict results and report format adherence separately from factual correctness. Review post-hoc recovery examples before using sensitivity results. Predefine and annotate a manageable source-matched, model/condition-masked sample of HaluEval/RAGTruth outputs; do not choose cases by favorable confidence or correctness. Do not rerun all experiments merely to repair formatting. If a new prompt/format protocol is tested, version it as a new experiment and retain this run. Hindi remains deferred.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(rows=result['rows_checked'],errors=dict(errors),comparisons=comparisons),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'report/mitigation_suite_audit_20261001')
    a=p.parse_args();run(a.archive,a.output)
