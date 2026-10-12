"""Generate thesis plots/tables from audited JSON; never rerun models or invent labels."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / '.report-deps').exists():
    sys.path.insert(0, str(ROOT / '.report-deps'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.report-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODELS = ['llama3.1-8b', 'qwen2.5-7b', 'deepseek-r1-7b']
NAMES = ['Llama', 'Qwen', 'DeepSeek distilled']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']

def escaped(value):
    return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#','{':r'\{','}':r'\}'}.get(c,c) for c in str(value))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'report/submission_20261002/assets')
    args = parser.parse_args()
    out = args.out.resolve()
    for folder in ('figures','tables','data'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    provenance = {}
    def read(relative):
        p = ROOT / relative
        data = p.read_bytes()
        provenance[relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    suite = read('report/mitigation_suite_audit_20261001/analysis.json')
    base = read('report/results_audit_20260928/analysis.json')['findings']
    extra = read('report/results_audit_20260928/supplementary_analysis.json')
    review = read('report/prediction_audit_20260920/quantitative_results.json')
    if suite['integrity_errors'] or suite['rows_checked'] != 99876 or suite['shards_checked'] != 39:
        raise ValueError('Unexpected mitigation audit status; inspect before publishing.')
    stats = {(r['dataset'],r['model'],r['method']):r for r in suite['stats']}
    if len(stats) != 30 or sum(r['n'] for r in stats.values()) != suite['rows_checked']:
        raise ValueError('Duplicate conditions, missing conditions or inconsistent counts')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':250})
    inventory = []
    def table(name, headers, rows, caption):
        rows = [[f'{v:.3f}' if isinstance(v,float) else str(v) for v in row] for row in rows]
        (out/'data'/f'{name}.json').write_text(json.dumps({'columns':headers,'rows':rows,'caption':caption},indent=2),encoding='utf-8')
        md = '# '+caption+'\n\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
        md += '\n'.join('| '+' | '.join(r)+' |' for r in rows)+'\n'
        (out/'tables'/f'{name}.md').write_text(md,encoding='utf-8')
        tex = '\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+escaped(caption)+'}\n\\begin{tabular}{'+'l'*len(headers)+'}\n\\toprule\n'
        tex += ' & '.join(map(escaped,headers))+r'\\'+'\n\\midrule\n'
        tex += '\n'.join(' & '.join(map(escaped,r))+r'\\' for r in rows)
        tex += '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
        (out/'tables'/f'{name}.tex').write_text(tex,encoding='utf-8')
    def save(fig,name,caption):
        fig.tight_layout()
        for extension in ('pdf','png','svg'):
            fig.savefig(out/'figures'/f'{name}.{extension}',bbox_inches='tight')
        plt.close(fig)
        inventory.append({'name':name,'caption':caption})
    def bars(name,series,ylabel,caption,limits):
        fig,ax=plt.subplots(figsize=(7.5,4.4))
        x=np.arange(3); width=.8/len(series)
        for i,(label,values) in enumerate(series.items()):
            if not np.isfinite(values).all(): raise ValueError(name+' has nonfinite values')
            rect=ax.bar(x+(i-(len(series)-1)/2)*width,values,width,label=label,color=COLORS[i])
            ax.bar_label(rect,fmt='%.2f' if '%' in ylabel else '%.3f',fontsize=8,padding=3)
        ax.set_xticks(x,NAMES); ax.set_ylabel(ylabel); ax.set_ylim(*limits)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=min(len(series),3),frameon=False)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        save(fig,name,caption)

    official={metric:[100*base[f'truthfulqa_mc_v1/{m}/truthfulqa_{metric.lower()}.jsonl']['means']['score'] for m in MODELS] for metric in ('MC1','MC2')}
    bars('truthfulqa_official',official,'Score (%)','Official likelihood-based TruthfulQA; 817 questions. Not generated-choice accuracy.',(0,100))
    table('truthfulqa_official',['Model','MC1 (%)','MC2 (%)'],[[n,official['MC1'][i],official['MC2'][i]] for i,n in enumerate(NAMES)],inventory[-1]['caption'])

    series={}
    for analysis,label in [('strict','strict'),('recovered','post-hoc recovery')]:
        for method in ('baseline','claim_check'):
            series[f'{method.replace("_"," ")} / {label}']=[100*stats['truthfulqa',m,method][analysis+'_accuracy'] for m in MODELS]
    bars('truthfulqa_mitigation',series,'Generated-choice accuracy (%)','TruthfulQA generated choices: strict primary results and separate post-hoc recovery; 817 questions.',(0,100))
    table('truthfulqa_mitigation',['Model','Base strict %','Check strict %','Base recovery %','Check recovery %'],[[n]+[v[i] for v in series.values()] for i,n in enumerate(NAMES)],inventory[-1]['caption'])

    fig,ax=plt.subplots(figsize=(7.5,4.6)); delta_rows=[]
    for i,m in enumerate(MODELS):
        for j,analysis in enumerate(('strict','posthoc_recovery')):
            r=next(r for r in suite['comparisons'] if r['model']==m and r['analysis']==analysis)
            lo,hi=np.array(r['paired_bootstrap_95_ci'])*100; delta=100*r['delta']
            ax.errorbar(delta,i+(j-.5)*.22,xerr=[[delta-lo],[hi-delta]],fmt='o',color=COLORS[j],capsize=4,label=('Strict','Post-hoc recovery')[j] if i==0 else None)
            delta_rows.append([NAMES[i],analysis,delta,lo,hi])
    ax.axvline(0,color='#777777',linestyle='--');ax.set_yticks(range(3),NAMES);ax.set_xlabel('Claim check minus baseline (percentage points)');ax.legend(frameon=False);ax.invert_yaxis()
    save(fig,'truthfulqa_paired_deltas','Paired question bootstrap: 2,000 resamples; exploratory 95% intervals, no multiplicity correction.')
    table('truthfulqa_paired_deltas',['Model','Analysis','Delta (pp)','Lower','Upper'],delta_rows,inventory[-1]['caption'])

    for dataset,n in [('halueval_qa',10000),('ragtruth',450)]:
        series={method:[100*stats[dataset,m,method]['invalid_generation']/n for m in MODELS] for method in ('baseline','cad','evidence_first')}
        bars(dataset+'_invalid',series,'Invalid generations (%)',f'{dataset}: format/reasoning invalidity, not factual error. {n} outputs per condition.',(0,110))
        table(dataset+'_generation',['Model','Method','N','Invalid','Truncated','Mean tokens'],[[NAMES[i],method,n,stats[dataset,m,method]['invalid_generation'],stats[dataset,m,method]['truncated'],stats[dataset,m,method]['mean_tokens']] for i,m in enumerate(MODELS) for method in ('baseline','cad','evidence_first')],inventory[-1]['caption'])

    series={method:[stats['mu_shroom',m,method]['strict_metrics']['hallucinated_character_f1'] for m in MODELS] for method in ('baseline','claim_check')}
    bars('mushroom_english_mitigation',series,'Strict character F1','Mu-SHROOM English: 154 responses; detection performance, not generation hallucination reduction.',(0,.1))
    table('mushroom_english_mitigation',['Model','Method','Strict F1','Valid / 154'],[[NAMES[i],method,series[method][i],154-stats['mu_shroom',m,method]['strict_metrics']['invalid_instances']] for i,m in enumerate(MODELS) for method in series],inventory[-1]['caption'])

    rag=extra['ragtruth'];series={'Overall IoU':[rag[m]['span_level']['mean_hard_span_iou'] for m in MODELS], 'Positive-only IoU':[rag[m]['positive_only_span_iou'] for m in MODELS]}
    bars('ragtruth_iou_sensitivity',series,'Mean span IoU','RAGTruth: overall IoU credits empty/empty matches; positive-only IoU restricts to 943 positive responses.',(0,.36))
    table('ragtruth_detection',['Model','Accuracy %','Balanced %','Coverage %','Char F1','Positive IoU'],[[NAMES[i],100*rag[m]['response_level']['accuracy'],100*rag[m]['response_level']['balanced_accuracy'],100*rag[m]['response_level']['coverage'],rag[m]['span_level']['hallucinated_character_f1'],rag[m]['positive_only_span_iou']] for i,m in enumerate(MODELS)],inventory[-1]['caption'])

    mu=[base[f'mu_shroom/{m}/predictions.jsonl'] for m in MODELS]
    bars('mushroom_multilingual',{'Mean IoU':[r['means']['iou'] for r in mu]},'Mean hard-span IoU','Mu-SHROOM baseline: 1,902 responses, 14 languages; invalid predictions receive zero IoU.',(0,.4))
    table('mushroom_multilingual',['Model','IoU','Coverage %','Balanced accuracy %'],[[NAMES[i],r['means']['iou'],100*(1-r['counts']['invalid']/r['counts']['rows']),100*r['balanced_accuracy']] for i,r in enumerate(mu)],inventory[-1]['caption'])

    cross=extra['crosslingual'];fig,ax=plt.subplots(figsize=(7.5,4.5)); x=np.arange(3)
    for j,lang in enumerate(('en','zh')):
        values=[100*cross[lang][k] for k in ('accuracy','grounded_recall','hallucination_recall')]
        ax.bar_label(ax.bar(x+(j-.5)*.35,values,.35,label=lang.upper(),color=COLORS[j]),fmt='%.2f',fontsize=8)
    ax.set_xticks(x,['Accuracy','Grounded recall','Hallucination recall']);ax.set_ylim(0,110);ax.set_ylabel('Percent');ax.legend(frameon=False)
    save(fig,'crosslingual_exploratory','Exploratory Qwen EN/ZH candidate detection; 282 scored candidates per language. Translation and validator confounds remain.')
    table('crosslingual_exploratory',['Language','N','Accuracy %','Grounded recall %','Hallucination recall %'],[[lang, cross[lang]['scored_rows']]+[100*cross[lang][k] for k in ('accuracy','grounded_recall','hallucination_recall')] for lang in ('en','zh')],inventory[-1]['caption'])

    traces=[base[f'token_confidence_v1/{m}/en/predictions.jsonl'] for m in MODELS]
    bars('token_probability_descriptive',{'Complete-answer mean':[r['means']['final_token_probability_complete_only'] for r in traces]},'Mean selected-token probability','Descriptive token likelihood only. Not a calibrated factuality probability; semantic labels pending.',(0,1.08))
    table('token_probability_descriptive',['Model','Complete','Truncated','Mean probability'],[[NAMES[i],r['counts']['complete_final_answers'],r['counts']['truncated'],r['means']['final_token_probability_complete_only']] for i,r in enumerate(traces)],inventory[-1]['caption'])
    table('halueval_review_paired',['Model A','Model B','Delta pp','Lower','Upper'],[[r['a'],r['b'],r['accuracy_difference_pp'],*r['cluster_bootstrap_95_ci_pp']] for r in review['paired_comparisons']],'HaluEval candidate-review paired source-cluster comparisons; exploratory intervals.')
    captions='\n\n'.join('## '+r['name']+'\n\n'+r['caption'] for r in inventory)
    (out/'CAPTIONS.md').write_text(captions+'\n',encoding='utf-8')
    snippets='\n'.join('\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=.95\\textwidth]{assets/figures/'+r['name']+'.pdf}\n\\caption{'+escaped(r['caption'])+'}\n\\label{fig:'+r['name']+'}\n\\end{figure}\n' for r in inventory)
    (out/'figure_includes.tex').write_text(snippets,encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'inputs_sha256':provenance,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,'figures':inventory,'limits':'Saved audit inputs; no new inference, no invented human labels, no new confidence intervals.'},indent=2),encoding='utf-8')
    print(f'Generated {len(inventory)} figures (PDF/PNG/SVG) and {len(list((out/"tables").glob("*.tex")))} tables in {out}')

if __name__=='__main__':
    main()
