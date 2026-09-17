"""CPU-only diagnostic reanalysis; never overwrite predictions or confidence."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hallu_llm.reasoned_protocol import audit_review, consistency_flags


def reanalyze(rows):
    output = []
    for row in rows:
        review, flags = audit_review(row.get('reasoning', ''), row['knowledge'])
        verdict = review.get('verdict') if review else None
        mapped = {'SUPPORTED': 'grounded', 'UNSUPPORTED': 'hallucinated'}.get(verdict)
        final = row.get('generation', '').strip().upper()
        conflicts = consistency_flags(review, final == 'YES') if final in {'YES', 'NO'} else ['invalid_final_choice']
        output.append({
            'sample_id': row['sample_id'], 'candidate_type': row['candidate_type'],
            'actual_label': row['actual_label'], 'parsed_review': review,
            'review_derived_label': mapped,
            'review_derived_correct': None if mapped is None else mapped == row['actual_label'],
            'original_final_choice': final, 'parser_and_evidence_flags': flags,
            'decision_consistency_flags': conflicts,
            'reasoning_truncated': row.get('reasoning_truncated', False),
        })
    counts = Counter(flag for row in output for flag in row['parser_and_evidence_flags'])
    valid = sum(row['review_derived_label'] is not None for row in output)
    return output, {
        'instances': len(output), 'parseable_verdicts': valid,
        'review_derived_correct': sum(row['review_derived_correct'] is True for row in output),
        'review_derived_accuracy_on_parseable': (
            sum(row['review_derived_correct'] is True for row in output) / valid if valid else None),
        'parser_and_evidence_flag_counts': dict(counts),
        'final_decision_conflicts_or_invalid': sum(bool(row['decision_consistency_flags']) for row in output),
        'interpretation': 'Post-hoc review-derived diagnostic, not original final-decision performance. '
                          'All rows retained; evidence defects not excluded. No confidence is assigned '
                          'to mapped review verdicts. Manual evidence review remains necessary.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default='results/detection')
    parser.add_argument('--output', default='results/detection/saved_review_reanalysis.json')
    args = parser.parse_args()
    root = Path(args.results_root)
    report = {}
    for short, model in [('qwen', 'qwen2.5-7b'), ('llama', 'llama3.1-8b'), ('deepseek', 'deepseek-r1-7b')]:
        path = root / f'nscc_{short}_review_v2_micro' / model / 'en/predictions.jsonl'
        if not path.is_file():
            raise SystemExit(f'Missing saved predictions: {path}')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        reviewed, summary = reanalyze(rows)
        report[model] = dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             summary=summary, rows=reviewed)
        print(model + '\n' + json.dumps(summary, indent=2))
    output = Path(args.output)
    source_paths = {Path(record['source']).resolve() for record in report.values()}
    if output.resolve() in source_paths:
        raise SystemExit('Output must not overwrite source predictions')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
