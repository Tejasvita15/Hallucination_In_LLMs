"""Single generated verdict; invalid responses stay in the denominator."""
import time
from collections import Counter
from .reasoned_protocol import review_prompt, audit_review


def summarize_reviews(rows, expected, model, language, resumed):
    scored = [r for r in rows if not r['excluded_from_primary_metrics']]
    valid = [r for r in scored if r['predicted_label'] is not None]
    correct = sum(r['prediction_correct'] for r in scored)
    by_label = {label: [r for r in scored if r['actual_label'] == label]
                for label in ('grounded', 'hallucinated')}
    rates = {label: (sum(r['prediction_correct'] for r in items) / len(items) if items else None)
             for label, items in by_label.items()}
    flags = Counter(f for r in rows for f in r['review_quality_flags'])
    return {
        'protocol': 'single_stage_evidence_review_v3', 'model': model, 'language': language,
        'expected_instances': expected, 'completed_instances': len(rows),
        'run_complete': len(rows) == expected, 'resumed_instances': resumed,
        'scored_instances': len(scored), 'excluded_instances': len(rows)-len(scored),
        'valid_verdict_instances': len(valid), 'invalid_instances': len(scored)-len(valid),
        'coverage': len(valid)/len(scored) if scored else None,
        'accuracy_all_instances': correct/len(scored) if scored else None,
        'accuracy_valid_only': correct/len(valid) if valid else None,
        'grounded_recall_all_instances': rates['grounded'],
        'hallucination_recall_all_instances': rates['hallucinated'],
        'balanced_accuracy_all_instances': (sum(rates.values())/2 if all(v is not None for v in rates.values()) else None),
        'review_flag_counts': dict(flags),
        'protocol_quality_passed': len(rows) == expected and not flags,
        'semantic_review_still_required': True,
        'confidence_metrics_available': False,
        'interpretation': 'Invalid or truncated verdicts count as incorrect in all-instance accuracy. '
                          'Evidence-format flags are retained, not used to select favorable rows. '
                          'Review verdicts have no attached second-stage confidence.',
    }


def evaluate_reviews(generator, instances, model, language, budget, checkpoint,
                     max_runtime_seconds=82800):
    if max_runtime_seconds <= 0:
        raise ValueError('max_runtime_seconds must be positive')
    started = time.monotonic()
    rows, resumed = [], 0
    for instance in instances:
        key = [str(instance.sample_id), instance.candidate_type]
        cached = checkpoint.load(key)
        if cached is not None:
            rows.append(cached)
            resumed += 1
            continue
        if time.monotonic() - started >= max_runtime_seconds:
            break
        prompt = review_prompt(instance.knowledge, instance.question, instance.candidate_answer)
        result = generator.generate_review(prompt, budget)
        review, flags = audit_review(result['text'], instance.knowledge)
        if result['truncated']:
            flags.append('reasoning_truncated')
        verdict = review.get('verdict') if review else None
        predicted = {'SUPPORTED': 'grounded', 'UNSUPPORTED': 'hallucinated'}.get(verdict)
        if result['truncated']:
            predicted = None
        actual = 'hallucinated' if instance.is_hallucinated else 'grounded'
        row = {
            'sample_id': instance.sample_id, 'sample_index': instance.sample_index,
            'candidate_type': instance.candidate_type, 'model': model, 'language': language,
            'actual_label': actual, 'predicted_label': predicted,
            'prediction_correct': predicted == actual,
            'prediction_source': 'generated_review_verdict' if predicted else 'invalid',
            'decision_mode': 'single_stage_evidence_review_v3',
            'knowledge': instance.knowledge, 'question': instance.question,
            'candidate_answer': instance.candidate_answer, 'prompt': prompt,
            'reasoning': result['text'], 'reasoning_tokens': result['tokens'],
            'reasoning_truncated': result['truncated'], 'evidence_review': review,
            'review_quality_flags': flags,
            'excluded_from_primary_metrics': instance.excluded_from_primary_metrics,
            'source_review_reason': instance.source_review_reason,
        }
        checkpoint.save(key, row)
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f'Checkpointed {len(rows)}/{len(instances)} decisions', flush=True)
    return summarize_reviews(rows, len(instances), model, language, resumed), rows
