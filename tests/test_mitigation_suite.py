"""Task isolation, reasoning segmentation and scoring; no GPU/model smoke runs."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hallu_llm.mitigation_suite import make_prompt,generate_row,score_row,extract_sections,summarize,methods_for
from hallu_llm.mitigation import build_prompt


class CharacterTokenizer:
    def decode(self,ids,**kwargs): return ''.join(chr(i) for i in ids)


def fake_trace(text):
    return dict(generated_text=text,truncated=False,token_trace=[dict(token_id=ord(c),is_special=False,
        raw_context_probability=.9,raw_context_entropy_nats=.2,decoding_probability=.8,decoding_entropy_nats=.3) for c in text])


class SuiteTests(unittest.TestCase):
    def test_prompt_gold_and_reference_isolation(self):
        rows=[dict(task='grounded_generation',question='Q?',knowledge='PASSAGE',reference_answer='GOLD_SECRET'),
              dict(task='prompted_choice',question='Q?',choices=['one','two'],correct_index=1),
              dict(task='closed_book_span_detection',question='Q?',candidate='CANDIDATE',gold_spans=[[1,3]],reference='SECRET_URL')]
        for row in rows:
            for method in methods_for(row['task']):
                text=make_prompt(row,method);payload=json.loads(text.split('\n',1)[1])
                self.assertNotIn('GOLD_SECRET',text);self.assertNotIn('SECRET_URL',text)
                self.assertNotIn('correct_index',payload);self.assertNotIn('gold_spans',payload)
        self.assertNotIn('PASSAGE',make_prompt(rows[0],'cad',False))
        with self.assertRaises(ValueError): make_prompt(rows[1],'cad')

    def test_repeated_evidence_prespecified_v2(self):
        answer,preface,start,warnings=extract_sections('EVIDENCE: x\nEVIDENCE: y\nANSWER: z','evidence_first','grounded_generation')
        self.assertEqual(answer,'z');self.assertEqual(warnings,['repeated_preface_heading'])
        self.assertIsNone(extract_sections('EVIDENCE: x ANSWER: z ANSWER: a','evidence_first','grounded_generation')[0])

    def test_deepseek_reasoning_excluded(self):
        text='<think>Wrong tentative answer 1</think>CHECK: compare choices\nFINAL: 2'
        generator=SimpleNamespace(model=None,tokenizer=CharacterTokenizer(),model_name='deepseek-r1-7b',format_prompt=lambda p:p)
        row=dict(task='prompted_choice',question='Q',choices=['a','b'],correct_index=1)
        with patch('hallu_llm.mitigation_suite.generate_mitigated',return_value=fake_trace(text)):
            result=generate_row(generator,row,'claim_check',2048)
        self.assertEqual(result['final_answer'],'2');self.assertTrue(result['correct'])
        self.assertTrue(all(t['segment']!='final' for t in result['token_trace'][:text.index('FINAL:')]))

    def test_unfinished_reasoning_is_invalid(self):
        generator=SimpleNamespace(model=None,tokenizer=CharacterTokenizer(),model_name='deepseek-r1',format_prompt=lambda p:p)
        row=dict(task='prompted_choice',question='Q',choices=['a','b'],correct_index=0)
        with patch('hallu_llm.mitigation_suite.generate_mitigated',return_value=fake_trace('<think>1')):
            result=generate_row(generator,row,'baseline',2048)
        self.assertFalse(result['correct']);self.assertFalse(result['valid_for_semantic_review'])

    def test_cad_prompt_branches(self):
        generator=SimpleNamespace(model=None,tokenizer=CharacterTokenizer(),model_name='qwen',format_prompt=lambda p:p)
        row=dict(task='grounded_generation',question='TASK',knowledge='EVIDENCE_SENTINEL',reference='GOLD_SECRET')
        def decoder(proxy,source,method,*args):
            actual=proxy.format_prompt(build_prompt(source,False))
            absent=proxy.format_prompt(build_prompt(source,False,False))
            self.assertIn('EVIDENCE_SENTINEL',actual);self.assertNotIn('EVIDENCE_SENTINEL',absent)
            self.assertNotIn('GOLD_SECRET',actual+absent)
            return fake_trace('Answer')
        with patch('hallu_llm.mitigation_suite.generate_mitigated',side_effect=decoder):
            result=generate_row(generator,row,'cad',2048)
        self.assertTrue(result['valid_for_semantic_review'])

    def test_choice_invalid_stays_in_denominator(self):
        row=dict(task='prompted_choice',choices=['a','b'],correct_index=0)
        output=[]
        for text in ['1','I think 1','3']:
            trace=score_row(row,dict(valid_for_semantic_review=True,final_answer=text,truncated=False))
            output.append(trace)
        result=summarize(output,row['task'],3)
        self.assertEqual(result['prompted_choice_accuracy_all'],1/3);self.assertEqual(result['choice_invalid'],2)

    def test_span_schema_and_bounds(self):
        row=dict(task='closed_book_span_detection',candidate='hello',gold_spans=[[0,2]])
        for text,valid in [('{"hallucinated_spans": [{"start":0,"end":2}]}',True),
                           ('{"hallucinated_spans": [{"start":0,"end":6}]}',False),
                           ('{"hallucinated_spans": [{"start":0.1,"end":2}]}',False),
                           ('{"hallucinated_spans": []}',True),('{"spans": []}',False)]:
            result=score_row(row,dict(valid_for_semantic_review=True,final_answer=text))
            self.assertEqual(result['span_output_parseable'],valid,text)


if __name__=='__main__': unittest.main()
