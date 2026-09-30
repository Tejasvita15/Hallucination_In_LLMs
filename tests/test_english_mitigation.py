"""CPU-only decoder invariants; no pretrained model downloads or smoke runs."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hallu_llm.mitigation import build_prompt, contrast_log_probs, generate_mitigated, parse_final


class Inputs(dict):
    def to(self,device):
        return Inputs({k:v.to(device) for k,v in self.items()})


class Tokenizer:
    all_special_ids=[3]
    eos_token_id=3
    def __call__(self,text,**kwargs):
        ids=[0] if '"knowledge": ""' in text else [0,1]
        return Inputs(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long))
    def decode(self,ids,skip_special_tokens=True,**kwargs):
        return ''.join({0:'a',1:'b',2:'c',3:''}[i] for i in ids)
    def convert_ids_to_tokens(self,i): return str(i)


class Model:
    device=torch.device('cpu')
    config=SimpleNamespace(max_position_embeddings=100)
    generation_config=SimpleNamespace(eos_token_id=[3])
    def __init__(self,end=True): self.calls=[];self.end=end
    def __call__(self,input_ids,attention_mask,past_key_values,**kwargs):
        prefix=tuple(input_ids[0].tolist())
        full=(past_key_values or ())+prefix
        if attention_mask.shape[1]!=len(full): raise AssertionError('Wrong cache attention length')
        self.calls.append((past_key_values,prefix,full))
        contextual=full[:2]==(0,1)
        step=len(full)-(2 if contextual else 1)
        if step>=1 and self.end:
            logits=torch.tensor([-5.,-5.,-5.,5.])
        elif contextual:
            logits=torch.tensor([1.,2.,3.,-5.])
        else:
            logits=torch.tensor([3.,1.,0.,-5.])
        return SimpleNamespace(logits=logits[None,None,:],past_key_values=full)


class MitigationTests(unittest.TestCase):
    def setUp(self):
        self.source={'question':'Question?','knowledge':'Facts','right_answer':'SECRET_REFERENCE','hallucinated_answer':'SECRET_BAD'}

    def test_references_never_in_prompt(self):
        for context in (False,True):
            for evidence in (False,True):
                prompt=build_prompt(self.source,evidence,context)
                self.assertNotIn('SECRET',prompt)
                payload=json.loads(prompt.split('\n',1)[1])
                self.assertEqual(payload['knowledge'],'Facts' if context else '')

    def test_zero_alpha_and_equal_distributions(self):
        x=torch.tensor([1.,3.,2.]);y=torch.tensor([4.,2.,0.])
        raw,zero=contrast_log_probs(x,y,0)
        self.assertTrue(torch.allclose(raw,zero,atol=1e-6))
        self.assertTrue(torch.allclose(raw,contrast_log_probs(x,x,1)[1],atol=1e-6))

    def test_contrast_formula_and_separate_scores(self):
        x=torch.tensor([1.,3.,2.]);y=torch.tensor([4.,2.,0.])
        raw,cad=contrast_log_probs(x,y,1)
        self.assertTrue(torch.allclose(cad,torch.log_softmax(2*x-y,dim=-1),atol=1e-6))
        self.assertFalse(torch.allclose(raw,cad))
        with self.assertRaises(ValueError): contrast_log_probs(x,y,float('nan'))

    def test_two_caches_receive_same_generated_prefix(self):
        model=Model();g=SimpleNamespace(model=model,tokenizer=Tokenizer(),format_prompt=lambda x:x)
        result=generate_mitigated(g,self.source,'cad',5)
        self.assertEqual(len(model.calls),4)
        self.assertEqual(model.calls[2][1],model.calls[3][1])
        self.assertNotEqual(model.calls[2][0],model.calls[3][0])
        self.assertTrue(result['ended_with_eos'])
        self.assertFalse(result['truncated'])
        self.assertEqual(result['generated_tokens'],2)
        self.assertEqual(result['final_tokens'],1)
        self.assertNotEqual(result['token_trace'][0]['raw_context_probability'],result['token_trace'][0]['decoding_probability'])

    def test_baseline_single_branch_and_truncation(self):
        model=Model(end=False);g=SimpleNamespace(model=model,tokenizer=Tokenizer(),format_prompt=lambda x:x)
        result=generate_mitigated(g,self.source,'baseline',2)
        self.assertEqual(len(model.calls),2)
        self.assertTrue(result['truncated'])
        self.assertFalse(result['valid_for_semantic_review'])
        for row in result['token_trace']:
            self.assertEqual(row['raw_context_probability'],row['decoding_probability'])

    def test_final_section_parser(self):
        final,evidence,start,error=parse_final('EVIDENCE: quoted fact\nANSWER: answer',True)
        self.assertEqual(final,'answer');self.assertEqual(evidence,'quoted fact');self.assertIsNone(error)
        self.assertIsNotNone(parse_final('ANSWER: x EVIDENCE: y',True)[3])
        self.assertIsNotNone(parse_final('EVIDENCE: x ANSWER: y ANSWER: z',True)[3])
        self.assertIsNone(parse_final('Missing headings',True)[0])


if __name__=='__main__': unittest.main()
