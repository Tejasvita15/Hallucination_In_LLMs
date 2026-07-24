from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import token_confidence_from_scores


@dataclass
class Generation:
    text: str
    mean_confidence: float
    mean_entropy: float


@dataclass
class HFGenerator:
    model_name: str
    tokenizer: Any
    model: Any

    @classmethod
    def load(
        cls,
        model_path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> "HFGenerator":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=_dtype(dtype),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        return cls(model_path, tokenizer, model)

    def format_prompt(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Generation:
        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        do_sample = temperature > 0
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        with torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        gen_ids = output.sequences[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        confidence, entropy = token_confidence_from_scores(output.scores)
        return Generation(text=text, mean_confidence=confidence, mean_entropy=entropy)


def _dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")

