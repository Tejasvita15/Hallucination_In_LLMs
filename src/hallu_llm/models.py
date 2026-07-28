from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .metrics import token_confidence_from_scores


@dataclass
class Generation:
    text: str
    mean_confidence: float
    mean_entropy: float


@dataclass
class BinaryGeneration:
    text: str
    predicted_positive: bool
    positive_probability: float
    negative_probability: float
    decision_confidence: float
    decision_entropy: float
    label_probability_mass: float
    first_generated_token_id: int | None


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
        from transformers import AutoModelForCausalLM, AutoTokenizer

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

    def generate_binary(
        self,
        prompt: str,
        positive_label: str = "YES",
        negative_label: str = "NO",
        max_new_tokens: int = 4,
    ) -> BinaryGeneration:
        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        if not output.scores:
            raise RuntimeError("Model returned no token scores for the binary decision.")

        positive_ids = self._label_first_token_ids(positive_label)
        negative_ids = self._label_first_token_ids(negative_label)
        overlap = positive_ids & negative_ids
        positive_ids -= overlap
        negative_ids -= overlap
        if not positive_ids or not negative_ids:
            raise RuntimeError(
                f"Could not identify distinct first tokens for labels {positive_label!r} and {negative_label!r}."
            )

        first_probs = torch.softmax(output.scores[0][0].float(), dim=-1)
        raw_positive = float(first_probs[list(positive_ids)].sum().item())
        raw_negative = float(first_probs[list(negative_ids)].sum().item())
        label_mass = raw_positive + raw_negative
        if label_mass <= 0:
            raise RuntimeError("The model assigned zero probability mass to both binary labels.")

        positive_probability = raw_positive / label_mass
        negative_probability = raw_negative / label_mass
        predicted_positive = positive_probability >= negative_probability
        confidence = max(positive_probability, negative_probability)
        entropy = -sum(
            probability * math.log(max(probability, 1e-12))
            for probability in (positive_probability, negative_probability)
        )

        generated_ids = output.sequences[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        first_token_id = int(generated_ids[0].item()) if generated_ids.numel() else None
        return BinaryGeneration(
            text=text,
            predicted_positive=predicted_positive,
            positive_probability=positive_probability,
            negative_probability=negative_probability,
            decision_confidence=confidence,
            decision_entropy=entropy,
            label_probability_mass=label_mass,
            first_generated_token_id=first_token_id,
        )

    def _label_first_token_ids(self, label: str) -> set[int]:
        casings = {label, label.lower(), label.title()}
        variants = casings | {f" {variant}" for variant in casings}
        token_ids: set[int] = set()
        for variant in variants:
            encoded = self.tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                token_ids.add(int(encoded[0]))
        return token_ids


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
