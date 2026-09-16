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
    decision_method: str = "direct_generation"
    reasoning: str = ""
    reasoning_truncated: bool = False
    reasoning_tokens: int = 0


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

    def generate_reasoned_binary(
        self, prompt: str, max_reasoning_tokens: int = 2048,
    ) -> BinaryGeneration:
        """Exploratory two-stage protocol: reasoning followed by forced choice.

        Scores are measured after the reasoning and explicit answer prefix.
        This is not a parsed free-form answer or the direct-generation protocol.
        """
        if max_reasoning_tokens < 1:
            raise ValueError("max_reasoning_tokens must be positive")
        formatted = self.format_prompt(
            prompt + "\nAnalyze the evidence carefully before deciding."
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            sequences = self.model.generate(
                **inputs, max_new_tokens=max_reasoning_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False, output_scores=False,
            )
        ids = sequences[0][prompt_len:]
        reasoning = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        eos_ids = getattr(self.model.generation_config, "eos_token_id", None)
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        truncated = len(ids) >= max_reasoning_tokens and int(ids[-1]) not in eos_ids

        # Score one-token labels at a new, explicit final-decision position.
        final_prompt = formatted + reasoning
        if "</think>" not in reasoning:
            final_prompt += "\n</think>"
        final_prompt += "\nFinal decision (YES = hallucinated; NO = grounded):"
        final_inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.model.device)
        label_ids = [self.tokenizer.encode(label, add_special_tokens=False)
                     for label in (" YES", " NO")]
        if any(len(tokens) != 1 for tokens in label_ids) or label_ids[0] == label_ids[1]:
            raise ValueError("Reasoned binary scoring requires distinct single-token YES/NO labels")
        with torch.no_grad():
            logits = self.model(**final_inputs).logits[0, -1].float()
        choices = [tokens[0] for tokens in label_ids]
        probabilities = torch.softmax(logits[choices], dim=-1)
        raw_mass = torch.softmax(logits, dim=-1)[choices].sum().item()
        positive, negative = probabilities.tolist()
        choice = 0 if positive >= negative else 1
        return BinaryGeneration(
            text=("YES", "NO")[choice], predicted_positive=choice == 0,
            positive_probability=positive, negative_probability=negative,
            decision_confidence=max(positive, negative),
            decision_entropy=-sum(p * math.log(max(p, 1e-12)) for p in (positive, negative)),
            label_probability_mass=raw_mass, first_generated_token_id=None,
            decision_method="reason_then_forced_choice", reasoning=reasoning,
            reasoning_truncated=truncated, reasoning_tokens=len(ids),
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
