from __future__ import annotations

from dataclasses import replace

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import QASample


class Translator:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def translate(self, text: str, max_length: int = 512) -> str:
        if not text.strip():
            return ""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.model.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)


def translate_samples(samples: list[QASample], translator: Translator | None) -> list[QASample]:
    if translator is None:
        return samples
    translated: list[QASample] = []
    for sample in samples:
        translated.append(
            replace(
                sample,
                question=translator.translate(sample.question),
                context=translator.translate(sample.context),
                right_answer=translator.translate(sample.right_answer),
                hallucinated_answer=translator.translate(sample.hallucinated_answer),
            )
        )
    return translated

