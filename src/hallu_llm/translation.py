from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import QASample
from .text import split_english_sentences


@dataclass(frozen=True)
class TranslationSegment:
    source_text: str
    translated_text: str
    source_tokens: int
    output_tokens: int
    input_truncated: bool
    output_truncated: bool


@dataclass(frozen=True)
class TranslationResult:
    text: str
    source_tokens: int
    output_tokens: int
    input_truncated: bool
    output_truncated: bool
    segments: tuple[TranslationSegment, ...] = ()


class Translator:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
        dtype: str = "float32",
        num_beams: int = 1,
    ) -> None:
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        if num_beams < 1:
            raise ValueError("num_beams must be at least 1")
        self.num_beams = num_beams
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if "nllb" in model_name.lower() and (not source_lang or not target_lang):
            raise ValueError(
                "NLLB requires both source_lang and target_lang, for example "
                "'eng_Latn' and 'zho_Hans'."
            )
        tokenizer_kwargs = {"src_lang": source_lang} if source_lang else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if source_lang and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = source_lang
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=_translation_dtype(dtype),
        ).to(self.device)
        self.model.eval()

    def translate(
        self,
        text: str,
        max_length: int = 512,
        split_sentences: bool = False,
    ) -> str:
        return self.translate_with_metadata(
            text,
            max_length=max_length,
            split_sentences=split_sentences,
        ).text

    def translate_with_metadata(
        self,
        text: str,
        max_length: int = 512,
        split_sentences: bool = False,
    ) -> TranslationResult:
        if not text.strip():
            return TranslationResult("", 0, 0, False, False)
        source_segments = (
            split_english_sentences(text)
            if split_sentences
            else [text.strip()]
        )
        segments = tuple(
            self._translate_segment(segment, max_length=max_length)
            for segment in source_segments
        )
        return TranslationResult(
            text=" ".join(
                segment.translated_text
                for segment in segments
                if segment.translated_text
            ),
            source_tokens=sum(segment.source_tokens for segment in segments),
            output_tokens=sum(segment.output_tokens for segment in segments),
            input_truncated=any(segment.input_truncated for segment in segments),
            output_truncated=any(segment.output_truncated for segment in segments),
            segments=segments,
        )

    def _translate_segment(
        self,
        text: str,
        max_length: int,
    ) -> TranslationSegment:
        full_ids = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.model.device)
        generate_kwargs = {
            "max_length": max_length,
            "num_beams": self.num_beams,
            "do_sample": False,
        }
        if self.target_lang:
            target_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
            if target_id is None or target_id == self.tokenizer.unk_token_id:
                raise ValueError(
                    f"Unknown target language token {self.target_lang!r} "
                    f"for {self.model_name}"
                )
            generate_kwargs["forced_bos_token_id"] = target_id
        with torch.no_grad():
            ids = self.model.generate(**inputs, **generate_kwargs)
        translated = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        output_tokens = int(ids.shape[1])
        ended_with_eos = bool(
            self.tokenizer.eos_token_id is not None
            and output_tokens
            and int(ids[0, -1].item()) == self.tokenizer.eos_token_id
        )
        return TranslationSegment(
            source_text=text,
            translated_text=translated,
            source_tokens=len(full_ids),
            output_tokens=output_tokens,
            input_truncated=len(full_ids) > max_length,
            output_truncated=output_tokens >= max_length and not ended_with_eos,
        )


def _translation_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported translation dtype: {name}")


def translate_samples(samples: list[QASample], translator: Translator | None) -> list[QASample]:
    if translator is None:
        return samples
    translated: list[QASample] = []
    for sample in samples:
        translated.append(
            replace(
                sample,
                question=translator.translate(sample.question),
                context=translator.translate(sample.context, split_sentences=True),
                right_answer=translator.translate(sample.right_answer),
                hallucinated_answer=translator.translate(sample.hallucinated_answer),
            )
        )
    return translated
