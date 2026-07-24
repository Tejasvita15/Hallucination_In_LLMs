from __future__ import annotations


PROMPT_PREFIXES = {
    "en": "Answer the following question based on the given context. Be factual and concise.",
    "zh": "请根据给定的背景信息回答问题。请保持事实准确并简洁。",
    "hi": "दिए गए संदर्भ के आधार पर प्रश्न का उत्तर दें। तथ्यात्मक और संक्षिप्त रहें।",
    "ar": "أجب عن السؤال بناء على السياق المعطى. كن دقيقا ومختصرا.",
    "id": "Jawab pertanyaan berikut berdasarkan konteks yang diberikan. Jawablah secara faktual dan ringkas.",
    "es": "Responde la siguiente pregunta basándote en el contexto dado. Sé factual y conciso.",
}

FIELD_LABELS = {
    "en": ("Context", "Question", "Answer"),
    "zh": ("背景", "问题", "回答"),
    "hi": ("संदर्भ", "प्रश्न", "उत्तर"),
    "ar": ("السياق", "السؤال", "الإجابة"),
    "id": ("Konteks", "Pertanyaan", "Jawaban"),
    "es": ("Contexto", "Pregunta", "Respuesta"),
}


def qa_prompt(question: str, context: str, language_code: str = "en", prefix: str | None = None) -> str:
    prefix = prefix or PROMPT_PREFIXES.get(language_code, PROMPT_PREFIXES["en"])
    context_label, question_label, answer_label = FIELD_LABELS.get(language_code, FIELD_LABELS["en"])
    return (
        f"{prefix}\n\n"
        f"{context_label}: {context}\n\n"
        f"{question_label}: {question}\n\n"
        f"{answer_label}:"
    )

