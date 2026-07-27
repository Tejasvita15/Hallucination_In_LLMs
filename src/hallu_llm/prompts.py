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

DETECTION_INSTRUCTIONS = {
    "en": (
        "Determine whether the candidate answer contains hallucinated information. "
        "Use only the provided knowledge. Answer YES if any factual claim in the "
        "candidate answer is unsupported by or contradicts the knowledge. Answer NO "
        "if the candidate answer is fully supported. Output exactly YES or NO with "
        "no explanation."
    ),
    "zh": (
        "请判断候选答案是否包含幻觉信息。只能依据给出的知识。如果候选答案中的任何事实陈述"
        "无法从知识中得到支持或与知识矛盾，请回答 YES；如果候选答案完全受到知识支持，"
        "请回答 NO。只输出 YES 或 NO，不要解释。"
    ),
}

DETECTION_FIELD_LABELS = {
    "en": ("Knowledge", "Question", "Candidate answer", "Decision"),
    "zh": ("知识", "问题", "候选答案", "判断"),
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


def halueval_detection_prompt(
    knowledge: str,
    question: str,
    candidate_answer: str,
    language_code: str = "en",
    instruction: str | None = None,
) -> str:
    instruction = instruction or DETECTION_INSTRUCTIONS.get(language_code, DETECTION_INSTRUCTIONS["en"])
    knowledge_label, question_label, candidate_label, decision_label = DETECTION_FIELD_LABELS.get(
        language_code,
        DETECTION_FIELD_LABELS["en"],
    )
    return (
        f"{instruction}\n\n"
        f"{knowledge_label}: {knowledge}\n\n"
        f"{question_label}: {question}\n\n"
        f"{candidate_label}: {candidate_answer}\n\n"
        f"{decision_label}:"
    )
