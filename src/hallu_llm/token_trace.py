"""Greedy token traces without retaining a vocabulary tensor for every step.

Probabilities are from the processed generation distribution, not factuality
probabilities. Special tokens remain in the trace and are excluded from means.
"""
import math


class GreedyTrace:
    def __init__(self, label_ids=None):
        self.rows = []
        self.label_ids = label_ids
        self.label_scores = None

    def __call__(self, input_ids, scores):
        import torch
        if scores.shape[0] != 1:
            raise ValueError("Token tracing requires batch size one and greedy decoding")
        logp = torch.log_softmax(scores[0].float(), dim=-1)
        if self.label_ids and not self.rows:
            log_masses = torch.stack([torch.logsumexp(logp[ids], dim=0) for ids in self.label_ids])
            conditional = log_masses.softmax(dim=0).tolist()
            self.label_scores = {"positive_probability": conditional[0], "negative_probability": conditional[1],
                                 "label_probability_mass": float(log_masses.logsumexp(dim=0).exp())}
        token_id = int(scores[0].argmax())
        finite = torch.isfinite(logp)
        entropy = float(-(logp[finite].exp() * logp[finite]).sum())
        chosen = float(logp[token_id])
        if not math.isfinite(chosen) or not math.isfinite(entropy):
            raise ValueError("Non-finite token distribution")
        self.rows.append({"token_id": token_id, "log_probability": chosen,
                          "probability": math.exp(chosen), "entropy_nats": entropy})
        return scores


def trace_generation(generator, prompt, max_new_tokens=2048, label_ids=None):
    import torch
    from transformers import LogitsProcessorList
    tokenizer, model = generator.tokenizer, generator.model
    formatted = generator.format_prompt(prompt)
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(model.device)
    n = inputs["input_ids"].shape[1]
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit and n + max_new_tokens > context_limit:
        raise ValueError(f"Prompt ({n}) + output budget ({max_new_tokens}) exceeds context ({context_limit}); input was not silently truncated")
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    collector = GreedyTrace(label_ids)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                num_beams=1, num_return_sequences=1,
                                logits_processor=LogitsProcessorList([collector]),
                                return_dict_in_generate=False, output_scores=False,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=eos)
    ids = output[0, n:].tolist()
    if ids != [r["token_id"] for r in collector.rows]:
        raise RuntimeError("Generated IDs differ from captured greedy scores")
    raw = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    reasoning_expected = "deepseek-r1" in generator.model_name.lower() or "<think>" in formatted or "<think>" in text
    final_start = raw.rfind("</think>") + len("</think>") if "</think>" in raw else None
    final_present = not reasoning_expected or final_start is not None
    special = set(tokenizer.all_special_ids)
    reached_final = final_present and final_start is None
    for i, row in enumerate(collector.rows):
        row["token_piece"] = tokenizer.convert_ids_to_tokens(ids[i])
        row["is_special"] = ids[i] in special
        # Conservative boundary: a token straddling </think> is excluded.
        if final_start is not None and not reached_final:
            prefix = tokenizer.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            reached_final = len(prefix) >= final_start
        row["segment"] = "final" if reached_final else "reasoning_or_unfinished"
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    ended = bool(ids) and ids[-1] in eos_ids
    final_rows = [r for r in collector.rows if r["segment"] == "final" and not r["is_special"]]
    final_text = tokenizer.decode([r["token_id"] for r in final_rows], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False).strip() if final_present else None
    return {"generated_text": text, "final_answer": final_text, "final_answer_present": bool(final_text),
            "truncated": len(ids) >= max_new_tokens and not ended, "ended_with_eos": ended,
            "prompt_tokens": n, "generated_tokens": len(ids), "token_trace": collector.rows,
            "first_step_label_scores": collector.label_scores,
            "mean_final_token_probability": (sum(r["probability"] for r in final_rows) / len(final_rows) if final_rows else None),
            "mean_final_token_entropy_nats": (sum(r["entropy_nats"] for r in final_rows) / len(final_rows) if final_rows else None),
            "confidence_semantics": "processed greedy next-token likelihood; not factual correctness"}
