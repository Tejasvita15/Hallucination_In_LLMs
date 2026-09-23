"""Memory-bounded generation for the new external benchmark protocol only."""
import math
from types import SimpleNamespace
from .models import HFGenerator
from .token_trace import trace_generation


class ExternalGenerator(HFGenerator):
    def generate(self, prompt, max_new_tokens, temperature=0.0, top_p=1.0):
        if temperature != 0.0:
            raise ValueError("Audited external protocol uses greedy decoding")
        trace = trace_generation(self, prompt, max_new_tokens)
        return SimpleNamespace(text=trace["final_answer"] or "", raw_text=trace["generated_text"],
            mean_confidence=trace["mean_final_token_probability"], mean_entropy=trace["mean_final_token_entropy_nats"],
            truncated=trace["truncated"], trace=trace)

    def generate_binary(self, prompt, positive_label="YES", negative_label="NO", max_new_tokens=2048):
        positive = self._label_first_token_ids(positive_label)
        negative = self._label_first_token_ids(negative_label)
        overlap = positive & negative
        positive, negative = positive - overlap, negative - overlap
        if not positive or not negative:
            raise ValueError("Binary labels do not have distinct token IDs")
        trace = trace_generation(self, prompt, max_new_tokens, [list(positive), list(negative)])
        scores = trace["first_step_label_scores"]
        p, n = scores["positive_probability"], scores["negative_probability"]
        return SimpleNamespace(text=trace["final_answer"] or "", raw_text=trace["generated_text"],
            predicted_positive=p >= n, **scores, decision_confidence=max(p, n),
            decision_entropy=-sum(v * math.log(max(v, 1e-30)) for v in (p, n)),
            truncated=trace["truncated"], trace=trace)
