"""
MVEI evaluator.

Input  : a json file produced by the inference stage, format
         {id: {"image_path", "prompt", "ground_truth", "other_info", "generations": [...]}}

Output : a dict of metrics
         {
             "acc_total": float,
             "acc_sentiment_polarity": float,
             "acc_emotion_interpretation": float,
             "acc_scene_context": float,
             "acc_perception_subjectivity": float,
             "positive_ratio": float,    # share of answers parsed as 'A'
             "give_up_ratio":  float,    # share of answers parsed as 'C' (give up)
         }
"""

import json
import re
from typing import Dict


# Map evaluation classes to the keys used in the output dict.
CLASS_TO_KEY = {
    "sentiment polarity":      "acc_sentiment_polarity",
    "emotion interpretation":  "acc_emotion_interpretation",
    "scene context":           "acc_scene_context",
    "perception subjectivity": "acc_perception_subjectivity",
}

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer_str(generation: str) -> str:
    """
    Resolution order:
      1) If <answer>...</answer> exists, return its inner content.
      2) Else, look for the LAST <answer> or </think> tag in the text and
         return everything after it. If both exist, take the one that appears
         later (i.e. closer to the final answer).
      3) Else, return the whole generation.
    """
    if not generation:
        return ""

    # 1) Full <answer>...</answer>
    m = ANSWER_RE.search(generation)
    if m:
        return m.group(1)

    # 2) Open-only <answer> or closing </think>
    text_lower = generation.lower()
    idx_ans   = text_lower.rfind("<answer>")
    idx_think = text_lower.rfind("</think>")

    cut = max(idx_ans, idx_think)
    if cut >= 0:
        # Skip past the matched tag.
        tag = "<answer>" if idx_ans > idx_think else "</think>"
        return generation[cut + len(tag):]

    # 3) Fallback: whole string
    return generation


def _has_isolated_lower(letter: str, text: str) -> bool:
    """
    True iff `text` contains the lowercase `letter` such that neither neighbor
    character is a letter. Punctuation / digits / spaces around it are fine.
    """
    pattern = rf"(?:^|[^A-Za-z]){re.escape(letter)}(?:$|[^A-Za-z])"
    return re.search(pattern, text) is not None


def parse_choice(answer_str: str) -> str:
    """
        - 若 answer_str 中(忽视大小写)包含 'incorrect',直接判为 'B'
        - 否则若包含 'correct',直接判为 'A'
        - 'A' if the answer contains uppercase 'A', OR isolated lowercase 'a'
        - else 'B' under the same rule
        - else 'C'  (give up)
    """
    if not answer_str:
        return "C"

    # New: keyword-based shortcut (case-insensitive, no neighbor-char check).
    lower = answer_str.lower()
    if "incorrect" in lower:
        return "B"
    if "correct" in lower:
        return "A"

    if "A" in answer_str or _has_isolated_lower("a", answer_str):
        return "A"
    if "B" in answer_str or _has_isolated_lower("b", answer_str):
        return "B"
    return "C"


def is_correct(choice: str, ground_truth: str) -> bool:
    """A <-> correct, B <-> incorrect, C is always wrong."""
    gt = (ground_truth or "").strip().lower()
    if choice == "A":
        return gt == "correct"
    if choice == "B":
        return gt == "incorrect"
    return False


class MVEIEvaluator:
    def evaluate(self, file_path: str) -> Dict[str, float]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        per_class = {cls: [0, 0] for cls in CLASS_TO_KEY}
        total_correct = 0
        total = 0
        n_A = 0
        n_C = 0
        unknown_class = 0

        for sid, rec in data.items():
            generations = rec.get("generations") or []
            generation = generations[0] if generations else ""
            answer_str = extract_answer_str(generation)
            choice = parse_choice(answer_str)
            correct = is_correct(choice, rec.get("ground_truth", ""))

            cls = (
                rec.get("other_info", {})
                   .get("statement_meta", {})
                   .get("class")
            )
            if cls in per_class:
                per_class[cls][1] += 1
                if correct:
                    per_class[cls][0] += 1
            else:
                unknown_class += 1

            total += 1
            if correct:
                total_correct += 1
            if choice == "A":
                n_A += 1
            elif choice == "C":
                n_C += 1

        def _ratio(num, den):
            return num / den if den else 0.0

        metrics = {
            "acc_total":      _ratio(total_correct, total),
            "positive_ratio": _ratio(n_A, total),
            "give_up_ratio":  _ratio(n_C, total),
        }
        for cls, key in CLASS_TO_KEY.items():
            n_correct, n_cls = per_class[cls]
            metrics[key] = _ratio(n_correct, n_cls)

        ordered_keys = [
            "acc_total",
            "acc_sentiment_polarity",
            "acc_emotion_interpretation",
            "acc_scene_context",
            "acc_perception_subjectivity",
            "positive_ratio",
            "give_up_ratio",
        ]
        ordered = {k: metrics[k] for k in ordered_keys}

        if unknown_class:
            print(f"[warn] {unknown_class} samples have unknown / missing class")
        print(f"[eval-MVEI] total={total}  correct={total_correct}  "
              f"A={n_A}  C={n_C}")
        return ordered


if __name__ == "__main__":
    import sys, pprint
    ev = MVEIEvaluator()
    pprint.pprint(ev.evaluate(sys.argv[1]))