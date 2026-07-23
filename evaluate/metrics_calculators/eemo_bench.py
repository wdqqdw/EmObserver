"""
EEmo-Bench evaluator (works for both Single and Pair subsets).

Usage: subset must be specified at evaluate() time:
    EEmoBenchEvaluator().evaluate(file_path, subset="single")
    EEmoBenchEvaluator().evaluate(file_path, subset="pair")

Output dict always carries the same 6 keys; entries belonging to the OTHER
subset are filled with None so per-subset runs can be merged later.
"""

import json
import re
from typing import Dict, Optional


# -------- Answer extraction (shared with MVEI / VECBench) --------------------
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer_str(generation: str):
    """
    Returns (answer_str, from_tag) where from_tag is True only when the string
    was extracted from a full <answer>...</answer> block.
    """
    if not generation:
        return "", False
    m = ANSWER_RE.search(generation)
    if m:
        return m.group(1), True
    text_lower = generation.lower()
    idx_ans   = text_lower.rfind("<answer>")
    idx_think = text_lower.rfind("</think>")
    cut = max(idx_ans, idx_think)
    if cut >= 0:
        tag = "<answer>" if idx_ans > idx_think else "</think>"
        return generation[cut + len(tag):], False
    return generation, False

# Match the LAST {...} block in the prompt; non-greedy body, allow newlines.
CAND_DICT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_candidates(prompt: str) -> Optional[dict]:
    """
    Find the LAST '{...}' fragment in `prompt` and parse it as a dict.
    Tries json.loads first (after single->double quote swap), then ast.literal_eval.
    Returns None on any failure.
    """
    if not prompt:
        return None
    matches = CAND_DICT_RE.findall(prompt)
    if not matches:
        return None
    frag = matches[-1]

    # Try ast.literal_eval first (handles single quotes natively).
    import ast
    try:
        obj = ast.literal_eval(frag)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    try:
        obj = json.loads(frag.replace("'", '"'))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _has_isolated_lower(letter: str, text: str) -> bool:
    """Lowercase letter with no adjacent ASCII letter on either side."""
    pattern = rf"(?:^|[^A-Za-z]){re.escape(letter)}(?:$|[^A-Za-z])"
    return re.search(pattern, text) is not None


CHOICES = ["A", "B", "C", "D"]


def parse_choice(answer_str: str) -> str:
    """
    Iterate A,B,C,D in order. For each letter return it if either:
      - the uppercase form occurs in answer_str, OR
      - an isolated lowercase form occurs.
    Fallback to 'A' if nothing matches.
    """
    if not answer_str:
        return "A"
    for ch in CHOICES:
        if ch in answer_str or _has_isolated_lower(ch.lower(), answer_str):
            return ch
    return "A"


# -------- Question-type bucket ----------------------------------------------
QTYPE_TO_BUCKET = {
    "Yes-or-No": "yes_no",
    "What/How":  "what_how",
}


# -------- Output schema ------------------------------------------------------
OUTPUT_KEYS = [
    "acc_single_yes_no",
    "acc_single_what_how",
    "acc_single_overall",
    "acc_pair_yes_no",
    "acc_pair_what_how",
    "acc_pair_overall",
]


class EEmoBenchEvaluator:
    def evaluate(self, file_path: str, subset: Optional[str] = None) -> Dict:
        if subset not in ("single", "pair"):
            raise ValueError(
                f"EEmoBenchEvaluator.evaluate requires subset='single' or 'pair', got {subset!r}"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # bucket -> [n_correct, n_total]
        buckets = {"yes_no": [0, 0], "what_how": [0, 0]}
        unknown_qtype = 0

        for sid, rec in data.items():
            generations = rec.get("generations") or []
            generation = generations[0] if generations else ""
            answer_str, from_tag = extract_answer_str(generation)

            gt = (rec.get("ground_truth") or "").strip()

            choice = None
            # New rule: only when answer came from <answer>...</answer>.
            if from_tag:
                cand = extract_candidates(rec.get("prompt", ""))
                if cand and gt in cand:
                    gt_text = str(cand[gt])
                    if gt_text and gt_text.lower() in answer_str.lower():
                        choice = gt   # treat as picking the ground truth

            # Fallback to the original letter-based parsing.
            if choice is None:
                choice = parse_choice(answer_str)

            correct = (choice == gt)

            qtype = (rec.get("other_info", {}) or {}).get("question_type")
            bucket = QTYPE_TO_BUCKET.get(qtype)
            if bucket is None:
                unknown_qtype += 1
                continue
            buckets[bucket][1] += 1
            if correct:
                buckets[bucket][0] += 1

        def _acc(b):
            n_c, n = buckets[b]
            return (n_c / n) if n else 0.0

        yn   = _acc("yes_no")
        what = _acc("what_how")
        overall = (yn + what) / 2.0   # un-weighted mean as specified

        # Fill the keys for our subset, leave the other subset as None.
        result = {k: None for k in OUTPUT_KEYS}
        result[f"acc_{subset}_yes_no"]   = yn
        result[f"acc_{subset}_what_how"] = what
        result[f"acc_{subset}_overall"]  = overall

        if unknown_qtype:
            print(f"[warn] {unknown_qtype} samples with unknown question_type")
        print(f"[eval-EEmo-{subset}] yes_no={buckets['yes_no']}  what_how={buckets['what_how']}")
        return result


if __name__ == "__main__":
    import sys, pprint
    ev = EEmoBenchEvaluator()
    pprint.pprint(ev.evaluate(sys.argv[1], subset=sys.argv[2]))