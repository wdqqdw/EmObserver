"""
VECBench evaluator.

Input  : a json file produced by the inference stage, format
         {id: {"image_path", "prompt", "ground_truth", "other_info", "generations": [...]}}
         where other_info contains:
           - "candidates": List[str]  (the option pool for this sample)
           - "task":       str        (one of the 9 task tags below)

Output : a dict of metrics in the order specified by the spec.
"""

import json
import re
from typing import Dict, List


# -------- Task taxonomy ------------------------------------------------------
ID_VER_TASKS  = ["EmoSet-8", "FI-8", "WebEmo-25", "WebEmo-7"]
ID_VSA_TASKS  = ["FI-2", "WebEmo-2"]
OOD_VER_TASKS = ["UnbiasedEmo-6", "Abstract-8", "Artphoto-8"]

ALL_TASKS = ID_VER_TASKS + ID_VSA_TASKS + OOD_VER_TASKS

# Some preprocessing pipelines spell it 'UnbaisedEmo-6'; accept both.
TASK_ALIASES = {
    "UnbaisedEmo-6": "UnbiasedEmo-6",
}


# -------- Answer extraction (same logic as MVEI) -----------------------------
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer_str(generation: str) -> str:
    """
    1) <answer>...</answer> -> inner content
    2) else, take text after the latest of <answer> / </think>
    3) else, return the whole generation
    """
    if not generation:
        return ""

    m = ANSWER_RE.search(generation)
    if m:
        return m.group(1)

    text_lower = generation.lower()
    idx_ans   = text_lower.rfind("<answer>")
    idx_think = text_lower.rfind("</think>")

    cut = max(idx_ans, idx_think)
    if cut >= 0:
        tag = "<answer>" if idx_ans > idx_think else "</think>"
        return generation[cut + len(tag):]

    return generation


def parse_choice(answer_str: str, candidates: List[str]) -> str:
    """
    Iterate candidates in their original order; return the first one whose
    string occurs in answer_str (case-insensitive). Falls back to candidates[0].
    """
    if not candidates:
        return ""
    haystack = (answer_str or "").lower()
    for cand in candidates:
        if cand.lower() in haystack:
            return cand
    return candidates[0]


# -------- Evaluator ----------------------------------------------------------
class VECBenchEvaluator:
    def evaluate(self, file_path: str) -> Dict[str, float]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # task -> [n_correct, n_total]
        per_task = {t: [0, 0] for t in ALL_TASKS}
        unknown_task = 0

        for sid, rec in data.items():
            other = rec.get("other_info", {}) or {}
            raw_task = other.get("task", "")
            task = TASK_ALIASES.get(raw_task, raw_task)

            candidates = other.get("candidates") or []
            generations = rec.get("generations") or []
            generation = generations[0] if generations else ""

            answer_str = extract_answer_str(generation)
            choice = parse_choice(answer_str, candidates)
            gt = rec.get("ground_truth", "")
            correct = (choice == gt)

            if task in per_task:
                per_task[task][1] += 1
                if correct:
                    per_task[task][0] += 1
            else:
                unknown_task += 1

        def _acc(t):
            n_c, n = per_task[t]
            return (n_c / n) if n else 0.0

        def _mean(ts):
            vals = [_acc(t) for t in ts]
            return sum(vals) / len(vals) if vals else 0.0

        id_ver  = _mean(ID_VER_TASKS)
        id_vsa  = _mean(ID_VSA_TASKS)
        ood_ver = _mean(OOD_VER_TASKS)
        total   = (id_ver + id_vsa + ood_ver) / 3.0

        ordered = {
            "id_ver_acc":      id_ver,
            "id_vsa_acc":      id_vsa,
            "ood_ver_acc":     ood_ver,
            "total_acc":       total,
            "FI-8_acc":        _acc("FI-8"),
            "WebEmo-7_acc":    _acc("WebEmo-7"),
            "WebEmo-25_acc":   _acc("WebEmo-25"),
            "EmoSet-8_acc":    _acc("EmoSet-8"),
            "FI-2_acc":        _acc("FI-2"),
            "WebEmo-2_acc":    _acc("WebEmo-2"),
            "UnbiasedEmo-6_acc": _acc("UnbiasedEmo-6"),
            "Abstract-8_acc":  _acc("Abstract-8"),
            "Artphoto-8_acc":  _acc("Artphoto-8"),
        }

        if unknown_task:
            print(f"[warn] {unknown_task} samples have unknown task tag")
        # Per-task sample counts for sanity.
        sizes = ", ".join(f"{t}={per_task[t][1]}" for t in ALL_TASKS)
        print(f"[eval-VECBench] sizes: {sizes}")
        return ordered


if __name__ == "__main__":
    import sys, pprint
    pprint.pprint(VECBenchEvaluator().evaluate(sys.argv[1]))