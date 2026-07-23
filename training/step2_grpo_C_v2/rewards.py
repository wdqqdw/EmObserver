import re
from typing import List, Optional, Tuple

from swift.rewards.orm import ORM, orms


# =============================================================================
# Patterns
# =============================================================================
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def _extract_answer_lenient(text: str) -> Tuple[Optional[str], float]:
    """Lenient answer extraction.

    Returns (answer_str, format_score):
      - Full <answer>...</answer> found (1 or more):
            answer_str = content inside the LAST pair, format_score = 1.0
      - Only </think> or <answer> tag present (no matching pair):
            answer_str = text AFTER the LAST occurrence of whichever tag
                         appears latest in the text, format_score = 0.5
      - Neither present:
            answer_str = None, format_score = 0.0
    """
    if not isinstance(text, str) or not text:
        return None, 0.0

    # Case 1: complete <answer>...</answer> pair(s) -> take the LAST one
    pair_matches = list(_ANSWER_TAG_RE.finditer(text))
    if pair_matches:
        answer_str = pair_matches[-1].group(1).strip()
        return answer_str, 1.0

    # Case 2: only loose tags. Find the LAST position of either marker.
    last_answer = None
    for m in _ANSWER_OPEN_RE.finditer(text):
        last_answer = m
    last_think = None
    for m in _THINK_CLOSE_RE.finditer(text):
        last_think = m

    if last_answer is None and last_think is None:
        return None, 0.0

    # Pick whichever marker appears later in the text.
    if last_answer is None:
        chosen = last_think
    elif last_think is None:
        chosen = last_answer
    else:
        chosen = last_answer if last_answer.end() >= last_think.end() else last_think

    answer_str = text[chosen.end():].strip()
    return (answer_str if answer_str else None), 0.5


# =============================================================================
# Helpers for accuracy logic (unchanged)
# =============================================================================
def _contains_word_ci(s: str, word: str) -> bool:
    return word.lower() in s.lower()


def _contains_isolated_lower(s: str, letter: str) -> bool:
    if not s:
        return False
    for i, ch in enumerate(s):
        if ch != letter:
            continue
        left_ok = (i == 0) or (not s[i - 1].isalpha())
        right_ok = (i == len(s) - 1) or (not s[i + 1].isalpha())
        if left_ok and right_ok:
            return True
    return False


def _decide_answer(answer_str: Optional[str]) -> str:
    """Return 'A' / 'B' / 'C' following the user's priority rules."""
    if answer_str is None:
        return "C"
    s = answer_str

    # Priority 1: semantic words (check 'incorrect' first so it isn't shadowed by 'correct')
    if _contains_word_ci(s, "incorrect"):
        return "B"
    if _contains_word_ci(s, "correct"):
        return "A"

    # Priority 2: letter A/B with isolation rule for lowercase
    if ("A" in s) or _contains_isolated_lower(s, "a"):
        return "A"
    if ("B" in s) or _contains_isolated_lower(s, "b"):
        return "B"
    return "C"


def _normalize_ground_truth(sol: str) -> str:
    if not isinstance(sol, str):
        return ""
    s = sol.strip().lower()
    if s in ("a", "correct"):
        return "A"
    if s in ("b", "incorrect"):
        return "B"
    return ""


# =============================================================================
# Reward classes (registered via swift's ORM mechanism)
# =============================================================================
class MveiFormatReward(ORM):
    """Lenient format reward:
        - 1.0 if at least one <answer>...</answer> pair exists
        - 0.5 if only a loose <answer> or </think> tag exists
        - 0.0 otherwise
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for c in completions:
            _, score = _extract_answer_lenient(c)
            rewards.append(score)
        return rewards


class MveiAccuracyReward(ORM):
    """Reward = 1.0 if predicted answer matches solution; 0.0 otherwise (incl. give-up 'C').

    The predicted answer is decoded from `answer_str`, which is now produced by
    the lenient extractor (so even partially-formatted completions can be scored).
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        for c, sol in zip(completions, solution):
            gt = _normalize_ground_truth(sol)
            answer_str, _ = _extract_answer_lenient(c)
            pred = _decide_answer(answer_str)

            if pred == "C":
                rewards.append(0.0)
                continue
            rewards.append(1.0 if pred == gt else 0.0)
        return rewards


# =============================================================================
# Register into swift's global orms dict
# =============================================================================
orms["mvei_accuracy"] = MveiAccuracyReward
orms["mvei_format"]   = MveiFormatReward