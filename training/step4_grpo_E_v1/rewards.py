import re
from typing import List, Optional, Tuple

from swift.rewards.orm import ORM, orms


# =============================================================================
# Lenient extractor (used by accuracy reward; unchanged behavior)
# =============================================================================
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def _extract_answer_lenient(text: str) -> Tuple[Optional[str], float]:
    """Lenient answer extraction (used by accuracy reward).

    Returns (answer_str, _legacy_score). The score is kept for backward-compat
    but is no longer used by the new strict format reward.
    """
    if not isinstance(text, str) or not text:
        return None, 0.0

    pair_matches = list(_ANSWER_TAG_RE.finditer(text))
    if pair_matches:
        answer_str = pair_matches[-1].group(1).strip()
        return answer_str, 1.0

    last_answer = None
    for m in _ANSWER_OPEN_RE.finditer(text):
        last_answer = m
    last_think = None
    for m in _THINK_CLOSE_RE.finditer(text):
        last_think = m

    if last_answer is None and last_think is None:
        return None, 0.0

    if last_answer is None:
        chosen = last_think
    elif last_think is None:
        chosen = last_answer
    else:
        chosen = last_answer if last_answer.end() >= last_think.end() else last_think

    answer_str = text[chosen.end():].strip()
    return (answer_str if answer_str else None), 0.5


# =============================================================================
# Strict format reward
# =============================================================================
# Sub-blocks required inside <think>...</think>, IN THIS ORDER.
_REQUIRED_INNER_TAGS = ["element", "human", "context", "interaction", "analysis"]

# A "joiner" between adjacent required tags / between outer </think> and <answer>:
# either nothing, or exactly one newline character.
_JOIN_RE = r"(?:\n)?"

# Build the strict outer regex:
#   ^<think>...</think>(\n)?<answer>...</answer>$
# with leading/trailing whitespace tolerated (so trailing "\n" from the
# tokenizer/template doesn't kill the score).
_STRICT_OUTER_RE = re.compile(
    r"^\s*<think>(?P<inner>.*?)</think>" + _JOIN_RE +
    r"<answer>(?P<answer>.*?)</answer>\s*$",
    re.DOTALL,   # case-sensitive on purpose; tags must match exactly
)

# Build the strict inner regex:
#   <element>...</element>(\n)?<human>...</human>(\n)?...<analysis>...</analysis>
# Each block must be present, in order, with at most a single "\n" between them.
def _build_inner_regex() -> re.Pattern:
    parts = []
    for i, tag in enumerate(_REQUIRED_INNER_TAGS):
        parts.append(rf"<{tag}>(?P<{tag}>.*?)</{tag}>")
        if i != len(_REQUIRED_INNER_TAGS) - 1:
            parts.append(_JOIN_RE)
    pattern = r"^" + "".join(parts) + r"$"
    return re.compile(pattern, re.DOTALL)


_STRICT_INNER_RE = _build_inner_regex()


def _check_strict_format(text: str) -> bool:
    """Return True iff `text` satisfies all three format constraints."""
    if not isinstance(text, str) or not text:
        return False

    outer = _STRICT_OUTER_RE.match(text)
    if outer is None:
        return False

    inner = outer.group("inner")
    if _STRICT_INNER_RE.match(inner) is None:
        return False

    return True


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

    if _contains_word_ci(s, "incorrect"):
        return "B"
    if _contains_word_ci(s, "correct"):
        return "A"

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
    """Strict format reward (1.0 / 0.0):

    Requires ALL of:
      1. Overall shape: <think>...</think><answer>...</answer>
         (leading/trailing whitespace OK; between outer </think> and <answer>
         either nothing or a single '\\n').
      2. The gap between </think> and <answer> is empty or exactly one '\\n'.
      3. Inside <think>...</think>, the following sub-blocks appear IN THIS
         ORDER, each present exactly once:
             <element>...</element>
             <human>...</human>
             <context>...</context>
             <interaction>...</interaction>
             <analysis>...</analysis>
         Adjacent sub-blocks are joined by nothing or by a single '\\n'.

    Any deviation -> 0.0.
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return [1.0 if _check_strict_format(c) else 0.0 for c in completions]


class MveiAccuracyReward(ORM):
    """Reward = 1.0 if predicted answer matches solution; 0.0 otherwise.

    Predicted answer is decoded from the LAST <answer>...</answer> pair
    (lenient extractor unchanged), so partially-formatted completions can
    still be scored independently of the strict format reward.
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