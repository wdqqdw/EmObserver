"""
LLM-as-Judge reward plugin for MVEI Expansion (multimodal emotion-statement verification).

Single fused reward produced per sample by an Azure-hosted GPT-5.5 judge that
sees the ORIGINAL image(s) + the ORIGINAL statement + the student's full
completion:

  - mvei_llm_consistency : float in [0, 1]   (score / 5, integer score 1..5)

The judge rates how well the student's <analysis> reasoning leads to its
final <answer>: clear / coherent / well-pathed analysis from which the
answer is naturally derivable. Redundancy is penalized; analysis that is
unrelated to the answer is low; analysis that contradicts the answer is 1.

Register this name in your training config's reward list.

Environment overrides:
  MVEI_JUDGE_API_KEY        required
  MVEI_JUDGE_API_VERSION    required
  MVEI_JUDGE_ENDPOINT       required
  MVEI_JUDGE_MODEL          required deployment/model name
  MVEI_JUDGE_LOG_ID         optional request log identifier
  MVEI_JUDGE_MODEL_SIZE     default "5_5"
  MVEI_JUDGE_MAX_WORKERS    default "8"
  MVEI_JUDGE_TEMPERATURE    default "0.0"
  MVEI_JUDGE_MAX_TOKENS     default "256"
  MVEI_JUDGE_DEBUG          "1" to print one judge I/O per batch
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI

from swift.rewards.orm import ORM, orms


# =============================================================================
# Azure GPT-5.5 client
# =============================================================================

API_KEY = os.environ.get("MVEI_JUDGE_API_KEY")
API_VERSION = os.environ.get("MVEI_JUDGE_API_VERSION")
AZURE_ENDPOINT = os.environ.get("MVEI_JUDGE_ENDPOINT")
LOG_ID = os.environ.get("MVEI_JUDGE_LOG_ID")

MODEL_NAMES = {"5_5": os.environ.get("MVEI_JUDGE_MODEL")}
MAX_RETRIES = 3


def _encode_image(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_user_content(prompt_text: str, image_paths: List[str]) -> List[dict]:
    content: List[dict] = []
    for p in image_paths:
        try:
            content.append({
                "type": "image_url",
                "image_url": {"url": _encode_image(p)},
            })
        except Exception as e:
            print(f"[mvei-judge][warn] cannot encode image {p}: {e!r}", flush=True)
    content.append({"type": "text", "text": prompt_text})
    return content


class _JudgeClient:
    """Thin singleton-like wrapper so we don't rebuild the client per call."""

    _instance: Optional["_JudgeClient"] = None

    def __init__(self, model_size: str = "5_5"):
        required = {
            "MVEI_JUDGE_API_KEY": API_KEY,
            "MVEI_JUDGE_API_VERSION": API_VERSION,
            "MVEI_JUDGE_ENDPOINT": AZURE_ENDPOINT,
            "MVEI_JUDGE_MODEL": MODEL_NAMES.get(model_size.lower()),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing required LLM judge environment variables: "
                + ", ".join(missing)
            )
        key = model_size.lower()
        if key not in MODEL_NAMES:
            raise ValueError(f"Unknown model_size={model_size}")
        self.model_name = MODEL_NAMES[key]
        self.client = AzureOpenAI(
            api_key=API_KEY,
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            default_headers={"X-TT-LOGID": LOG_ID} if LOG_ID else None,
        )
        self.temperature = float(os.environ.get("MVEI_JUDGE_TEMPERATURE", "0.0"))
        self.max_tokens  = int(os.environ.get("MVEI_JUDGE_MAX_TOKENS", "256"))

    @classmethod
    def get(cls) -> "_JudgeClient":
        if cls._instance is None:
            size = os.environ.get("MVEI_JUDGE_MODEL_SIZE", "5_5")
            cls._instance = cls(model_size=size)
        return cls._instance

    def call(self, prompt_text: str, image_paths: List[str]) -> str:
        content = _build_user_content(prompt_text, image_paths)
        last_err: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    stream=False,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(2 ** (attempt - 1))
        print(f"[mvei-judge][error] giving up: {last_err!r}", flush=True)
        return ""


# =============================================================================
# Prompt template (fused 5-level rubric)
# =============================================================================

_JUDGE_SYSTEM = """You are a strict reasoning-quality judge for a multimodal emotion-statement
verification task. The student model is shown an image (or several images) and
a natural-language STATEMENT that asserts something about the emotional content
of the image(s). The student must decide whether the statement is correct.

The student exposes its reasoning inside <think>...</think>. Inside that
<think> block there should be an <analysis>...</analysis> sub-block whose
content is meant to lead, step by step, to the final verdict in
<answer>...</answer>.

You will receive:
  1. The original image(s)  (attached as image inputs)
  2. The original STATEMENT (text)
  3. The student's full COMPLETION (text)

Return ONLY a single-line JSON object. No prose, no markdown fences,
no extra keys."""


_JUDGE_USER_TEMPLATE = """=== STATEMENT (about the attached image) ===
{statement}

=== STUDENT COMPLETION ===
{completion}

=== RUBRIC ===

You must produce ONE integer field, `consistency_score` in 1..5, that
jointly evaluates the LAST <analysis>...</analysis> block (inside
<think>) AND the final <answer>...</answer>.

Definitions used below:
- "derivable":   a careful reader could reach the <answer> verdict by
                 ONLY following the <analysis> content; no logical jump
                 outside the analysis is required.
- "clear path":  the analysis explicitly addresses whether the STATEMENT
                 is correct, names the key evidence, and reaches an
                 explicit verdict; it is not merely emotion description.
- "coherent":    sentences flow logically, no self-contradiction inside
                 the analysis, no irrelevant tangents.
- "answer mapping":
        "A" or "correct"   -> the model claims the STATEMENT is correct
        "B" or "incorrect" -> the model claims the STATEMENT is incorrect
        Other phrasings    -> infer the claim from natural language.

Anchors (assign the SINGLE level that fits best):

  1 = Analysis CONTRADICTS the answer (analysis concludes one way, answer
      says the other); OR there is no <analysis> block; OR the <answer>
      is missing / unparseable; OR the analysis is entirely unrelated to
      the statement (e.g. only emotion description with no judgment of
      the statement, regardless of what the answer says).

  2 = Analysis is technically not contradictory to the answer, but the
      link is weak: the analysis barely addresses the statement, or the
      verdict in the analysis is implicit and could plausibly support
      either answer; the answer is NOT clearly derivable from the
      analysis.

  3 = Analysis does target the statement and supports the answer, but
      the path is unclear: missing intermediate steps, jumps in
      reasoning, or noticeable noise/redundancy that makes derivation
      effortful. The answer is derivable only with charitable reading.

  4 = Clear, coherent analysis with an explicit verdict that matches the
      answer; the answer is naturally derivable. Minor flaws are
      acceptable: one or two redundant sentences, a small unused detail,
      slight wordiness.

  5 = Analysis is concise, coherent, and on-point: it explicitly judges
      the statement, cites the specific visual / contextual evidence
      needed, has a clear logical path, and the answer is the obvious
      conclusion. No meaningful redundancy or off-topic content.

Hard penalties (apply BEFORE picking the level):
  * If the analysis is heavily redundant (lots of repetition, padding,
    or restating the statement multiple times without adding evidence),
    cap the score at 3 even if the answer matches.
  * If the analysis spends most of its words on generic emotion
    description WITHOUT tying that description to the statement's
    correctness, cap the score at 2.
  * If the analysis's stated verdict and the <answer> disagree, the
    score is exactly 1 (no exceptions).

=== OUTPUT FORMAT (STRICT) ===
Return ONE single-line JSON object, nothing else:
{{"consistency_score": <int 1-5>}}

Valid examples:
{{"consistency_score": 5}}
{{"consistency_score": 1}}"""


# =============================================================================
# Parsing helpers
# =============================================================================

_JSON_RE = re.compile(r"\{[^{}]*\"consistency_score\"[^{}]*\}", re.DOTALL)


def _parse_judge_output(text: str) -> int:
    """Returns consistency_score in 1..5.

    Failure-safe: on any parse problem returns 1 (lowest) so training keeps moving.
    """
    if not text:
        return 1
    m = _JSON_RE.search(text)
    raw = m.group(0) if m else text.strip().splitlines()[-1].strip()
    try:
        obj = json.loads(raw)
        c = int(obj.get("consistency_score", 1))
        return max(1, min(5, c))
    except Exception:
        return 1


# =============================================================================
# Extracting STATEMENT + IMAGES from the kwargs that ms-swift hands us
# =============================================================================

def _extract_images_from_messages(messages: Any) -> List[str]:
    paths: List[str] = []
    if not isinstance(messages, list):
        return paths
    for turn in messages:
        if not isinstance(turn, dict):
            continue
        content = turn.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("image", "image_url"):
                    url = part.get("image") or part.get("image_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if isinstance(url, str):
                        paths.append(url)
                elif "image" in part and isinstance(part["image"], str):
                    paths.append(part["image"])
    return paths


def _extract_statement_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    last_text = ""
    for turn in messages:
        if not isinstance(turn, dict) or turn.get("role") != "user":
            continue
        content = turn.get("content")
        if isinstance(content, str):
            last_text = content
        elif isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(part.get("text", ""))
            if chunks:
                last_text = "\n".join(chunks)
    return last_text


def _resolve_sample_inputs(
    idx: int,
    kwargs: Dict[str, Any],
) -> Tuple[str, List[str]]:
    stmt = ""
    imgs: List[str] = []

    msgs_list = kwargs.get("messages")
    if isinstance(msgs_list, list) and idx < len(msgs_list):
        msgs = msgs_list[idx]
        stmt = _extract_statement_from_messages(msgs)
        imgs = _extract_images_from_messages(msgs)

    if not stmt:
        for key in ("prompts", "prompt", "query", "question", "statement"):
            col = kwargs.get(key)
            if isinstance(col, list) and idx < len(col) and isinstance(col[idx], str):
                stmt = col[idx]
                break

    # ms-swift keeps the textual message and image column separate for the
    # released JSONL schema. Do not return early merely because a statement was
    # found in `messages`, otherwise the multimodal judge silently receives no
    # images.
    if not imgs:
        for key in ("images", "image", "image_path", "image_paths"):
            col = kwargs.get(key)
            if isinstance(col, list) and idx < len(col):
                v = col[idx]
                if isinstance(v, str):
                    imgs = [v]
                elif isinstance(v, list):
                    imgs = [x for x in v if isinstance(x, str)]
                break

    return stmt, imgs


# =============================================================================
# Reward class
# =============================================================================

_DEBUG = os.environ.get("MVEI_JUDGE_DEBUG", "0") == "1"
_MAX_WORKERS = int(os.environ.get("MVEI_JUDGE_MAX_WORKERS", "8"))


class MveiLLMConsistencyReward(ORM):
    """Fused 5-level reward in [0, 1] = consistency_score / 5."""

    # Per-process cache: (statement, completion, frozen-images) -> float reward
    _cache: Dict[Tuple[str, str, Tuple[str, ...]], float] = {}
    _cache_max = 4096

    def _score_one(
        self,
        statement: str,
        completion: str,
        images: List[str],
    ) -> float:
        key = (statement, completion, tuple(images))
        if key in self._cache:
            return self._cache[key]

        user_prompt = _JUDGE_USER_TEMPLATE.format(
            statement=statement.strip() or "(empty statement)",
            completion=completion.strip() or "(empty completion)",
        )
        full_prompt = _JUDGE_SYSTEM + "\n\n" + user_prompt

        raw = _JudgeClient.get().call(full_prompt, images)
        c = _parse_judge_output(raw)
        reward = c / 5.0

        if _DEBUG:
            print(
                f"[mvei-judge] stmt={statement[:60]!r} "
                f"imgs={len(images)} raw={raw!r} -> c={c} reward={reward:.3f}",
                flush=True,
            )

        if len(self._cache) >= self._cache_max:
            self._cache.clear()
        self._cache[key] = reward
        return reward

    def _score_batch(
        self,
        completions: List[str],
        kwargs: Dict[str, Any],
    ) -> List[float]:
        n = len(completions)
        if n == 0:
            return []

        per_sample: List[Tuple[str, str, List[str]]] = []
        for i, comp in enumerate(completions):
            stmt, imgs = _resolve_sample_inputs(i, kwargs)
            per_sample.append((stmt, comp, imgs))

        results: List[Optional[float]] = [None] * n
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            fut2idx = {
                ex.submit(self._score_one, s, c, im): i
                for i, (s, c, im) in enumerate(per_sample)
            }
            for fut in as_completed(fut2idx):
                i = fut2idx[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    print(f"[mvei-judge][hard-fail] idx={i} {e!r}", flush=True)
                    results[i] = 1.0 / 5.0
        return [r if r is not None else (1.0 / 5.0) for r in results]

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[float]:
        return self._score_batch(completions, kwargs)


# =============================================================================
# Register
# =============================================================================

orms["mvei_llm_consistency"] = MveiLLMConsistencyReward
