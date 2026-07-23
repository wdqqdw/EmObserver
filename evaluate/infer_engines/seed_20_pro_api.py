"""
Seed-2.0-Pro inference wrapper backed by ByteDance Ark (Volc Engine) gateway.

Interface mirrors Qwen36PlusInferencer / GPT55Inferencer:
    model = Seed20ProInferencer(model_size="2_0_pro")
    out   = model.inference(input_dict)                       # n=1, temp=0.7
    out   = model.inference(input_dict,
                            sampling_params=SamplingParams(n=2, temperature=1.0))

Input  : {id: {"image_path": str | List[str], "prompt": str, ...}}
Output : {id: {..., "generations": List[str]}}   # length == n
"""

import base64
import copy
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
import os
from evaluate.infer_engines.paths import require_env

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# ----------------------------------------------------------------------------- 
# API configuration is supplied through environment variables.
# ----------------------------------------------------------------------------- 
API_KEY = os.environ.get("SEED_API_KEY")
BASE_URL = os.environ.get("SEED_BASE_URL")


# size tag -> deployment / endpoint id on the Ark gateway
MODEL_NAMES: Dict[str, str] = {
    "2_0_pro": os.environ.get("SEED_MODEL"),
}

SUPPORTED_SIZES: List[str] = list(MODEL_NAMES.keys())

DEFAULT_MAX_WORKERS = 16
MAX_RETRIES = 3


def _encode_image(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_user_content(prompt: str, image_paths: List[str]) -> List[dict]:
    content: List[dict] = []
    for p in image_paths:
        content.append({
            "type":      "image_url",
            "image_url": {"url": _encode_image(p)},
        })
    content.append({"type": "text", "text": prompt})
    return content


def _sp_to_kwargs(sp: Any) -> Dict[str, Any]:
    defaults = {"n": 1, "temperature": 0.7, "max_tokens": 1024}
    if sp is None:
        return defaults
    out = dict(defaults)
    for k in ("n", "temperature", "top_p", "max_tokens"):
        v = getattr(sp, k, None) if not isinstance(sp, dict) else sp.get(k)
        if v is not None:
            out[k] = v
    return out


class Seed20ProInferencer:
    def __init__(self, model_size: str = "2_0_pro"):
        require_env("SEED_API_KEY", "SEED_BASE_URL", "SEED_MODEL")
        key = model_size.lower()
        if key not in MODEL_NAMES:
            raise ValueError(
                f"Unknown model_size={model_size!r}. Supported: {SUPPORTED_SIZES}"
            )
        self.model_size = key
        self.model_name = MODEL_NAMES[key]
        self._client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
        )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _normalize_paths(image_path: Union[str, List[str]]) -> List[str]:
        return [image_path] if isinstance(image_path, str) else list(image_path)

    def _call_one(self, sample: Dict, api_kwargs: Dict) -> List[str]:
        image_paths = self._normalize_paths(sample["image_path"])
        content = _build_user_content(sample["prompt"], image_paths)

        last_err: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    stream=False,
                    #**api_kwargs,
                )
                return [c.message.content or "" for c in resp.choices]
            except Exception as e:
                last_err = e
                time.sleep(2 ** (attempt - 1))
        n = api_kwargs.get("n", 1)
        print(f"[seed2pro][error] {last_err!r}")
        return [""] * n

    # ----------------------------------------------------------------- public
    def inference(
        self,
        input: Dict[str, Dict],
        gpu_n: Optional[int] = None,
        sampling_params: Any = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> Dict[str, Dict]:
        if gpu_n is not None:
            print(f"[seed2pro] gpu_n={gpu_n} ignored (API engine).")

        api_kwargs = _sp_to_kwargs(sampling_params)
        ids = list(input.keys())
        result: Dict[str, Dict] = {sid: copy.deepcopy(input[sid]) for sid in ids}
        total = len(ids)

        n_ok = 0
        n_fail = 0
        start = time.time()

        def _make_iter(future_iter):
            if _HAS_TQDM:
                return tqdm(
                    future_iter,
                    total=total,
                    desc=f"[seed2pro:{self.model_size}]",
                    dynamic_ncols=True,
                    smoothing=0.1,
                )
            return future_iter

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(self._call_one, input[sid], api_kwargs): sid
                for sid in ids
            }
            iterator = _make_iter(as_completed(futures))
            done = 0
            for fut in iterator:
                sid = futures[fut]
                try:
                    gens = fut.result()
                    if any(g for g in gens):
                        n_ok += 1
                    else:
                        n_fail += 1
                    result[sid]["generations"] = gens
                except Exception as e:
                    print(f"[seed2pro][hard-fail] sid={sid} err={e!r}")
                    result[sid]["generations"] = [""] * api_kwargs.get("n", 1)
                    n_fail += 1
                done += 1

                if _HAS_TQDM:
                    iterator.set_postfix(ok=n_ok, fail=n_fail, refresh=False)
                else:
                    if done % 50 == 0 or done == total:
                        elapsed = time.time() - start
                        rate = done / elapsed if elapsed > 0 else 0.0
                        eta = (total - done) / rate if rate > 0 else float("inf")
                        print(f"[seed2pro] {done}/{total}  ok={n_ok} fail={n_fail}  "
                              f"{rate:.2f} it/s  eta={eta:.0f}s")

        print(f"[seed2pro] finished: {n_ok} ok, {n_fail} fail, "
              f"elapsed={time.time() - start:.1f}s")
        return result


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) >= 2:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            data = json.load(f)
        small = dict(list(data.items())[:4])
        m = Seed20ProInferencer(model_size="2_0_pro")
        res = m.inference(small)
        print(json.dumps(res, ensure_ascii=False, indent=2))
