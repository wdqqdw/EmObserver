# evaluate/infer_engines/qwen36_vllm.py
"""
Qwen3.6-27B vLLM 推理引擎 —— 严格模式 (出错即抛, 不吞异常)。

- 模型路径: ${MVEI_MODEL_ROOT}/Qwen3.6-27B
- 与框架约定一致:
    * __init__(model_size=..., **kwargs)
    * inference(input: dict[mkey, rec], gpu_n=None, sampling_params=...)
        -> dict[mkey, {..., "generations": [str, ...]}]
- sampling_params 接受 vllm.SamplingParams / SimpleNamespace / dict / None,
  内部强制转换为 vllm.SamplingParams (满足 vLLM v1 input_processor 校验)。
- 任何一条样本构造失败 / 输出为空 / 数量不匹配 -> 立刻 RuntimeError 中断。
"""

from __future__ import annotations

import os
import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image
from evaluate.infer_engines.paths import model_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 模型路径与默认配置
# ---------------------------------------------------------------------------
MODEL_PATHS: Dict[str, str] = {
    "27b": model_path("Qwen3.6-27B"),
}
SUPPORTED_SIZES = list(MODEL_PATHS.keys())
DEFAULT_SIZE = "27b"


# ---------------------------------------------------------------------------
# 兼容性 hot-patch: 关闭 transformers RotaryEmbeddingConfigMixin 的严格 key 校验
# ---------------------------------------------------------------------------
def _disable_rope_validation() -> None:
    try:
        from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin  # type: ignore
    except Exception:
        return

    if getattr(RotaryEmbeddingConfigMixin, "_qwen36_patched", False):
        return

    def _noop(self, *args, **kwargs):
        return None

    RotaryEmbeddingConfigMixin._check_received_keys = _noop  # type: ignore[attr-defined]
    RotaryEmbeddingConfigMixin._qwen36_patched = True  # type: ignore[attr-defined]
    logger.info("[qwen36_vllm] Patched RotaryEmbeddingConfigMixin._check_received_keys -> no-op")


# ---------------------------------------------------------------------------
# 图像 / 文本字段提取 —— 与其它 engine 通用约定对齐
# ---------------------------------------------------------------------------
_PROMPT_KEYS = ("question", "prompt", "text", "instruction", "query")
_IMAGE_KEYS = (
    "image_paths", "images", "image", "image_path",
    "img_paths", "imgs", "img",
)


def _extract_prompt(rec: Dict[str, Any]) -> str:
    for k in _PROMPT_KEYS:
        if k in rec and rec[k] is not None:
            v = rec[k]
            if isinstance(v, str) and v.strip():
                return v
    raise KeyError(f"无法从样本中找到 prompt 字段 (尝试: {_PROMPT_KEYS}); 样本 keys={list(rec.keys())}")


def _extract_images(rec: Dict[str, Any]) -> List[Any]:
    for k in _IMAGE_KEYS:
        if k in rec and rec[k] is not None:
            v = rec[k]
            if isinstance(v, (list, tuple)):
                return list(v)
            return [v]
    return []


def _to_pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        if os.path.isfile(img):
            return Image.open(img).convert("RGB")
        raw = base64.b64decode(img)
        return Image.open(BytesIO(raw)).convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        return Image.open(BytesIO(img)).convert("RGB")
    raise TypeError(f"不支持的图像类型: {type(img)}")


# ---------------------------------------------------------------------------
# SamplingParams 强制转换
# ---------------------------------------------------------------------------
def _coerce_sampling_params(sampling_params: Any):
    from vllm import SamplingParams as _VSP

    if isinstance(sampling_params, _VSP):
        return sampling_params

    if sampling_params is None:
        return _VSP(n=1, temperature=0.7, max_tokens=1024)

    sp = sampling_params
    _get = (sp.get if isinstance(sp, dict) else lambda k, d=None: getattr(sp, k, d))

    def _to_int(v, d):
        try:
            return int(v) if v is not None else d
        except Exception:
            return d

    def _to_float(v, d):
        try:
            return float(v) if v is not None else d
        except Exception:
            return d

    n = _to_int(_get("n", 1), 1)
    temperature = _to_float(_get("temperature", 0.7), 0.7)
    top_p = _to_float(_get("top_p", 1.0), 1.0)
    top_k = _to_int(_get("top_k", -1), -1)
    repetition_penalty = _to_float(_get("repetition_penalty", 1.0), 1.0)
    max_tokens_raw = _get("max_tokens", None)
    if max_tokens_raw is None:
        max_tokens_raw = _get("max_new_tokens", 1024)
    max_tokens = _to_int(max_tokens_raw, 1024)

    stop = _get("stop", None)
    seed = _get("seed", None)

    kwargs = dict(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )
    if stop is not None:
        kwargs["stop"] = stop
    if seed is not None:
        try:
            kwargs["seed"] = int(seed)
        except Exception:
            pass

    return _VSP(**kwargs)


# ---------------------------------------------------------------------------
# 主推理器
# ---------------------------------------------------------------------------
class Qwen36Inferencer:
    """Qwen3.6-27B vLLM 推理封装 (严格模式)。"""

    def __init__(
        self,
        model_size: Optional[str] = None,
        size: Optional[str] = None,  # 向后兼容
        **kwargs,
    ):
        chosen = model_size or size or DEFAULT_SIZE
        if chosen not in MODEL_PATHS:
            raise ValueError(
                f"未知 model_size={chosen}, 可选: {list(MODEL_PATHS.keys())}"
            )
        self.size = chosen
        self.model_size = chosen
        self.model_path = MODEL_PATHS[chosen]
        self._llm = None
        self._processor = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # 懒加载 LLM
    # ------------------------------------------------------------------
    def _ensure_llm(self, gpu_n: Optional[int] = None):
        if self._llm is not None:
            return

        _disable_rope_validation()

        from vllm import LLM
        from transformers import AutoProcessor, AutoTokenizer

        tp = int(gpu_n or int(os.environ.get("QWEN36_TP", "8")))
        logger.info(f"[qwen36_vllm] Loading model={self.model_path} tp={tp}")

        common_kwargs = dict(
            model=self.model_path,
            tensor_parallel_size=tp,
            trust_remote_code=True,
            dtype="bfloat16",
            enable_prefix_caching=True,
            limit_mm_per_prompt={"image": 8, "video": 1},
            max_model_len=int(os.environ.get("QWEN36_MAX_MODEL_LEN", "32768")),
            gpu_memory_utilization=float(os.environ.get("QWEN36_GPU_UTIL", "0.90")),
        )

        extra_kwargs = dict(
            reasoning_parser="qwen3",
            mm_encoder_tp_mode="data",
            mm_processor_cache_type="shm",
        )

        try:
            self._llm = LLM(**common_kwargs, **extra_kwargs)
        except TypeError as e:
            logger.warning(f"[qwen36_vllm] LLM 不支持部分新参数 ({e}), 回退到基础参数")
            self._llm = LLM(**common_kwargs)

        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    # ------------------------------------------------------------------
    # 构造单条 vLLM 输入 —— 失败直接抛
    # ------------------------------------------------------------------
    def _build_one(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text = _extract_prompt(rec)
        image_items = _extract_images(rec)
        pil_images = [_to_pil(im) for im in image_items]

        content: List[Dict[str, Any]] = []
        for _ in pil_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        vllm_input: Dict[str, Any] = {"prompt": prompt}
        if pil_images:
            vllm_input["multi_modal_data"] = {"image": pil_images}
        return vllm_input

    # ------------------------------------------------------------------
    # 主入口 —— 严格模式
    # ------------------------------------------------------------------
    def inference(
        self,
        input: Dict[str, Dict[str, Any]],
        gpu_n: Optional[int] = None,
        sampling_params: Any = None,
    ) -> Dict[str, Dict[str, Any]]:

        # 1) sampling_params 强制转换
        sampling_params = _coerce_sampling_params(sampling_params)

        # 2) 懒加载
        self._ensure_llm(gpu_n=gpu_n)

        # 3) 归一化: 框架契约是 dict[mkey, rec]
        if not isinstance(input, dict):
            raise TypeError(
                f"[qwen36_vllm] inference() 期望 dict[mkey, rec], "
                f"实际收到 {type(input)}"
            )
        if not input:
            return {}

        keys: List[str] = list(input.keys())
        recs: List[Dict[str, Any]] = [input[k] for k in keys]

        # 4) 构造 vLLM 输入 —— 任一失败立刻抛, 错误带 key
        vllm_inputs: List[Dict[str, Any]] = []
        for k, rec in zip(keys, recs):
            try:
                vllm_inputs.append(self._build_one(rec))
            except Exception as e:
                raise RuntimeError(
                    f"[qwen36_vllm] 构造 prompt 失败 key={k}: {type(e).__name__}: {e}"
                ) from e

        # 5) 调用 vLLM
        logger.info(f"[qwen36_vllm] start generate, num_samples={len(vllm_inputs)}")
        outputs = self._llm.generate(vllm_inputs, sampling_params=sampling_params)

        # 6) 严格校验 + 整理结果
        if len(outputs) != len(keys):
            raise RuntimeError(
                f"[qwen36_vllm] vLLM 返回数量不匹配: in={len(keys)} out={len(outputs)}"
            )

        results: Dict[str, Dict[str, Any]] = {}
        for k, src_rec, out in zip(keys, recs, outputs):
            if out is None or not getattr(out, "outputs", None):
                raise RuntimeError(f"[qwen36_vllm] 空输出: key={k}")
            gens = [o.text for o in out.outputs]
            if not gens or all((not g or not g.strip()) for g in gens):
                raise RuntimeError(f"[qwen36_vllm] 全部生成为空: key={k}")

            # 透传 rec 里除图像数据外的元字段, 追加 generations
            results[k] = {
                **{kk: vv for kk, vv in src_rec.items() if kk not in _IMAGE_KEYS},
                "generations": gens,
            }

        return results


# ---------------------------------------------------------------------------
# 顶层导出 (供 INFER_ENGINES 注册表使用)
# ---------------------------------------------------------------------------
_singleton: Dict[str, Qwen36Inferencer] = {}


def get_engine(model_size: str = DEFAULT_SIZE) -> Qwen36Inferencer:
    if model_size not in _singleton:
        _singleton[model_size] = Qwen36Inferencer(model_size=model_size)
    return _singleton[model_size]


def inference(input, gpu_n=None, sampling_params=None, model_size: str = DEFAULT_SIZE):
    return get_engine(model_size).inference(
        input, gpu_n=gpu_n, sampling_params=sampling_params
    )


__all__ = [
    "Qwen36Inferencer",
    "get_engine",
    "inference",
    "MODEL_PATHS",
    "SUPPORTED_SIZES",
]