from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable

import torch


DEFAULT_LOCAL_QWEN_PATH = os.environ.get(
    "GEPA_LOCAL_QWEN_PATH",
    "/root/dataDisk/dsee_temp_storage/Qwen/Qwen3-32B",
)

_MODEL = None
_TOKENIZER = None


@dataclass(frozen=True)
class LocalHFConfig:
    model_path: str = DEFAULT_LOCAL_QWEN_PATH
    device: str | None = None
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    trust_remote_code: bool = True


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE).strip()


def _normalize_model_path(model: str) -> str:
    model = (model or "").strip()
    if model.startswith("local:"):
        model = model.split(":", 1)[1]
    return model or DEFAULT_LOCAL_QWEN_PATH


def _load_model(cfg: LocalHFConfig):
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = _normalize_model_path(cfg.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=cfg.trust_remote_code)

    device = cfg.device or os.environ.get("GEPA_LOCAL_LM_DEVICE", "auto")
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=cfg.trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
            trust_remote_code=cfg.trust_remote_code,
        ).to(device)

    _MODEL = model.eval()
    _TOKENIZER = tokenizer
    return _MODEL, _TOKENIZER


def make_local_hf_lm(cfg: LocalHFConfig) -> Callable[[str], str]:
    """
    Return a local Hugging Face causal-LM callable for GEPA reflection.
    """

    def lm(prompt: str) -> str:
        model, tokenizer = _load_model(cfg)
        messages = [{"role": "user", "content": str(prompt)}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = str(prompt)

        inputs = tokenizer([text], return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        do_sample = cfg.temperature > 0
        generate_kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = cfg.temperature
            generate_kwargs["top_p"] = cfg.top_p

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return _strip_thinking(decoded)

    return lm
