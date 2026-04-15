from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional


def _normalize_model_name(model: str) -> str:
    """
    Accept either a LiteLLM-style name (e.g. 'deepseek/deepseek-chat') or a raw DeepSeek model id
    (e.g. 'deepseek-chat') and return the DeepSeek model id.
    """
    model = (model or "").strip()
    if not model:
        return "deepseek-chat"
    if "/" in model:
        provider, name = model.split("/", 1)
        if provider.lower() == "deepseek":
            return name
    return model


@dataclass(frozen=True)
class DeepSeekClientConfig:
    model: str = "deepseek/deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: str = "sk-2f79bb184ea94824bb78a02da4973939"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: int = 60
    max_tokens: int = 512
    temperature: float = 0.2
    retries: int = 3
    retry_backoff_seconds: float = 2.0


def make_deepseek_lm(cfg: DeepSeekClientConfig) -> Callable[[str], str]:
    """
    Return a callable `lm(prompt: str) -> str` for GEPA that does NOT use LiteLLM.

    This uses DeepSeek's OpenAI-compatible chat completions endpoint:
      POST {base_url}/v1/chat/completions
    """
    model_id = _normalize_model_name(cfg.model)
    endpoint = cfg.base_url.rstrip("/") + "/v1/chat/completions"

    api_key = (cfg.api_key or "").strip()
    if not api_key and cfg.api_key_env:
        api_key = os.environ.get(cfg.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"{cfg.api_key_env} is not set. Set it in your job environment to use DeepSeek reflection."
        )

    def lm(prompt: str) -> str:
        prompt = str(prompt)
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(cfg.temperature),
            "max_tokens": int(cfg.max_tokens),
        }
        data = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        last_err: Optional[BaseException] = None
        for attempt in range(cfg.retries + 1):
            try:
                req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(body)
                return parsed["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                if attempt >= cfg.retries:
                    break
                time.sleep(cfg.retry_backoff_seconds * (attempt + 1))

        raise RuntimeError(f"DeepSeek call failed after {cfg.retries+1} attempts: {last_err}") from last_err

    return lm
