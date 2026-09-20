from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from . import config

@dataclass(frozen=True)
class Completion:
    content: str
    prompt_tokens: int | None = None

def _pace() -> None:
    if config.LLM_PACE_SECONDS > 0:
        time.sleep(config.LLM_PACE_SECONDS)

def complete(
    messages: list[dict[str, Any]], response_format: dict[str, Any] | None = None
) -> Completion:
    import litellm

    _pace()

    kwargs: dict[str, Any] = {
        "model": config.LITELLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "api_key": config.GEMINI_API_KEY,
        "num_retries": 5,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    response = litellm.completion(**kwargs)
    usage = getattr(response, "usage", None)
    return Completion(
        content=response.choices[0].message.content or "",
        prompt_tokens=getattr(usage, "prompt_tokens", None),
    )

def embed(texts: list[str]) -> list[list[float]]:
    import litellm

    _pace()
    response = litellm.embedding(
        model=config.EMBEDDING_MODEL,
        input=list(texts),
        api_key=config.GEMINI_API_KEY,
    )
    return [list(item["embedding"]) for item in response.data]
