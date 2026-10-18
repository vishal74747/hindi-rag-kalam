"""One place for the Groq call: used by ask.py (answers) and retrieval.py (query translation).

Reads GROQ_API_KEY from the environment; never hard-code keys. Rate-limit and connection errors
are retried with exponential backoff, because the free tier rate-limits often.
"""
from __future__ import annotations

import os
import re
import time

DEFAULT_MODEL = "openai/gpt-oss-120b"


def have_key() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def chat(system: str, user: str, max_tokens: int = 1500, retries: int = 3) -> str:
    import groq  # lazy: only needed when an LLM is actually used

    client = groq.Groq()
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    retryable = tuple(getattr(groq, n) for n in ("RateLimitError", "APIConnectionError", "InternalServerError")
                      if hasattr(groq, n))  # rate limit, network, and 5xx "over capacity" errors
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,  # reasoning models spend tokens thinking before the answer
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            break
        except retryable:
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s
    text = resp.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:  # a reasoning model can spend all max_tokens thinking; do not pass "" on as an answer
        raise RuntimeError(f"empty reply from {model} (finish_reason={resp.choices[0].finish_reason}); "
                           "raise max_tokens or retry")
    return text