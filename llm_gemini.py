import errno
import random
import time
from typing import List, Optional, Tuple

import httpx
from google import genai

from .config import (
    GEMINI_API_KEY,
    GEMINI_RETRY_MAX_ATTEMPTS,
    GEMINI_RETRY_BASE_SECONDS,
    GEMINI_RETRY_JITTER_SECONDS,
)
from .net_proxy import apply_proxy_env

apply_proxy_env()

client = genai.Client(api_key=GEMINI_API_KEY)


def is_retryable_llm_error(exc: Exception) -> bool:
    """网络抖动、代理断连、服务端临时不可用等：应重试（仍可能最终失败）。"""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectTimeout,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            ConnectionError,
            TimeoutError,
        ),
    ):
        return True

    if isinstance(exc, OSError):
        we = getattr(exc, "winerror", None)
        if we in (10054, 10053, 10060, 10061):
            return True
        en = getattr(exc, "errno", None)
        if en in (
            errno.ECONNRESET,
            errno.ECONNREFUSED,
            errno.ETIMEDOUT,
            errno.EPIPE,
            errno.EHOSTUNREACH,
            errno.ENETUNREACH,
        ):
            return True

    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, Exception) and cause is not exc:
        if is_retryable_llm_error(cause):
            return True

    s = str(exc).lower()
    retry_marks = [
        "503",
        "unavailable",
        "high demand",
        "resource_exhausted",
        "429",
        "deadline exceeded",
        "timed out",
        "timeout",
        "connection reset",
        "10054",
        "winerror",
        "connecterror",
        "connection aborted",
        "broken pipe",
        "remoteprotocolerror",
        "server disconnected without sending a response",
        # 中文 Windows / 代理常见提示
        "远程主机",
        "强迫关闭",
        "无法连接",
    ]
    return any(m in s for m in retry_marks)


def generate_with_retry_and_fallback(
    *,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    primary_model: str,
    fallback_models: Optional[List[str]] = None,
) -> Tuple[str, str]:
    candidates: List[str] = [primary_model] + [
        m for m in (fallback_models or []) if m and m != primary_model
    ]
    last_error: Optional[Exception] = None

    for model_name in candidates:
        attempts = max(1, int(GEMINI_RETRY_MAX_ATTEMPTS))
        for attempt in range(1, attempts + 1):
            try:
                print(
                    f"Sending request to Gemini... model={model_name}, attempt={attempt}/{attempts}"
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    },
                )
                text = getattr(response, "text", None)
                if not text:
                    raise RuntimeError("Gemini returned empty response text.")
                print(f"Received response from Gemini model={model_name}")
                return text, model_name
            except Exception as e:
                last_error = e
                retryable = is_retryable_llm_error(e)
                if (not retryable) or attempt >= attempts:
                    print(
                        f"Gemini call failed model={model_name}, attempt={attempt}/{attempts}, "
                        f"retryable={retryable}: {e}"
                    )
                    break
                backoff = float(GEMINI_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, float(GEMINI_RETRY_JITTER_SECONDS))
                sleep_s = backoff + jitter
                print(
                    f"Gemini transient error model={model_name}, retry in {sleep_s:.2f}s: {e}"
                )
                time.sleep(sleep_s)

        if model_name != candidates[-1]:
            print(
                f"Switching fallback model after {attempts} failed attempts: "
                f"{model_name} -> {candidates[candidates.index(model_name) + 1]}"
            )

    if last_error:
        raise last_error
    raise RuntimeError("Gemini request failed without explicit error.")

