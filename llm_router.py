import time
import random
import threading
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import (
    OPENAI_API_KEY,
    DEEPSEEK_API_KEY,
    OPENAI_BASE_URL,
    DEEPSEEK_BASE_URL,
    OPENAI_MODEL_PRIMARY,
    OPENAI_MODEL_SECONDARY,
    DEEPSEEK_MODEL_PRIMARY,
    GEMINI_ANSWER_MODEL,
    GEMINI_FALLBACK_MODELS,
    GEMINI_RETRY_MAX_ATTEMPTS,
    GEMINI_RETRY_BASE_SECONDS,
    GEMINI_RETRY_JITTER_SECONDS,
    OPENAI_MAX_CONCURRENCY,
    OPENAI_RETRY_MAX_ATTEMPTS,
    OPENAI_RETRY_BASE_SECONDS,
    OPENAI_RETRY_JITTER_SECONDS,
    OPENAI_RETRY_MAX_SLEEP_SECONDS,
)
from .llm_gemini import generate_with_retry_and_fallback, is_retryable_llm_error
from .net_proxy import apply_proxy_env, get_proxy_url

apply_proxy_env()


def _sleep_backoff(attempt: int) -> None:
    backoff = float(GEMINI_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, float(GEMINI_RETRY_JITTER_SECONDS))
    time.sleep(backoff + jitter)


def _http_retryable(exc: Exception) -> bool:
    return is_retryable_llm_error(exc)


class OpenAIHTTPError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        retry_after_seconds: Optional[float],
        response_text: str,
    ):
        super().__init__(message)
        self.status_code = int(status_code)
        self.retry_after_seconds = retry_after_seconds
        self.response_text = response_text


_openai_sem = threading.Semaphore(max(1, int(OPENAI_MAX_CONCURRENCY)))


def _parse_retry_after_seconds(headers: httpx.Headers) -> Optional[float]:
    ra = (headers.get("retry-after") or "").strip()
    if not ra:
        return None
    # Retry-After can be seconds or HTTP-date.
    try:
        return float(ra)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(ra)
        now = dt.now(tz=dt.tzinfo)
        return max(0.0, (dt - now).total_seconds())
    except Exception:
        return None


def _sleep_openai_backoff(attempt: int, retry_after_seconds: Optional[float]) -> None:
    base = float(OPENAI_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, float(OPENAI_RETRY_JITTER_SECONDS))
    sleep_s = base + jitter
    if retry_after_seconds is not None:
        # Respect server guidance if it's larger than our local backoff.
        sleep_s = max(sleep_s, float(retry_after_seconds))
    sleep_s = min(float(OPENAI_RETRY_MAX_SLEEP_SECONDS), max(0.0, sleep_s))
    time.sleep(sleep_s)


def _openai_responses_once(prompt: str, model: str, *, max_output_tokens: int) -> Tuple[str, str]:
    key = (OPENAI_API_KEY or "").strip()
    if not key:
        raise RuntimeError("缺少 OPENAI_API_KEY（在 src/config.py 中配置），无法调用 OpenAI。")

    url = f"{OPENAI_BASE_URL}/responses"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
    }
    with _openai_sem:
        proxy = get_proxy_url()
        try:
            client = httpx.Client(timeout=180.0, proxy=proxy) if proxy else httpx.Client(timeout=180.0)
        except TypeError:
            # Older httpx versions use `proxies=`
            client = (
                httpx.Client(timeout=180.0, proxies=proxy) if proxy else httpx.Client(timeout=180.0)
            )
        with client:
            r = client.post(url, headers=headers, json=payload)

    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        status = int(getattr(r, "status_code", 0) or 0)
        text = ""
        try:
            text = r.text or ""
        except Exception:
            text = ""
        retry_after = _parse_retry_after_seconds(r.headers)
        raise OpenAIHTTPError(
            f"OpenAI HTTP {status} for {url}. body={text[:2000]}",
            status_code=status,
            retry_after_seconds=retry_after,
            response_text=text,
        ) from e

    data = r.json()
    # Responses API: best-effort extract
    text = ""
    if isinstance(data, dict):
        text = data.get("output_text") or ""
        if not text and "output" in data and isinstance(data["output"], list):
            # fallback: scan segments
            for item in data["output"]:
                if not isinstance(item, dict):
                    continue
                for c in item.get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        text += c.get("text", "") or ""
    if not text:
        raise RuntimeError("OpenAI 返回为空（未提取到 output_text）。")
    return text, f"openai:{model}"


def generate_answer_via_openai_responses(prompt: str, model: str, *, max_output_tokens: int) -> Tuple[str, str]:
    """OpenAI 单模型：重试（不含降级）。"""
    attempts = max(1, int(OPENAI_RETRY_MAX_ATTEMPTS or GEMINI_RETRY_MAX_ATTEMPTS))
    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"[openai] request model={model}, attempt={attempt}/{attempts}")
            return _openai_responses_once(prompt, model, max_output_tokens=max_output_tokens)
        except Exception as e:
            last_err = e
            if isinstance(e, OpenAIHTTPError):
                body_preview = (e.response_text or "").strip().replace("\r", " ").replace("\n", " ")
                if len(body_preview) > 360:
                    body_preview = body_preview[:360] + "…"
                print(
                    f"[openai] error model={model}, status={e.status_code}, "
                    f"retry_after={e.retry_after_seconds}, attempt={attempt}/{attempts}, body={body_preview}"
                )
            if attempt >= attempts or not _http_retryable(e):
                break
            if isinstance(e, OpenAIHTTPError):
                _sleep_openai_backoff(attempt, e.retry_after_seconds)
            else:
                _sleep_backoff(attempt)
    raise last_err or RuntimeError("OpenAI 调用失败。")


def generate_answer_via_openai_with_fallback(
    prompt: str,
    primary_model: str,
    fallback_model: Optional[str],
    *,
    max_output_tokens: int,
) -> Tuple[str, str]:
    """
    OpenAI：primary 失败（可重试错误且重试耗尽）后，自动切换到 fallback。
    """
    last_err: Optional[Exception] = None
    candidates = [primary_model] + ([fallback_model] if fallback_model else [])
    for model_name in candidates:
        attempts = max(1, int(OPENAI_RETRY_MAX_ATTEMPTS or GEMINI_RETRY_MAX_ATTEMPTS))
        for attempt in range(1, attempts + 1):
            try:
                print(f"[openai] request model={model_name}, attempt={attempt}/{attempts}")
                return _openai_responses_once(prompt, model_name, max_output_tokens=max_output_tokens)
            except Exception as e:
                last_err = e
                if isinstance(e, OpenAIHTTPError):
                    body_preview = (e.response_text or "").strip().replace("\r", " ").replace("\n", " ")
                    if len(body_preview) > 360:
                        body_preview = body_preview[:360] + "…"
                    print(
                        f"[openai] error model={model_name}, status={e.status_code}, "
                        f"retry_after={e.retry_after_seconds}, attempt={attempt}/{attempts}, body={body_preview}"
                    )
                if attempt >= attempts or not _http_retryable(e):
                    break
                if isinstance(e, OpenAIHTTPError):
                    _sleep_openai_backoff(attempt, e.retry_after_seconds)
                else:
                    _sleep_backoff(attempt)
        # 当前模型失败且还有下一个候选，就切换下一模型继续
        if model_name != candidates[-1]:
            print(f"[openai] switching fallback model: {model_name} -> {candidates[candidates.index(model_name)+1]}")
    raise last_err or RuntimeError("OpenAI 调用失败。")

def generate_answer_via_deepseek_chat(prompt: str, model: str) -> Tuple[str, str]:
    key = (DEEPSEEK_API_KEY or "").strip()
    if not key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY（在 src/config.py 中配置），无法调用 DeepSeek。")

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    model = (model or "").strip()
    if model not in ("deepseek-chat", "deepseek-reasoner"):
        raise RuntimeError(
            f"DeepSeek 模型名不合法：{model!r}。可用：deepseek-chat / deepseek-reasoner"
        )
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    attempts = max(1, int(GEMINI_RETRY_MAX_ATTEMPTS))
    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            proxy = get_proxy_url()
            try:
                client = httpx.Client(timeout=180.0, proxy=proxy) if proxy else httpx.Client(timeout=180.0)
            except TypeError:
                client = (
                    httpx.Client(timeout=180.0, proxies=proxy)
                    if proxy
                    else httpx.Client(timeout=180.0)
                )
            with client:
                r = client.post(url, headers=headers, json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = ""
                try:
                    body = r.text or ""
                except Exception:
                    body = ""
                raise RuntimeError(
                    f"DeepSeek HTTP {r.status_code} for {url}. body={body[:2000]}"
                ) from e
            data = r.json()
            text = ""
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices and isinstance(choices, list) and isinstance(choices[0], dict):
                    msg = choices[0].get("message") or {}
                    if isinstance(msg, dict):
                        text = msg.get("content") or ""
            if not text:
                raise RuntimeError("DeepSeek 返回为空（未提取到 message.content）。")
            return text, f"deepseek:{model}"
        except Exception as e:
            last_err = e
            if attempt >= attempts or not _http_retryable(e):
                break
            _sleep_backoff(attempt)
    raise last_err or RuntimeError("DeepSeek 调用失败。")


def generate_answer(
    *,
    prompt: str,
    provider: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> Tuple[str, str]:
    """
    统一路由：返回 (text, provider:model_used)
    """
    p = (provider or "gemini").strip().lower()
    m_raw = (model or "").strip()

    if p == "openai":
        # OpenAI：若选择 primary（默认 o3），失败 3 次后自动降级到 secondary（默认 o1）
        m = m_raw or OPENAI_MODEL_PRIMARY
        if m == OPENAI_MODEL_PRIMARY and OPENAI_MODEL_SECONDARY:
            return generate_answer_via_openai_with_fallback(
                prompt,
                OPENAI_MODEL_PRIMARY,
                OPENAI_MODEL_SECONDARY,
                max_output_tokens=max_output_tokens,
            )
        return generate_answer_via_openai_responses(
            prompt, m, max_output_tokens=max_output_tokens
        )
    if p == "deepseek":
        m = m_raw or DEEPSEEK_MODEL_PRIMARY
        return generate_answer_via_deepseek_chat(prompt, m)

    # gemini
    if m_raw:
        primary = m_raw
    else:
        primary = GEMINI_ANSWER_MODEL

    # pro 默认允许 fallback；如果用户直接选 flash，则不再 fallback
    if primary == "gemini-2.5-flash":
        fallbacks: List[str] = []
    else:
        fallbacks = list(GEMINI_FALLBACK_MODELS)

    text, used = generate_with_retry_and_fallback(
        prompt=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        primary_model=primary,
        fallback_models=fallbacks,
    )
    return text, f"gemini:{used}"

