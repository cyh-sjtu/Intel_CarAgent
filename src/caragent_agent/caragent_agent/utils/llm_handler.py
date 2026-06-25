"""Unified LangChain-backed LLM client with shared request governance."""

from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import AsyncCallbackHandler

from typing import Dict, List, Union, Any, Optional
import asyncio
from contextlib import contextmanager
import logging
import os
import threading

from caragent_agent.agents.async_agent.runtime.resource_scheduler import (
    llm_background_yield_to_foreground_enabled,
)
from caragent_agent.config.config import config, ensure_api_key_env

class LLMCallbackHandler(AsyncCallbackHandler):
    """Monitor LLM calls and collect simple usage statistics."""
    
    def __init__(self):
        self.stats = {
            'total_calls': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'errors': 0
        }
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self.stats['total_calls'] += 1
        logging.info(f"LLM call started: {serialized.get('name', 'unknown')}")

    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[Any], **kwargs) -> None:
        self.stats['total_calls'] += 1
        logging.info(f"Chat model call started: {serialized.get('name', 'unknown')}")
    
    async def on_llm_end(self, response, **kwargs) -> None:
        # Track token usage when available
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            self.stats['total_tokens'] += token_usage.get('total_tokens', 0)
    
    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.stats['errors'] += 1
        logging.error(f"LLM call failed: {error}")

class UnifiedLLMClient:
    """Unified LangChain LLM client bridging multiple providers."""

    _limiters_lock = threading.Lock()
    _model_limiters: Dict[str, tuple[int, threading.Semaphore]] = {}
    _provider_limiters: Dict[str, tuple[int, threading.Semaphore]] = {}
    _priority_local = threading.local()
    _foreground_lock = threading.Lock()
    _foreground_pending = 0
    _foreground_active = 0
    _foreground_provider_demand: Dict[str, int] = {}
    
    def __init__(self):
        self.models: Dict[str, BaseLLM] = {}
        self.callback_handler = LLMCallbackHandler()
        self._init_models()
    
    def _init_models(self):
        """Initialize all supported model backends."""
        dashscope_api_key = ensure_api_key_env("qwen")
        deepseek_api_key = ensure_api_key_env("deepseek")
        doubao_api_key = ensure_api_key_env("doubao")
        raw_request_timeout = config.get("llm_request_timeout_sec", 25)
        if isinstance(raw_request_timeout, dict):
            raw_request_timeout = raw_request_timeout.get("default", 25)
        default_request_timeout = float(raw_request_timeout)
        max_retries = int(config.get("llm_max_retries", 1))

        # Only initialize providers that are actually configured so local
        # development can use a subset of backends without startup failures.
        qwen_models = [
            "qwen-vl-max",
            "qwen3-vl-plus",
            "qwen-turbo",
            "qwen-max",
            "qwen3-vl-plus",
            "qwen-plus",
            "qwen3.6-plus",
            "qwen3.6-flash",
            "qwen3-vl-flash",
        ]
        if dashscope_api_key:
            for model_name in qwen_models:
                self.models[model_name] = ChatOpenAI(
                    api_key=dashscope_api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    model=self._get_qwen_model_id(model_name),
                    temperature=0.0,
                    timeout=self._request_timeout_for_model(model_name, default_request_timeout),
                    max_retries=max_retries,
                    callbacks=[self.callback_handler]
                )

        if deepseek_api_key:
            deepseek_models = [
                "deepseek-chat",      # compatibility alias
                "deepseek-v4-flash",
                "deepseek-v4-pro",
            ]
            for model_name in deepseek_models:
                self.models[model_name] = ChatOpenAI(
                    api_key=deepseek_api_key,
                    base_url="https://api.deepseek.com/v1",
                    model=self._get_deepseek_model_id(model_name),
                    temperature=0.0,
                    timeout=self._request_timeout_for_model(model_name, default_request_timeout),
                    max_retries=max_retries,
                    callbacks=[self.callback_handler]
                )

        doubao_models = ["doubao-1-5-vision-lite", "doubao-1-5-vision-pro", "doubao-deepseek-v3"]
        if doubao_api_key:
            for model_name in doubao_models:
                self.models[model_name] = ChatOpenAI(
                    api_key=doubao_api_key,
                    base_url="https://ark.cn-beijing.volces.com/api/v3",
                    model=self._get_doubao_model_id(model_name),
                    temperature=0.0,
                    timeout=self._request_timeout_for_model(model_name, default_request_timeout),
                    max_retries=max_retries,
                    callbacks=[self.callback_handler]
                )
    
    def _get_qwen_model_id(self, model_name: str) -> str:
        """Resolve Qwen alias to actual model id."""
        mapping = {
            "qwen-vl-max": "qwen-vl-max-latest",
            "qwen3-vl-plus": "qwen3-vl-plus", 
            "qwen-turbo": "qwen-turbo",
            "qwen-max": "qwen-max",
            "qwen-plus": "qwen-plus",
            "qwen3.6-plus": "qwen3.6-plus",
            "qwen3.6-flash": "qwen3.6-flash",
            "qwen3-vl-flash": "qwen3-vl-flash",
        }
        return mapping.get(model_name, model_name)

    def _get_deepseek_model_id(self, model_name: str) -> str:
        """Resolve DeepSeek alias to the current public API model ids."""

        mapping = {
            "deepseek-chat": "deepseek-v4-flash",
            "deepseek-v4-flash": "deepseek-v4-flash",
            "deepseek-v4-pro": "deepseek-v4-pro",
        }
        return mapping.get(model_name, model_name)
    
    def _get_doubao_model_id(self, model_name: str) -> str:
        """Resolve Doubao alias to actual model id."""
        mapping = {
            "doubao-1-5-vision-lite": "doubao-1.5-vision-lite-250315",
            "doubao-1-5-vision-pro": "doubao-1-5-vision-pro-32k-250115",
            "doubao-deepseek-v3": "deepseek-v3-250324"
        }
        return mapping.get(model_name, model_name)

    @classmethod
    def reset_shared_limiters_for_tests(cls) -> None:
        """Clear process-wide limiters so tests can safely change config knobs."""

        with cls._limiters_lock:
            cls._model_limiters.clear()
            cls._provider_limiters.clear()
        with cls._foreground_lock:
            cls._foreground_pending = 0
            cls._foreground_active = 0
            cls._foreground_provider_demand.clear()
        cls._priority_local.priority = "foreground"

    @classmethod
    @contextmanager
    def request_priority(cls, priority: str):
        """Temporarily tag requests from this thread as foreground or background."""

        normalized = str(priority or "").strip().lower()
        if normalized not in {"foreground", "background"}:
            normalized = "foreground"
        previous = getattr(cls._priority_local, "priority", "foreground")
        cls._priority_local.priority = normalized
        try:
            yield
        finally:
            cls._priority_local.priority = previous

    @classmethod
    def _current_priority(cls) -> str:
        """Return the current thread's request priority."""

        priority = getattr(cls._priority_local, "priority", "foreground")
        return "background" if priority == "background" else "foreground"

    @classmethod
    def _increment_foreground_pending(cls, provider: Optional[str] = None) -> None:
        with cls._foreground_lock:
            cls._foreground_pending += 1
            if provider:
                cls._foreground_provider_demand[provider] = (
                    cls._foreground_provider_demand.get(provider, 0) + 1
                )

    @classmethod
    def _foreground_pending_to_active(cls) -> None:
        with cls._foreground_lock:
            cls._foreground_pending = max(0, cls._foreground_pending - 1)
            cls._foreground_active += 1

    @classmethod
    def _decrement_foreground_pending(cls, provider: Optional[str] = None) -> None:
        with cls._foreground_lock:
            cls._foreground_pending = max(0, cls._foreground_pending - 1)
            if provider:
                next_value = max(
                    0,
                    cls._foreground_provider_demand.get(provider, 0) - 1,
                )
                if next_value:
                    cls._foreground_provider_demand[provider] = next_value
                else:
                    cls._foreground_provider_demand.pop(provider, None)

    @classmethod
    def _decrement_foreground_active(cls, provider: Optional[str] = None) -> None:
        with cls._foreground_lock:
            cls._foreground_active = max(0, cls._foreground_active - 1)
            if provider:
                next_value = max(
                    0,
                    cls._foreground_provider_demand.get(provider, 0) - 1,
                )
                if next_value:
                    cls._foreground_provider_demand[provider] = next_value
                else:
                    cls._foreground_provider_demand.pop(provider, None)

    @classmethod
    def _foreground_has_demand(cls, provider: Optional[str] = None) -> bool:
        with cls._foreground_lock:
            if provider:
                return cls._foreground_provider_demand.get(provider, 0) > 0
            return (cls._foreground_pending + cls._foreground_active) > 0

    def _provider_for_model(self, model_name: str) -> str:
        """Return the API provider namespace used for shared provider throttling."""

        normalized = str(model_name or "").strip().lower()
        if normalized.startswith("qwen"):
            return "qwen"
        if normalized.startswith("deepseek"):
            return "deepseek"
        if normalized.startswith("doubao"):
            return "doubao"
        return "default"

    def _configured_int(
        self,
        key: str,
        lookup_key: str,
        default: int,
        *,
        secondary_lookup_key: Optional[str] = None,
    ) -> int:
        """Read either scalar or mapping-style integer config values."""

        raw_value = config.get(key, default)
        if isinstance(raw_value, dict):
            value = raw_value.get(lookup_key)
            if value is None and secondary_lookup_key:
                value = raw_value.get(secondary_lookup_key)
            if value is None:
                value = raw_value.get("default", default)
        else:
            value = raw_value

        try:
            return int(value)
        except Exception:
            return default

    def _configured_float(
        self,
        key: str,
        lookup_key: str,
        default: float,
        *,
        secondary_lookup_key: Optional[str] = None,
    ) -> float:
        """Read either scalar or mapping-style float config values."""

        raw_value = config.get(key, default)
        if isinstance(raw_value, dict):
            value = raw_value.get(lookup_key)
            if value is None and secondary_lookup_key:
                value = raw_value.get(secondary_lookup_key)
            if value is None:
                value = raw_value.get("default", default)
        else:
            value = raw_value

        try:
            return float(value)
        except Exception:
            return float(default)

    def _request_timeout_for_model(self, model_name: str, default: float) -> float:
        provider = self._provider_for_model(model_name)
        return max(
            1.0,
            self._configured_float(
                "llm_request_timeout_sec",
                model_name,
                default,
                secondary_lookup_key=provider,
            ),
        )

    @classmethod
    def _get_shared_limiter(
        cls,
        limiter_map: Dict[str, tuple[int, threading.Semaphore]],
        key: str,
        limit: int,
    ) -> Optional[threading.Semaphore]:
        """Return one process-wide semaphore for a provider/model limit."""

        if limit <= 0:
            return None

        with cls._limiters_lock:
            existing = limiter_map.get(key)
            if existing is None or existing[0] != limit:
                limiter_map[key] = (limit, threading.Semaphore(limit))
            return limiter_map[key][1]

    def _request_contains_image(self, messages: List[Dict]) -> bool:
        """Return True when a request contains image payloads and needs a VLM."""

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type") or "").strip().lower()
                    if item_type in {"image", "image_url"}:
                        return True
                    if "image" in item or "image_url" in item:
                        return True
            elif isinstance(content, dict):
                item_type = str(content.get("type") or "").strip().lower()
                if item_type in {"image", "image_url"}:
                    return True
                if "image" in content or "image_url" in content:
                    return True
        return False

    def _coerce_model_list(self, raw_value: Any) -> list[str]:
        """Normalize fallback model config into a clean ordered list."""

        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            return [
                item.strip()
                for item in raw_value.split(",")
                if item.strip()
            ]
        if isinstance(raw_value, (list, tuple)):
            return [
                str(item).strip()
                for item in raw_value
                if str(item).strip()
            ]
        return []

    def _fallback_models_for_request(
        self,
        model: str,
        messages: List[Dict],
    ) -> list[str]:
        """Return safe fallback models for this request."""

        if self._request_contains_image(messages):
            return []
        if config.get("llm_enable_text_fallback", True) is False:
            return []

        provider = self._provider_for_model(model)
        fallback_config = config.get("llm_fallback_models", {})
        candidates: list[str] = []

        if isinstance(fallback_config, dict):
            for key in (model, provider, "default"):
                candidates.extend(self._coerce_model_list(fallback_config.get(key)))
        else:
            candidates.extend(self._coerce_model_list(fallback_config))

        deduped: list[str] = []
        for candidate in candidates:
            if candidate == model or candidate in deduped:
                continue
            deduped.append(candidate)
        return deduped

    def _candidate_models_for_request(
        self,
        model: str,
        messages: List[Dict],
    ) -> list[str]:
        """Return primary model followed by safe text-only fallbacks."""

        return [model] + self._fallback_models_for_request(model, messages)

    def _background_text_model_for_request(
        self,
        model: str,
        messages: List[Dict],
    ) -> Optional[str]:
        """Return a configured background-only text model when it is safe to use."""

        if self._current_priority() != "background":
            return None
        if self._request_contains_image(messages):
            return None

        raw_model = str(config.get("llm_background_text_model") or "").strip()
        if not raw_model or raw_model == model:
            return None
        if raw_model not in self.models:
            logging.warning(
                "Configured llm_background_text_model %s is unavailable; "
                "using requested model %s.",
                raw_model,
                model,
            )
            return None
        return raw_model

    def _effective_primary_model_for_request(
        self,
        model: str,
        messages: List[Dict],
    ) -> str:
        """Choose the actual primary model for this request's priority lane."""

        background_model = self._background_text_model_for_request(model, messages)
        return background_model or model

    def _is_transient_error(self, error: Exception) -> bool:
        """Return True for rate-limit/network errors worth retrying or falling back."""

        text = str(error or "").lower()
        transient_markers = (
            "429",
            "rate limit",
            "rate_limit",
            "limit_burst_rate",
            "too many requests",
            "timeout",
            "timed out",
            "connection error",
            "connection reset",
            "temporarily unavailable",
            "service unavailable",
            "gateway timeout",
            "bad gateway",
        )
        return any(marker in text for marker in transient_markers)

    def _retry_delay(self, attempt_index: int) -> float:
        """Compute conservative exponential backoff delay for transient failures."""

        try:
            base = float(config.get("llm_retry_backoff_base_sec", 0.75))
        except Exception:
            base = 0.75
        try:
            maximum = float(config.get("llm_retry_backoff_max_sec", 6.0))
        except Exception:
            maximum = 6.0
        return max(0.0, min(maximum, base * (2 ** max(0, attempt_index))))

    def _background_poll_interval(self) -> float:
        """Return how often background calls should retry acquiring capacity."""

        try:
            return max(0.05, float(config.get("llm_background_poll_interval_sec", 0.35)))
        except Exception:
            return 0.35

    async def _wait_for_foreground_quiet(
        self,
        provider: Optional[str] = None,
    ) -> None:
        """Let foreground requests drain before background requests consume slots."""

        if not llm_background_yield_to_foreground_enabled(config):
            return

        poll_interval = self._background_poll_interval()
        while self._foreground_has_demand(provider):
            await asyncio.sleep(poll_interval)

    async def _acquire_thread_limiter(
        self,
        limiter: Optional[threading.Semaphore],
    ) -> Optional[threading.Semaphore]:
        """Acquire a threading semaphore without blocking the event loop."""

        if limiter is None:
            return None
        await asyncio.to_thread(limiter.acquire)
        return limiter

    async def _acquire_limiters(
        self,
        limiters: List[Optional[threading.Semaphore]],
        *,
        priority: str,
        provider: Optional[str] = None,
    ) -> list[threading.Semaphore]:
        """Acquire all limiters, with background calls yielding to foreground."""

        concrete_limiters = [limiter for limiter in limiters if limiter is not None]
        if (
            priority != "background"
            or not llm_background_yield_to_foreground_enabled(config)
        ):
            acquired: list[threading.Semaphore] = []
            for limiter in concrete_limiters:
                acquired_limiter = await self._acquire_thread_limiter(limiter)
                if acquired_limiter is not None:
                    acquired.append(acquired_limiter)
            return acquired

        poll_interval = self._background_poll_interval()
        while True:
            await self._wait_for_foreground_quiet(provider)
            acquired = []
            for limiter in concrete_limiters:
                if limiter.acquire(blocking=False):
                    acquired.append(limiter)
                    continue
                for acquired_limiter in reversed(acquired):
                    acquired_limiter.release()
                acquired = []
                break
            if len(acquired) == len(concrete_limiters):
                return acquired
            await asyncio.sleep(poll_interval)

    async def _chat_completion_once(
        self,
        model: str,
        langchain_messages: List[Any],
        **kwargs,
    ) -> str:
        """Execute one throttled provider call without retry/fallback policy."""

        provider = self._provider_for_model(model)
        model_limit = self._configured_int(
            "llm_max_concurrency_per_model",
            model,
            2,
            secondary_lookup_key=provider,
        )
        provider_limit = self._configured_int(
            "llm_max_concurrency_per_provider",
            provider,
            2,
        )
        model_limiter = self._get_shared_limiter(
            self._model_limiters,
            model,
            model_limit,
        )
        provider_limiter = self._get_shared_limiter(
            self._provider_limiters,
            provider,
            provider_limit,
        )

        priority = self._current_priority()
        is_foreground = priority != "background"
        foreground_active = False
        if is_foreground:
            self._increment_foreground_pending(provider)

        acquired: list[threading.Semaphore] = []
        try:
            acquired = await self._acquire_limiters(
                [provider_limiter, model_limiter],
                priority=priority,
                provider=provider,
            )
            if is_foreground:
                self._foreground_pending_to_active()
                foreground_active = True

            llm = self.models[model]
            response = await llm.agenerate([langchain_messages], **kwargs)
            return response.generations[0][0].text
        finally:
            if is_foreground:
                if foreground_active:
                    self._decrement_foreground_active(provider)
                else:
                    self._decrement_foreground_pending(provider)
            for limiter in reversed(acquired):
                limiter.release()

    async def _chat_completion_with_retries(
        self,
        model: str,
        langchain_messages: List[Any],
        **kwargs,
    ) -> str:
        """Execute one model call with client-level transient retry/backoff."""

        max_attempts = self._configured_int(
            "llm_client_max_attempts",
            model,
            2,
            secondary_lookup_key=self._provider_for_model(model),
        )
        max_attempts = max(1, max_attempts)
        last_error: Optional[Exception] = None

        for attempt_index in range(max_attempts):
            try:
                return await self._chat_completion_once(
                    model,
                    langchain_messages,
                    **kwargs,
                )
            except Exception as exc:
                last_error = exc
                if (
                    attempt_index >= max_attempts - 1
                    or not self._is_transient_error(exc)
                ):
                    logging.error(f"Chat completion failed for {model}: {exc}")
                    raise

                delay = self._retry_delay(attempt_index)
                logging.warning(
                    "Transient chat completion failure for %s "
                    "(attempt %s/%s): %s. Retrying in %.2fs.",
                    model,
                    attempt_index + 1,
                    max_attempts,
                    exc,
                    delay,
                )
                if delay > 0:
                    await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Chat completion failed for {model} without an exception.")
    
    async def chat_completion(self, model: str, messages: List[Dict], **kwargs) -> str:
        """Execute a single chat completion."""
        langchain_messages = self._convert_messages(messages)
        effective_model = self._effective_primary_model_for_request(model, messages)
        candidate_models = self._candidate_models_for_request(effective_model, messages)
        configured_models = ", ".join(sorted(self.models)) or "none"
        last_error: Optional[Exception] = None

        for index, candidate_model in enumerate(candidate_models):
            if candidate_model not in self.models:
                last_error = ValueError(
                    f"Model '{candidate_model}' is unavailable because its provider "
                    f"is not configured. Configured models: {configured_models}."
                )
                if index >= len(candidate_models) - 1:
                    raise last_error
                logging.warning(
                    "Skipping unavailable fallback candidate %s for primary model %s.",
                    candidate_model,
                    effective_model,
                )
                continue

            try:
                return await self._chat_completion_with_retries(
                    candidate_model,
                    langchain_messages,
                    **kwargs,
                )
            except Exception as exc:
                last_error = exc
                if (
                    index >= len(candidate_models) - 1
                    or not self._is_transient_error(exc)
                ):
                    raise
                logging.warning(
                    "Falling back from %s to %s after transient failure: %s",
                    candidate_model,
                    candidate_models[index + 1],
                    exc,
                )

        if last_error is not None:
            raise last_error
        raise ValueError(
            f"Model '{effective_model}' is unavailable because its provider is not configured. "
            f"Configured models: {configured_models}."
        )
    
    async def batch_chat_completion(self, requests: List[Dict]) -> Dict[str, Any]:
        """Run multiple chat completions concurrently."""
        priority = self._current_priority()
        foreground_guard_providers: list[str] = []
        if priority != "background":
            for req in requests:
                effective_model = self._effective_primary_model_for_request(
                    req["model"],
                    req["messages"],
                )
                provider = self._provider_for_model(effective_model)
                foreground_guard_providers.append(provider)
                self._increment_foreground_pending(provider)

        try:
            tasks = []
            for req in requests:
                task = self._process_single_request(
                    req["request_id"],
                    req["model"],
                    req["messages"],
                    req.get("kwargs", {})
                )
                tasks.append(task)

        # Execute all requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results keyed by request id
            final_results = {}
            for i, req in enumerate(requests):
                req_id = req["request_id"]
                result = results[i]

                if isinstance(result, Exception):
                    final_results[req_id] = {req_id: {"error": str(result)}}
                else:
                    final_results[req_id] = result

            return final_results
        finally:
            for provider in foreground_guard_providers:
                self._decrement_foreground_pending(provider)
    
    async def _process_single_request(self, req_id: str, model: str, messages: List[Dict], kwargs: Dict) -> Dict:
        """Handle a single request with error capture."""
        try:
            response = await self.chat_completion(model, messages, **kwargs)
            return {req_id: response}
        except Exception as e:
            return {req_id: {"error": str(e)}}
    
    def _convert_messages(self, messages: List[Dict]) -> List:
        """Convert request messages to LangChain-compatible objects."""
        langchain_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Handle multimodal content to match LangChain/Qwen expectations
            if isinstance(content, list):
                converted_content = []
                for item in content:
                    item_type = item.get("type")
                    if item_type == "text":
                        converted_content.append({"type": "text", "text": item["text"]})
                    elif item_type == "image_url":
                        # Keep OpenAI-compatible multimodal shape for provider
                        # adapters that sit behind ChatOpenAI-compatible APIs.
                        converted_content.append({
                            "type": "image_url",
                            "image_url": {"url": item["image_url"]["url"]},
                        })
                    elif item_type == "image":
                        # Backward-compatible normalization for any legacy
                        # callers that still pass {"type": "image", "image": ...}.
                        converted_content.append({
                            "type": "image_url",
                            "image_url": {"url": item["image"]},
                        })
                if converted_content and all(
                    item.get("type") == "text" for item in converted_content
                ):
                    content = "\n".join(
                        str(item.get("text") or "") for item in converted_content
                    )
                else:
                    content = converted_content
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
        
        return langchain_messages
    
    def get_stats(self) -> Dict:
        """Return collected usage statistics."""
        return self.callback_handler.stats

# Usage example
async def demo_unified_client():
    """Demonstrate basic UnifiedLLMClient usage."""
    client = UnifiedLLMClient()
    
    # 单个请求
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "请介绍一下Python"}
    ]
    
    response = await client.chat_completion("qwen-plus", messages)
    print(f"单个请求响应: {response[:100]}...")

    batch_requests = [
        {
            "request_id": 0,
            "model": "qwen-plus",
            "messages": [{"role": "user", "content": "什么是AI?"}]
        },
        {
            "request_id": 1,
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "什么是机器学习?"}]
        }
    ]
    
    batch_results = await client.batch_chat_completion(batch_requests)
    print(f"批量请求结果: {len(batch_results)} 个响应")

    stats = client.get_stats()
    print(f"调用统计: {stats}")

if __name__ == "__main__":
    asyncio.run(demo_unified_client())
