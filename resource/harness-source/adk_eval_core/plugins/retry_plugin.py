"""ADK plugin providing automatic exponential backoff retries for transient LLM errors."""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import re
from typing import TYPE_CHECKING

from google.adk.plugins.base_plugin import BasePlugin

from adk_eval_core.errors import EvalCoreError

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse

logger = logging.getLogger(__name__)

_STATUS_CODE_PATTERN = re.compile(r"\b(429|500|502|503|504)\b")


class ModelRetryPlugin(BasePlugin):
    """ADK plugin providing exponential backoff retries for model calls.

    Handles rate limits (429), server errors (500, 502, 503, 504), ResourceExhausted,
    and network connection timeouts.
    """

    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float | None = None,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        name: str = "model_retry",
        *,
        initial_backoff: float | None = None,
    ) -> None:
        super().__init__(name=name)
        self.max_retries = max_retries
        self.initial_delay = (
            initial_delay
            if initial_delay is not None
            else (initial_backoff if initial_backoff is not None else 3.0)
        )
        self.initial_backoff = self.initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

    def _is_retryable_error(self, error: Exception) -> bool:
        if isinstance(
            error,
            (
                EvalCoreError,
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                TypeError,
            ),
        ):
            return False

        err_str = str(error)
        err_type = type(error).__name__

        # 1. ADK ResourceExhaustedError or rate limit / quota errors
        if "ResourceExhausted" in err_type or "ResourceExhausted" in err_str:
            return True

        # 2. Status code attribute check
        status_code = getattr(error, "status_code", None)
        if status_code in (429, 500, 502, 503, 504):
            return True

        # 3. Common LiteLLM / OpenAI / provider retryable exception types
        retryable_types = (
            "InternalServerError",
            "APIError",
            "ServiceUnavailableError",
            "BadGatewayError",
            "GatewayTimeoutError",
            "RateLimitError",
            "APIConnectionError",
            "OverloadedError",
            "OpenAIError",
        )
        if any(t in err_type for t in retryable_types):
            return True

        # 4. HTTP status code word boundary regex and error substrings
        if _STATUS_CODE_PATTERN.search(err_str):
            return True

        if any(
            phrase in err_str.lower()
            for phrase in (
                "no 'choices'",
                "internalservererror",
                "rate limit",
                "rate_limit",
                "ratelimit",
                "overloaded",
                "service unavailable",
                "bad gateway",
                "gateway timeout",
                "resourceexhausted",
                "quota exceeded",
            )
        ):
            return True

        # 5. Network/connection errors
        return isinstance(error, (TimeoutError, ConnectionError, OSError))

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> LlmResponse | None:
        """Intercept model errors and retry with exponential backoff on transient errors."""
        if not self._is_retryable_error(error):
            logger.warning(
                "Non-retryable model error: %s: %s", type(error).__name__, error
            )
            return None

        # Extract agent model reference
        inv_context = getattr(callback_context, "_invocation_context", None)
        agent = getattr(callback_context, "agent", None) or (
            getattr(inv_context, "agent", None) if inv_context else None
        )
        model = (
            getattr(agent, "canonical_model", None) or getattr(agent, "model", None)
            if agent
            else None
        )

        if model is None:
            logger.error("No model found on agent; cannot perform retry.")
            return None

        delay = self.initial_delay
        for attempt in range(1, self.max_retries + 1):
            # Jitter: +-20% randomization
            jittered_delay = delay * (0.8 + 0.4 * random.random())
            logger.warning(
                "⚠️ Model error (%s: %s). Retrying attempt %d/%d in %.1fs...",
                type(error).__name__,
                error,
                attempt,
                self.max_retries,
                jittered_delay,
            )
            await asyncio.sleep(jittered_delay)

            try:
                gen_or_coro = model.generate_content_async(llm_request)
                if hasattr(gen_or_coro, "__aiter__"):
                    responses: list[LlmResponse] = []
                    async for res in gen_or_coro:
                        responses.append(res)
                    if responses:
                        logger.info(
                            "✅ Model call succeeded on retry attempt %d/%d.",
                            attempt,
                            self.max_retries,
                        )
                        return responses[-1]
                elif inspect.isawaitable(gen_or_coro):
                    response = await gen_or_coro
                    if response is not None:
                        logger.info(
                            "✅ Model call succeeded on retry attempt %d/%d.",
                            attempt,
                            self.max_retries,
                        )
                        return response
                else:
                    response = gen_or_coro
                    if response is not None:
                        logger.info(
                            "✅ Model call succeeded on retry attempt %d/%d.",
                            attempt,
                            self.max_retries,
                        )
                        return response
            except Exception as retry_err:  # noqa: BLE001
                error = retry_err
                if not self._is_retryable_error(retry_err):
                    logger.error(
                        "Non-retryable error during retry: %s: %s",
                        type(retry_err).__name__,
                        retry_err,
                    )
                    return None

            delay = min(self.max_delay, delay * self.backoff_factor)

        logger.error(
            "❌ Max retries (%d) exhausted for error: %s", self.max_retries, error
        )
        return None
