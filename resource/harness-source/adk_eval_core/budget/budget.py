"""Token budget tracking and model pricing for ADK agent evaluations.

Provides:
- ``ModelPricing`` / ``PricingTable``: per-token cost lookup from YAML config.
- ``TokenBudget``: accumulates token usage, computes running cost in USD,
  and provides budget enforcement (exceeded detection) and status reporting.
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from adk_eval_core.errors import BudgetExceededError, InvalidBudgetError

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "models.yaml"


def _is_valid_non_negative(val: Any) -> bool:
    """Return True if val is a finite non-negative int or float (excluding bool)."""
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(val)
        and val >= 0
    )


@dataclass(frozen=True)
class ModelPricing:
    """Per-token pricing for a model (USD per 1M tokens)."""

    model_id: str
    path: str
    display_name: str
    input_price_per_million: float  # $/1M input tokens
    cached_input_price_per_million: float  # $/1M cached input tokens
    output_price_per_million: float  # $/1M output tokens

    def __post_init__(self) -> None:
        if not (
            _is_valid_non_negative(self.input_price_per_million)
            and _is_valid_non_negative(self.cached_input_price_per_million)
            and _is_valid_non_negative(self.output_price_per_million)
        ):
            raise InvalidBudgetError(
                "Model pricing rates must be finite non-negative numbers",
                model_id=self.model_id,
            )

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        """Compute cost in USD for a given number of tokens."""
        if not (
            _is_valid_non_negative(input_tokens)
            and _is_valid_non_negative(output_tokens)
            and _is_valid_non_negative(cached_input_tokens)
        ):
            raise InvalidBudgetError(
                "Token counts must be finite non-negative numbers",
                model_id=self.model_id,
            )
        non_cached_input = max(0, input_tokens - cached_input_tokens)
        input_cost = (non_cached_input / 1_000_000) * self.input_price_per_million
        cached_cost = (cached_input_tokens / 1_000_000) * self.cached_input_price_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_million
        return input_cost + cached_cost + output_cost


class PricingTable:
    """Registry of model pricing loaded from a YAML config file.

    Usage::

        table = PricingTable.from_yaml("models.yaml")
        pricing = table.get("openai/google/gemini-3-flash-preview")
        cost = pricing.cost(input_tokens=5000, output_tokens=1200)
    """

    def __init__(self, models: dict[str, ModelPricing]) -> None:
        self._models = dict(models)
        self._by_path = {m.path: m for m in self._models.values() if m.path}

    def register(self, pricing: ModelPricing) -> None:
        """Register or update a model pricing entry in the table."""
        self._models[pricing.model_id] = pricing
        if pricing.path:
            self._by_path[pricing.path] = pricing

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> PricingTable:
        """Load pricing from a YAML config file.

        Supports both dictionary mapping format (models: {id: {...}})
        and list format (models: [{model_id: ...}]).
        Returns a shallow copy so caller mutations do not corrupt the cache.
        """
        if path is None:
            path = _DEFAULT_CONFIG_PATH
        path_str = str(Path(path).resolve())
        cached = cls._from_yaml_cached(path_str)
        return cls(dict(cached._models))

    @classmethod
    @functools.lru_cache
    def _from_yaml_cached(cls, path_str: str) -> PricingTable:
        path = Path(path_str)
        if not path.is_file() and not path.exists():
            logger.warning("Pricing config not found at %s — using empty table", path)
            return cls({})

        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        models_data = data.get("models", {})
        models: dict[str, ModelPricing] = {}

        if isinstance(models_data, dict):
            for model_id, info in models_data.items():
                if not isinstance(info, dict):
                    logger.warning("Skipping malformed model entry: %s", model_id)
                    continue
                try:
                    input_price = float(
                        info.get("input_price_per_million_tokens", info.get("input_price_per_million", 0.0))
                    )
                    cached_input_price = float(
                        info.get(
                            "cached_input_price_per_million_tokens",
                            info.get("cached_input_price_per_million", 0.0),
                        )
                    )
                    output_price = float(
                        info.get("output_price_per_million_tokens", info.get("output_price_per_million", 0.0))
                    )
                except (TypeError, ValueError):
                    logger.warning("Non-numeric price for model %s — skipping", model_id)
                    continue
                if not (
                    _is_valid_non_negative(input_price)
                    and _is_valid_non_negative(cached_input_price)
                    and _is_valid_non_negative(output_price)
                ):
                    logger.warning("Invalid or negative price for model %s — skipping", model_id)
                    continue
                models[model_id] = ModelPricing(
                    model_id=model_id,
                    path=info.get("path", f"models/{model_id}"),
                    display_name=info.get("display_name", model_id),
                    input_price_per_million=input_price,
                    cached_input_price_per_million=cached_input_price,
                    output_price_per_million=output_price,
                )
        elif isinstance(models_data, list):
            for entry in models_data:
                if not isinstance(entry, dict) or "model_id" not in entry:
                    continue
                mid = entry["model_id"]
                try:
                    input_price = float(
                        entry.get("input_price_per_million_tokens", entry.get("input_price_per_million", 0.0))
                    )
                    cached_input_price = float(
                        entry.get(
                            "cached_input_price_per_million_tokens",
                            entry.get("cached_input_price_per_million", 0.0),
                        )
                    )
                    output_price = float(
                        entry.get("output_price_per_million_tokens", entry.get("output_price_per_million", 0.0))
                    )
                except (TypeError, ValueError):
                    continue
                if not (
                    _is_valid_non_negative(input_price)
                    and _is_valid_non_negative(cached_input_price)
                    and _is_valid_non_negative(output_price)
                ):
                    logger.warning("Invalid or negative price for model %s — skipping", mid)
                    continue
                models[mid] = ModelPricing(
                    model_id=mid,
                    path=entry.get("path", f"models/{mid}"),
                    display_name=entry.get("display_name", mid),
                    input_price_per_million=input_price,
                    cached_input_price_per_million=cached_input_price,
                    output_price_per_million=output_price,
                )

        logger.info("Loaded pricing for %d models from %s", len(models), path)
        return cls(models)

    def get(self, model_id: str) -> ModelPricing | None:
        """Look up pricing for a model ID.

        Tries exact match first, then checks path and common prefixes.
        """
        if model_id in self._models:
            return self._models[model_id]

        if model_id in self._by_path:
            return self._by_path[model_id]

        # Try without 'openai/' prefix
        stripped = model_id.removeprefix("openai/")
        if stripped in self._models:
            return self._models[stripped]
        if stripped in self._by_path:
            return self._by_path[stripped]

        # Try adding 'openai/' prefix
        prefixed = f"openai/{model_id}"
        if prefixed in self._models:
            return self._models[prefixed]
        if prefixed in self._by_path:
            return self._by_path[prefixed]

        # Try stripping vendor prefixes (e.g. google/gemini-2.5-pro)
        if "/" in model_id:
            clean_id = model_id.split("/")[-1]
            if clean_id in self._models:
                return self._models[clean_id]

        # Try stripping version suffix (e.g., @latest, @20250101).
        base_id = model_id.split("@")[0]
        if base_id != model_id:
            result = self.get(base_id)
            if result is not None:
                return result

        return None

    @property
    def model_ids(self) -> list[str]:
        """Return all known model IDs."""
        return list(self._models.keys())

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        return self.get(model_id) is not None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PricingTable):
            return NotImplemented
        return self._models == other._models


@dataclass
class TokenBudget:
    """Tracks cumulative token usage and cost for an agent session.

    Usage::

        budget = TokenBudget(max_budget_usd=5.0)
        cost = budget.record_usage(
            model_id="gemini-3-flash-preview",
            input_tokens=1200,
            output_tokens=350,
        )
    """

    max_budget_usd: float | None = 5.0
    pricing_table: PricingTable | None = None
    max_llm_calls: int | None = None

    total_input_tokens: int = field(default=0, init=False)
    last_input_tokens: int = field(default=0, init=False)
    total_cached_input_tokens: int = field(default=0, init=False)
    total_output_tokens: int = field(default=0, init=False)
    total_cost_usd: float = field(default=0.0, init=False)
    llm_calls: int = field(default=0, init=False)
    _warned_models: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_budget_usd is not None and not _is_valid_non_negative(self.max_budget_usd):
            raise InvalidBudgetError("max_budget_usd must be a finite non-negative number")
        if self.max_llm_calls is not None and not _is_valid_non_negative(self.max_llm_calls):
            raise InvalidBudgetError("max_llm_calls must be a finite non-negative number")
        if self.pricing_table is None:
            self.pricing_table = PricingTable.from_yaml()

    def record_usage(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """Record usage from an LLM call and return its cost in USD."""
        if not (
            _is_valid_non_negative(input_tokens)
            and _is_valid_non_negative(output_tokens)
            and _is_valid_non_negative(cached_input_tokens)
        ):
            raise InvalidBudgetError(
                "Token counts must be finite non-negative numbers",
                model_id=model_id,
            )

        pricing = self.pricing_table.get(model_id) if self.pricing_table else None
        if pricing is None and model_id:
            if model_id not in self._warned_models:
                logger.warning("No pricing found for model %r — cost will be 0", model_id)
                self._warned_models.add(model_id)
            cost = 0.0
        else:
            cost = compute_cost(
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                pricing_table=self.pricing_table,
            )
        self.total_input_tokens += int(input_tokens)
        self.last_input_tokens = int(input_tokens)
        self.total_cached_input_tokens += int(cached_input_tokens)
        self.total_output_tokens += int(output_tokens)
        self.total_cost_usd += cost
        self.llm_calls += 1
        return cost

    @property
    def is_exceeded(self) -> bool:
        """Return True if cost exceeds max_budget_usd or llm_calls exceeds max_llm_calls."""
        cost_exceeded = False
        if self.max_budget_usd is not None:
            if self.max_budget_usd == 0.0:
                cost_exceeded = self.total_cost_usd > 0.0 or self.llm_calls > 0
            else:
                cost_exceeded = not math.isfinite(self.total_cost_usd) or self.total_cost_usd >= self.max_budget_usd
        calls_exceeded = self.max_llm_calls is not None and self.llm_calls >= self.max_llm_calls
        return bool(cost_exceeded or calls_exceeded)

    @property
    def remaining_usd(self) -> float:
        """Remaining budget in USD (float inf if unlimited)."""
        if self.max_budget_usd is None:
            return float("inf")
        return max(0.0, self.max_budget_usd - self.total_cost_usd)

    def check_budget(self) -> None:
        """Raise BudgetExceededError if budget or call limit is exceeded."""
        if self.is_exceeded:
            raise BudgetExceededError(
                message="Budget limit exceeded",
                total_cost_usd=self.total_cost_usd,
                max_budget_usd=self.max_budget_usd,
                llm_calls=self.llm_calls,
                max_llm_calls=self.max_llm_calls,
            )

    def to_status_dict(self) -> dict[str, Any]:
        """Return budget status for reporting to the agent."""
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6)
            if self.remaining_usd != float("inf")
            else "unlimited",
            "max_budget_usd": self.max_budget_usd,
            "total_input_tokens": self.total_input_tokens,
            "last_input_tokens": self.last_input_tokens,
            "total_cached_input_tokens": self.total_cached_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "llm_calls": self.llm_calls,
        }


def compute_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    pricing_table: PricingTable | None = None,
) -> float:
    """Compute cost in USD for a single LLM call."""
    if not (
        _is_valid_non_negative(input_tokens)
        and _is_valid_non_negative(output_tokens)
        and _is_valid_non_negative(cached_input_tokens)
    ):
        raise InvalidBudgetError(
            "Token counts must be finite non-negative numbers",
            model_id=model_id,
        )

    if pricing_table is None:
        pricing_table = PricingTable.from_yaml()

    pricing = pricing_table.get(model_id)
    if pricing is None:
        logger.warning("No pricing found for model %r — cost will be 0", model_id)
        return 0.0

    return pricing.cost(input_tokens, output_tokens, cached_input_tokens)
