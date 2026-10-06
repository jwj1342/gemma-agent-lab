"""Gemma 4 model registry setup for SWE-bench evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from adk_submission import ModelRegistry
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

# Auto-load .env environment variables if present without clobbering caller environment
load_dotenv(override=False)


def normalize_api_endpoint(url: str | None) -> str | None:
    """Normalize Model Proxy and LiteLLM API base endpoints.

    Ensures that Kaggle Model Proxy URLs point to the /openapi endpoint
    (e.g., https://mp-staging.kaggle.net/models/ -> https://mp-staging.kaggle.net/models/openapi)
    so OpenAI-compatible chat completion requests are properly routed.
    """
    if not url:
        return None
    url = url.rstrip('/')
    if url.endswith('/models'):
        return f'{url}/openapi'
    if '/models' in url and not (
        url.endswith('/openapi') or url.endswith('/v1') or url.endswith('/genai')
    ):
        return f'{url}/openapi'
    return url


def resolve_swegemma_adapter(
    base_model: Any,
    adapter_info: Any,
    model_registry: ModelRegistry | None = None,
) -> Any:
    """Resolve an adapter declaration to a configured LiteLlm model instance."""
    import copy

    adapter_name = getattr(adapter_info, 'name', str(adapter_info))
    if model_registry is not None and adapter_name in model_registry:
        return model_registry.get(adapter_name)

    if hasattr(base_model, 'model') and isinstance(base_model.model, str):
        target_model = f'openai/{adapter_name}'
        for prefix in ('openai/', 'hosted_vllm/', 'custom/'):
            if base_model.model.startswith(prefix):
                target_model = f'{prefix}{adapter_name}'
                break
        if hasattr(base_model, 'model_copy'):
            return base_model.model_copy(update={'model': target_model})
        cloned = copy.copy(base_model)
        cloned.model = target_model
        return cloned

    if isinstance(base_model, str):
        for prefix in ('openai/', 'hosted_vllm/', 'custom/'):
            if base_model.startswith(prefix):
                return f'{prefix}{adapter_name}'
        return f'openai/{adapter_name}'

    return base_model


def setup_gemma_model_registry(
    api_base: str | None = None,
    api_key: str | None = None,
    num_retries: int = 5,
    models_yaml_path: Path | str | None = None,
    served_model: str | None = None,
    adapter_manifest: Any | None = None,
    backend: str = 'vllm',
) -> ModelRegistry:
    """Populates an ADK ModelRegistry for local Gemma 4 variants via LiteLLM.

    Args:
        api_base: Optional LiteLLM / OpenAI-compatible endpoint URL.
        api_key: Optional API key for local inference server.
        num_retries: Number of retries for LiteLLM requests.
        models_yaml_path: Optional path to custom models.yaml definition.
        served_model: Optional served model name or filesystem path pinned by the local
            inference server (e.g. '/kaggle/input/models/google/gemma-4/transformers/...').
            When provided, all Gemma 4 aliases route to this target model identifier.
        adapter_manifest: Optional discovered AdapterManifest containing LoRA adapters.
        backend: Serving backend ('vllm' or 'transformers') determining LoRA model routing.

    Returns:
        ModelRegistry mapping Gemma 4 model names and adapter names to configured LiteLlm instances.
    """
    raw_endpoint = (
        api_base
        or os.environ.get('MODEL_PROXY_URL')
        or os.environ.get('LITELLM_API_BASE')
        or os.environ.get('LOCAL_INFERENCE_URL')
        or os.environ.get('OPENAI_BASE_URL')
        or 'http://localhost:8000/v1'
    )
    endpoint = normalize_api_endpoint(raw_endpoint)
    key = (
        api_key
        or os.environ.get('MODEL_PROXY_API_KEY')
        or os.environ.get('LITELLM_API_KEY')
        or os.environ.get('LOCAL_API_KEY')
        or os.environ.get('OPENAI_API_KEY')
        or 'EMPTY'
    )

    models = ModelRegistry()

    # Register standard Gemma 4 family model names
    gemma_configs = {
        'gemma-4-31b-it-qat-w4a16-ct': 'openai/gemma-4-31b-it-qat-w4a16-ct',
        'gemma-4-31b-it': 'openai/gemma-4-31b-it',
        'gemma-4-27b-it': 'openai/gemma-4-27b-it',
        'gemma-4-26b-a4b-it': 'openai/gemma-4-26b-a4b-it',
        'gemma-4-12b-it': 'openai/gemma-4-12b-it',
        'gemma-4-9b-it': 'openai/gemma-4-9b-it',
        'gemma-4-e4b-it': 'openai/gemma-4-e4b-it',
        'gemma-4-e2b-it': 'openai/gemma-4-e2b-it',
        'gemma-4-31b': 'openai/gemma-4-31b',
        'gemma-4-27b': 'openai/gemma-4-27b',
        'gemma-4-26b-a4b': 'openai/gemma-4-26b-a4b',
        'gemma-4-12b': 'openai/gemma-4-12b',
        'gemma-4-9b': 'openai/gemma-4-9b',
        'gemma-4-e4b': 'openai/gemma-4-e4b',
        'gemma-4-e2b': 'openai/gemma-4-e2b',
        'diffusiongemma-26b-a4b-it': 'openai/diffusiongemma-26b-a4b-it',
    }

    target_model_for_merged = (
        (
            served_model
            if served_model.startswith(('openai/', 'hosted_vllm/', 'custom/'))
            else f'openai/{served_model}'
        )
        if served_model
        else 'openai/gemma-4-31b-it'
    )

    for alias, default_model_path in gemma_configs.items():
        target_model = (
            target_model_for_merged if served_model else default_model_path
        )

        models.register(
            alias,
            LiteLlm(
                model=target_model,
                api_base=endpoint,
                api_key=key,
                num_retries=num_retries,
            ),
        )

    if adapter_manifest and getattr(adapter_manifest, 'adapters', None):
        def _make_adapter_model(info: Any) -> Any:
            adapter_name = getattr(info, 'name', str(info))
            lora_target = (
                target_model_for_merged
                if backend == 'transformers' and served_model
                else f'openai/{adapter_name}'
            )
            return LiteLlm(
                model=lora_target,
                api_base=endpoint,
                api_key=key,
                num_retries=num_retries,
            )

        adapter_manifest.register_all(models, _make_adapter_model)
        for adapter_name, info in adapter_manifest.adapters.items():
            models.register(adapter_name, _make_adapter_model(info))

    # If custom models.yaml is provided and exists, parse models from it using adk_submission's loader
    yaml_path = Path(models_yaml_path).resolve() if models_yaml_path else None
    if yaml_path and yaml_path.exists():
        from adk_submission.yaml_loader import load_yaml

        data = load_yaml(yaml_path, yaml_path.parent) or {}
        defined_models = data.get('models', {})
        for model_alias, cfg in defined_models.items():
            model_name = cfg.get('path') or f'openai/{model_alias}'
            model_endpoint = normalize_api_endpoint(cfg.get('api_base')) or endpoint
            model_key = cfg.get('api_key') or key
            models.register(
                model_alias,
                LiteLlm(
                    model=model_name,
                    api_base=model_endpoint,
                    api_key=model_key,
                    num_retries=num_retries,
                ),
            )

    return models

