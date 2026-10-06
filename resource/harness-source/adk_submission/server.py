"""Local inference server lifecycle manager for metric evaluation environments.

This module provides tools for competition evaluation metrics to spawn,
health-check, and cleanly shut down local API server subprocesses (supporting both
native Hugging Face Transformers and vLLM). It supports dynamic LoRA adapter mounting
from submission manifests and generates pre-configured ModelRegistry instances
populated with ADK LiteLlm clients.

Usage Example:
    Inside a Kaggle metric or evaluation harness:

        from adk_submission import compile_submission, discover_adapters, ToolRegistry
        from adk_submission.server import TransformersConfig, spawn_transformers_server

        config = TransformersConfig(model="google/gemma-4-26b-a4b-it", port=8000)
        adapters = discover_adapters(submission_dir)

        with spawn_transformers_server(config, adapter_manifest=adapters) as server:
            models = server.create_model_registry({"fast": "google/gemma-4-26b-a4b-it"})
            agent = compile_submission(
                submission_dir=submission_dir,
                tool_registry=tools,
                model_registry=models,
                adapter_manifest=adapters,
            )
            # Run evaluation against compiled agent
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from .discovery import AdapterManifest, infer_adapter_rank
from .errors import ServerStartupError, SubmissionValidationError
from .registry import ModelRegistry

import re

logger = logging.getLogger(__name__)

_VALID_ADAPTER_NAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")
_VALID_VLLM_LORA_RANKS: tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def _round_up_vllm_lora_rank(rank: int) -> int:
    """Round an adapter rank up to the smallest supported vLLM max_lora_rank bucket."""
    for valid_rank in _VALID_VLLM_LORA_RANKS:
        if valid_rank >= rank:
            return valid_rank
    return _VALID_VLLM_LORA_RANKS[-1]


@dataclass
class TransformersConfig:
    """Configuration options for spawning a local `transformers serve` API server.

    Attributes:
        model: HuggingFace model identifier or local directory path to the base model.
        port: TCP port on which the server listens. Defaults to 8000.
        host: Host IP interface to bind to. Defaults to "127.0.0.1".
        continuous_batching: Whether to enable continuous batching for higher throughput. Defaults to False.
        attn_implementation: Attention implementation backend (e.g., 'flash_attention_2', 'sdpa'). Defaults to None.
        dtype: Model precision/datatype (e.g. "auto", "bfloat16", "float16", "float32"). Defaults to "auto".
        device_map: Device placement strategy for PyTorch/Accelerate. Defaults to "auto".
        quantization: Quantization method (e.g. 'bnb-4bit', 'bnb-8bit'). Defaults to None.
        extra_args: Additional command-line flags passed directly to transformers serve.
        startup_timeout: Maximum time in seconds to wait for server health readiness. Defaults to 600.
        health_check_interval: Polling interval in seconds during health checking. Defaults to 1.0.
        nvidia_lib_dir: CUDA driver search path for Kaggle GPU environments. Defaults to "/usr/local/nvidia/lib64".
    """

    model: str
    port: int = 8000
    host: str = "127.0.0.1"
    continuous_batching: bool = False
    attn_implementation: str | None = None
    dtype: str = "auto"
    device_map: str = "auto"
    quantization: str | None = None
    extra_args: list[str] = field(default_factory=list)
    startup_timeout: int = 600
    health_check_interval: float = 1.0
    nvidia_lib_dir: str = "/usr/local/nvidia/lib64"


@dataclass
class VllmConfig:
    """Configuration options for spawning a local vLLM API server.

    Attributes:
        model: HuggingFace model identifier or local directory path to the base model.
        port: TCP port on which the vLLM server listens. Defaults to 8000.
        host: Host IP interface to bind to. Defaults to "127.0.0.1".
        tool_call_parser: Tool call parser identifier (e.g., "hermes", "mistral", "llama3_json", "gemma4").
            Defaults to "hermes". If None, tool call parsing flag is omitted.
        reasoning_parser: Reasoning parser identifier (e.g., "gemma4").
            Defaults to None. If None, reasoning parser flag is omitted.
        reasoning_config: Reasoning token delimiter configuration dict or JSON string
            (e.g. {"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}).
            When None and ``reasoning_parser == "gemma4"``, defaults to Gemma 4 channel tokens
            so vLLM's ``thinking_token_budget`` logits processor is enabled.
        default_chat_template_kwargs: Default keyword arguments passed to chat template rendering
            (e.g. {"enable_thinking": True} or JSON string). Defaults to None.
        limit_mm_per_prompt: Multimodal input limits per prompt (e.g. "image=0,audio=0"). Defaults to None.
        speculative_config: Speculative decoding configuration dict or JSON string (e.g. {"model": "...", "num_speculative_tokens": 3}).
            Defaults to None.
        kv_cache_dtype: Precision/datatype for KV cache (e.g. "auto", "fp8", "fp8_e4m3"). Defaults to None.
        max_model_len: Maximum model context length (tokens). Defaults to 32768.
        dtype: Model precision/datatype (e.g. "auto", "half", "bfloat16"). Defaults to "auto".
        gpu_memory_utilization: Fraction of GPU memory allocated to vLLM. Defaults to 0.90.
        enable_auto_tool_choice: Whether to enable auto tool choice in vLLM. Defaults to True.
        enable_lora: Whether to enable LoRA adapter serving. Defaults to True.
        max_loras: Maximum number of active LoRA adapters simultaneously loaded. Defaults to 1.
        max_lora_rank: Fallback maximum LoRA rank supported when adapter rank cannot be inferred. Defaults to 128.
        scheduler_reserve_full_isl: Whether the vLLM scheduler requires the full input sequence length to fit in
            uncomputed KV cache blocks before admitting a request. Defaults to False so hybrid sliding-window models
            (such as Gemma 4) do not stall in the waiting queue when KV cache is smaller than ``max_model_len``.
        extra_args: Additional command-line flags passed directly to vLLM.
        startup_timeout: Maximum time in seconds to wait for server health readiness. Defaults to 600.
        tensor_parallel_size: Number of GPUs for tensor parallelism. Defaults to 1.
        health_check_interval: Polling interval in seconds during health checking. Defaults to 1.0.
        nvidia_lib_dir: CUDA driver search path for Kaggle GPU environments. Defaults to "/usr/local/nvidia/lib64".
    """

    model: str
    port: int = 8000
    host: str = "127.0.0.1"
    tool_call_parser: str | None = "hermes"
    reasoning_parser: str | None = None
    reasoning_config: dict[str, Any] | str | None = None
    default_chat_template_kwargs: dict[str, Any] | str | None = None
    limit_mm_per_prompt: str | None = None
    speculative_config: dict[str, Any] | str | None = None
    kv_cache_dtype: str | None = None
    max_model_len: int = 32768
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.90
    enable_auto_tool_choice: bool = True
    enable_lora: bool = True
    max_loras: int = 1
    max_lora_rank: int = 128
    scheduler_reserve_full_isl: bool = False
    extra_args: list[str] = field(default_factory=list)
    startup_timeout: int = 600
    tensor_parallel_size: int = 1
    health_check_interval: float = 1.0
    nvidia_lib_dir: str = "/usr/local/nvidia/lib64"


def _resolve_lora_checkpoint_path(path: str | Path) -> str:
    """Resolve a LoRA path to its checkpoint directory when pointing to a PEFT weight file."""
    p = Path(path)
    if p.is_dir():
        return str(p)
    if (
        p.is_file()
        or p.suffix.lower() in {".safetensors", ".bin", ".pt", ".gguf", ".ggml"}
    ) and ((p.parent / "adapter_config.json").is_file() or p.stem == "adapter_model"):
        return str(p.parent)
    return str(path)


class BaseInferenceServer:
    """Base lifecycle manager for local inference API server subprocesses."""

    def __init__(
        self,
        config: TransformersConfig | VllmConfig,
        adapter_manifest: AdapterManifest | None = None,
        lora_modules: Mapping[str, str | Path] | None = None,
    ):
        self.config = config
        self.adapter_manifest = adapter_manifest
        self.lora_modules = dict(lora_modules) if lora_modules else {}
        self.process: subprocess.Popen | None = None
        with tempfile.NamedTemporaryFile(
            prefix=f"inference_server_{self.config.port}_",
            suffix=".log",
            delete=False,
        ) as tmp_log:
            self.log_path: str = tmp_log.name
        self._log_file_handle: Any = None

    @property
    def base_url(self) -> str:
        """Return the OpenAI-compatible v1 API endpoint base URL."""
        return f"http://{self.config.host}:{self.config.port}/v1"

    @property
    def health_url(self) -> str:
        """Return the server health check URL."""
        return f"http://{self.config.host}:{self.config.port}/health"

    def build_cmd(self) -> list[str]:
        raise NotImplementedError

    def build_env(self) -> dict[str, str]:
        """Construct execution environment with CUDA library and module paths."""
        env = os.environ.copy()
        env.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        env.setdefault("LITELLM_TELEMETRY", "False")

        # Ensure /tmp is not injected into PYTHONPATH or LD_LIBRARY_PATH
        if "PYTHONPATH" in env:
            cleaned_py = [
                p
                for p in env["PYTHONPATH"].split(":")
                if p and p != "/tmp"
            ]
            if cleaned_py:
                env["PYTHONPATH"] = ":".join(cleaned_py)
            else:
                env.pop("PYTHONPATH", None)

        if "LD_LIBRARY_PATH" in env:
            cleaned_ld = [
                p
                for p in env["LD_LIBRARY_PATH"].split(":")
                if p and p != "/tmp/torch/lib"
            ]
            if cleaned_ld:
                env["LD_LIBRARY_PATH"] = ":".join(cleaned_ld)
            else:
                env.pop("LD_LIBRARY_PATH", None)

        with contextlib.suppress(Exception):
            import torch  # type: ignore[import-not-found]

            if torch.__file__:
                torch_lib = str(Path(torch.__file__).parent / "lib")
                if os.path.exists(torch_lib):
                    existing_ld = env.get("LD_LIBRARY_PATH", "")
                    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{existing_ld}" if existing_ld else torch_lib

        nvidia_lib = getattr(self.config, "nvidia_lib_dir", "/usr/local/nvidia/lib64")
        if os.path.exists(nvidia_lib):
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{nvidia_lib}:{existing_ld}" if existing_ld else nvidia_lib
            existing_lib = env.get("LIBRARY_PATH", "")
            env["LIBRARY_PATH"] = f"{nvidia_lib}:{existing_lib}" if existing_lib else nvidia_lib
        return env

    def is_healthy(self) -> bool:
        """Query the /health endpoint to check server availability."""
        if requests is None:
            raise ImportError(
                "The 'requests' package is required for server health checks. "
                "Install it with: pip install 'adk-submission[serving]'"
            )
        try:
            resp = requests.get(self.health_url, timeout=2)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def start(self) -> Self:
        """Launch the server subprocess and wait until the health endpoint is ready."""
        if self.process is not None and self.process.poll() is None:
            logger.warning("Inference server process is already running.")
            return self

        cmd = self.build_cmd()
        env = self.build_env()

        logger.info(
            "Starting local %s on %s:%d (logging to %s)...",
            self.__class__.__name__,
            self.config.host,
            self.config.port,
            self.log_path,
        )
        try:
            self._log_file_handle = open(self.log_path, "w+", encoding="utf-8")  # noqa: SIM115
            self.process = subprocess.Popen(
                cmd,
                stdout=self._log_file_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except Exception as e:
            if self._log_file_handle is not None:
                with contextlib.suppress(Exception):
                    self._log_file_handle.close()
                self._log_file_handle = None
            raise ServerStartupError(f"Failed to spawn server process: {e}") from e

        try:
            self._wait_until_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def _wait_until_ready(self) -> None:
        """Poll the health check endpoint until ready or timeout."""
        start_time = time.monotonic()
        timeout = self.config.startup_timeout
        last_status_log = 0.0

        logger.info(
            "Loading model weights (timeout=%ss). Server logs: %s",
            timeout,
            self.log_path,
        )

        while time.monotonic() - start_time < timeout:
            if self.process is not None and self.process.poll() is not None:
                out = self._read_captured_output()
                raise ServerStartupError(
                    f"Inference server exited prematurely with code {self.process.returncode}.",
                    output=out,
                )

            if self.is_healthy():
                elapsed = time.monotonic() - start_time
                logger.info("Inference server is healthy and ready (took ~%.1fs)", elapsed)
                return

            elapsed = time.monotonic() - start_time
            if elapsed - last_status_log >= 15.0:
                last_status_log = elapsed
                logger.info("Waiting for inference server to become ready (~%.0fs elapsed)...", elapsed)

            time.sleep(self.config.health_check_interval)

        out = self._read_captured_output()
        self.stop()
        log_tail = out[-2000:] if out else "(no server output captured)"
        raise ServerStartupError(
            f"Inference server did not become healthy within {timeout}s.\n"
            f"--- Server Log Tail ({self.log_path}) ---\n{log_tail}",
            output=out,
        )

    def _read_captured_output(self) -> str:
        """Read and decode available output from the server log file."""
        if self._log_file_handle is not None:
            with contextlib.suppress(Exception):
                self._log_file_handle.flush()
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception:  # noqa: BLE001
                return ""
        return ""

    def stop(self, timeout: float = 10.0) -> None:
        """Gracefully terminate the server process with fallback to kill."""
        if self._log_file_handle is not None:
            with contextlib.suppress(Exception):
                self._log_file_handle.flush()
                self._log_file_handle.close()
            self._log_file_handle = None

        if self.process is None:
            return

        if self.process.poll() is not None:
            self.process = None
            return

        logger.info("Stopping inference server process (PID %s)...", self.process.pid)
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Inference server did not terminate within %.1fs; killing...", timeout)
                self.process.kill()
                self.process.wait(timeout=5)
        except Exception as e:  # noqa: BLE001
            logger.error("Error stopping server process: %s", e)
        finally:
            self.process = None
            if self.log_path and os.path.exists(self.log_path):
                with contextlib.suppress(Exception):
                    os.unlink(self.log_path)

    def create_model_registry(
        self,
        aliases: Mapping[str, str] | Sequence[str] | str | None = None,
        model_prefix: str | None = "hosted_vllm/",
        api_key: str | None = "EMPTY",
    ) -> ModelRegistry:
        """Create and populate a ModelRegistry with LiteLlm clients targeting this server."""
        registry = ModelRegistry()

        target_model_id = getattr(self, "effective_model_path", self.config.model)
        if aliases is None:
            model_aliases = {self.config.model: target_model_id}
        elif isinstance(aliases, str):
            model_aliases = {aliases: target_model_id}
        elif isinstance(aliases, Mapping):
            model_aliases = dict(aliases)
        else:
            model_aliases = {alias: target_model_id for alias in aliases}

        lora_names: list[str] = []
        if self.adapter_manifest and getattr(self.adapter_manifest, "adapters", None):
            lora_names.extend(self.adapter_manifest.adapters.keys())
        if self.lora_modules:
            lora_names.extend(self.lora_modules.keys())

        if isinstance(self, TransformersServer) and len(lora_names) > 1:
            raise ServerStartupError(
                "TransformersServer only supports serving a single merged LoRA adapter at a time; "
                "use VllmServer for multiple adapters."
            )
        if isinstance(self, TransformersServer) and lora_names and not getattr(self, "_merged_model_dir", None):
            raise ServerStartupError(
                "TransformersServer cannot register LoRA adapter aliases without a merged model directory."
            )

        for lora_name in lora_names:
            is_vllm_lora = isinstance(self, VllmServer) and getattr(self.config, "enable_lora", True)
            resolved_lora_target = lora_name if is_vllm_lora else target_model_id
            if lora_name not in model_aliases:
                model_aliases[lora_name] = resolved_lora_target
            prefixed_lora_name = f"adapter:{lora_name}"
            if prefixed_lora_name not in model_aliases:
                model_aliases[prefixed_lora_name] = resolved_lora_target

        def _format_model_id(model_name: str) -> str:
            if not model_prefix:
                return model_name
            if any(model_name.startswith(p) for p in ("hosted_vllm/", "openai/", "custom/")):
                return model_name
            return f"{model_prefix}{model_name}"

        with contextlib.suppress(Exception):
            import litellm

            litellm.telemetry = False
            litellm.suppress_debug_info = True
            litellm.set_verbose = False
            if hasattr(litellm, "_logging") and hasattr(litellm._logging, "_disable_debugging"):
                litellm._logging._disable_debugging()
            os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
            os.environ.setdefault("LITELLM_TELEMETRY", "False")
            os.environ.setdefault("LITELLM_LOG", "WARNING")
            for _logger_name in (
                "LiteLLM",
                "LiteLLM Router",
                "LiteLLM Proxy",
                "litellm",
                "httpx",
                "httpcore",
                "openai",
            ):
                _lg = logging.getLogger(_logger_name)
                _lg.setLevel(logging.WARNING)
                _lg.propagate = False

        try:
            from google.adk.models.lite_llm import LiteLlm
            from .resolvers.generation import install_litellm_reasoning_patch

            install_litellm_reasoning_patch()

            for alias, model_name in model_aliases.items():
                registry.register(
                    alias,
                    LiteLlm(
                        model=_format_model_id(model_name),
                        api_base=self.base_url,
                        api_key=api_key or "EMPTY",
                    ),
                )
        except ImportError:
            for alias, model_name in model_aliases.items():
                registry.register(alias, _format_model_id(model_name))

        return registry

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.stop()


class TransformersServer(BaseInferenceServer):
    """Manages the lifecycle of a local Hugging Face `transformers serve` API server subprocess."""

    def __init__(
        self,
        config: TransformersConfig,
        adapter_manifest: AdapterManifest | None = None,
        lora_modules: Mapping[str, str | Path] | None = None,
    ):
        super().__init__(config=config, adapter_manifest=adapter_manifest, lora_modules=lora_modules)
        self.config: TransformersConfig = config
        self._merged_model_dir: str | None = None

    @property
    def effective_model_path(self) -> str:
        """Return the model path to serve (merged path if LoRA was merged, else config.model)."""
        if self._merged_model_dir:
            return self._merged_model_dir
        return self.config.model

    def _merge_lora_if_needed(self) -> None:
        """Merge LoRA adapter into base model in a temporary directory if an adapter is provided."""
        manifest_count = len(self.adapter_manifest.adapters) if (self.adapter_manifest and self.adapter_manifest.adapters) else 0
        modules_count = len(self.lora_modules) if self.lora_modules else 0
        if manifest_count + modules_count > 1:
            raise ServerStartupError(
                "TransformersServer only supports serving a single merged LoRA adapter at a time; "
                "use VllmServer for multiple adapters."
            )

        lora_path = None
        if self.adapter_manifest and self.adapter_manifest.adapters:
            first_adapter = next(iter(self.adapter_manifest.adapters.values()))
            lora_path = str(first_adapter.path)
        elif self.lora_modules:
            first_path = next(iter(self.lora_modules.values()))
            lora_path = str(first_path)

        if not lora_path:
            return

        target_dir = tempfile.mkdtemp(prefix=f"merged_model_{self.config.port}_")
        logger.info(
            "Merging LoRA adapter from %s into base model %s at %s...",
            lora_path,
            self.config.model,
            target_dir,
        )
        try:
            import torch  # type: ignore[import-not-found]
            from peft import PeftModel  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            cfg_dtype = getattr(self.config, "dtype", "auto")
            if cfg_dtype and cfg_dtype != "auto" and hasattr(torch, cfg_dtype):
                resolved_dtype = getattr(torch, cfg_dtype)
            elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                resolved_dtype = torch.bfloat16
            elif torch.cuda.is_available():
                resolved_dtype = torch.float16
            else:
                resolved_dtype = torch.float32

            cfg_device_map = getattr(self.config, "device_map", "auto")
            resolved_device_map = (
                cfg_device_map
                if (cfg_device_map and cfg_device_map != "auto")
                else ("auto" if torch.cuda.is_available() else "cpu")
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                torch_dtype=resolved_dtype,
                device_map=resolved_device_map,
                trust_remote_code=True,
            )
            lora_dir = _resolve_lora_checkpoint_path(lora_path)
            model = PeftModel.from_pretrained(
                base_model, lora_dir, use_safetensors=True
            )
            merged = model.merge_and_unload()
            merged.save_pretrained(target_dir)

            tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
            tokenizer.save_pretrained(target_dir)
            self._merged_model_dir = target_dir
        except Exception as e:
            if os.path.exists(target_dir):
                with contextlib.suppress(Exception):
                    import shutil

                    shutil.rmtree(target_dir, ignore_errors=True)
            raise ServerStartupError(
                f"Failed to merge LoRA adapter '{lora_path}' into base model '{self.config.model}': {e}"
            ) from e

    def build_cmd(self) -> list[str]:
        """Construct the CLI command to launch the built-in transformers serve CLI."""
        # Detect whether transformers v5 or v4 CLI is available.
        is_v5 = False
        with contextlib.suppress(Exception):
            import importlib.util

            is_v5 = importlib.util.find_spec("transformers.cli.transformers") is not None

        if is_v5:
            cmd = [
                sys.executable,
                "-m",
                "transformers.cli.transformers",
                "serve",
                self.effective_model_path,
                "--host",
                self.config.host,
                "--port",
                str(self.config.port),
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "transformers.commands.transformers_cli",
                "serve",
                "--model",
                self.effective_model_path,
                "--host",
                self.config.host,
                "--port",
                str(self.config.port),
            ]

        if self.config.continuous_batching:
            cmd.append("--continuous-batching")

        if self.config.attn_implementation:
            cmd.extend(["--attn-implementation", self.config.attn_implementation])

        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])

        if self.config.dtype and self.config.dtype != "auto":
            cmd.extend(["--dtype", self.config.dtype])

        if self.config.device_map and self.config.device_map != "auto":
            cmd.extend(["--device", self.config.device_map])

        cmd.extend(self.config.extra_args)
        return cmd

    def start(self) -> BaseInferenceServer:
        """Merge LoRA if provided, then launch the server subprocess."""
        self._merge_lora_if_needed()
        return super().start()

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate server process and clean up any merged model directory."""
        super().stop(timeout=timeout)
        if self._merged_model_dir and os.path.exists(self._merged_model_dir):
            with contextlib.suppress(Exception):
                import shutil

                shutil.rmtree(self._merged_model_dir, ignore_errors=True)
            self._merged_model_dir = None


class VllmServer(BaseInferenceServer):
    """Manages the lifecycle of a local vLLM API server subprocess."""

    def __init__(
        self,
        config: VllmConfig,
        adapter_manifest: AdapterManifest | None = None,
        lora_modules: Mapping[str, str | Path] | None = None,
    ):
        super().__init__(config=config, adapter_manifest=adapter_manifest, lora_modules=lora_modules)
        self.config: VllmConfig = config

    def build_cmd(self) -> list[str]:
        """Construct the CLI command to launch vLLM OpenAI API server."""
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--max-model-len",
            str(self.config.max_model_len),
            "--dtype",
            self.config.dtype,
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
        ]

        if self.config.tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(self.config.tensor_parallel_size)])

        if self.config.enable_auto_tool_choice:
            cmd.append("--enable-auto-tool-choice")

        if self.config.tool_call_parser:
            cmd.extend(["--tool-call-parser", self.config.tool_call_parser])

        reasoning_parser = getattr(self.config, "reasoning_parser", None)
        if reasoning_parser:
            cmd.extend(["--reasoning-parser", str(reasoning_parser)])

        reasoning_cfg = getattr(self.config, "reasoning_config", None)
        if (
            reasoning_cfg is None
            and reasoning_parser == "gemma4"
            and "--reasoning-config" not in self.config.extra_args
        ):
            reasoning_cfg = {
                "reasoning_start_str": "<|channel>",
                "reasoning_end_str": "<channel|>",
            }
        if reasoning_cfg:
            if isinstance(reasoning_cfg, dict):
                reasoning_cfg = json.dumps(reasoning_cfg)
            cmd.extend(["--reasoning-config", str(reasoning_cfg)])

        kwargs_val = getattr(self.config, "default_chat_template_kwargs", None)
        if kwargs_val:
            if isinstance(kwargs_val, dict):
                kwargs_val = json.dumps(kwargs_val)
            cmd.extend(["--default-chat-template-kwargs", str(kwargs_val)])

        if getattr(self.config, "limit_mm_per_prompt", None):
            cmd.extend(["--limit-mm-per-prompt", str(self.config.limit_mm_per_prompt)])

        if getattr(self.config, "kv_cache_dtype", None):
            cmd.extend(["--kv-cache-dtype", str(self.config.kv_cache_dtype)])

        spec_val = getattr(self.config, "speculative_config", None)
        if spec_val:
            if isinstance(spec_val, dict):
                spec_val = json.dumps(spec_val)
            cmd.extend(["--speculative-config", str(spec_val)])

        if (
            not getattr(self.config, "scheduler_reserve_full_isl", True)
            and "--no-scheduler-reserve-full-isl" not in self.config.extra_args
            and "--scheduler-reserve-full-isl" not in self.config.extra_args
        ):
            cmd.append("--no-scheduler-reserve-full-isl")

        modules_to_mount: dict[str, str] = {}
        detected_ranks: list[int] = []
        all_ranks_known = True

        if self.adapter_manifest:
            for name, info in self.adapter_manifest.adapters.items():
                modules_to_mount[name] = _resolve_lora_checkpoint_path(info.path)
                r_val = getattr(info, "rank", None)
                if r_val is None:
                    r_val = infer_adapter_rank(info.path)
                if r_val is not None and r_val > 0:
                    detected_ranks.append(r_val)
                else:
                    all_ranks_known = False
        for name, path in self.lora_modules.items():
            modules_to_mount[name] = _resolve_lora_checkpoint_path(path)
            r_val = infer_adapter_rank(path)
            if r_val is not None and r_val > 0:
                detected_ranks.append(r_val)
            else:
                all_ranks_known = False

        if self.config.enable_lora and modules_to_mount:
            cmd.append("--enable-lora")
            effective_max_loras = len(modules_to_mount)
            if all_ranks_known and detected_ranks:
                effective_max_lora_rank = _round_up_vllm_lora_rank(max(detected_ranks))
            else:
                effective_max_lora_rank = _round_up_vllm_lora_rank(
                    max([self.config.max_lora_rank, *detected_ranks])
                )
            cmd.extend(["--max-loras", str(effective_max_loras)])
            cmd.extend(["--max-lora-rank", str(effective_max_lora_rank)])
            cmd.append("--lora-modules")
            for name, path in modules_to_mount.items():
                if not _VALID_ADAPTER_NAME_RE.match(name):
                    raise SubmissionValidationError(
                        f"Invalid adapter name '{name}': must match ^[a-zA-Z0-9_.-]+$"
                    )
                cmd.append(f"{name}={path}")

        cmd.extend(self.config.extra_args)
        return cmd


def spawn_transformers_server(
    config: TransformersConfig | None = None,
    adapter_manifest: AdapterManifest | None = None,
    lora_modules: Mapping[str, str | Path] | None = None,
    **kwargs: Any,
) -> TransformersServer:
    """Create a TransformersServer instance for native Hugging Face `transformers serve`."""
    if config is None:
        config = TransformersConfig(**kwargs)
    return TransformersServer(
        config=config,
        adapter_manifest=adapter_manifest,
        lora_modules=lora_modules,
    )


def spawn_vllm_server(
    config: VllmConfig | None = None,
    adapter_manifest: AdapterManifest | None = None,
    lora_modules: Mapping[str, str | Path] | None = None,
    **kwargs: Any,
) -> VllmServer:
    """Create a VllmServer instance for vLLM serving."""
    if config is None:
        config = VllmConfig(**kwargs)
    return VllmServer(
        config=config,
        adapter_manifest=adapter_manifest,
        lora_modules=lora_modules,
    )


def spawn_server(
    backend: str = "transformers",
    config: TransformersConfig | VllmConfig | None = None,
    adapter_manifest: AdapterManifest | None = None,
    lora_modules: Mapping[str, str | Path] | None = None,
    **kwargs: Any,
) -> BaseInferenceServer:
    """Universal inference server launcher supporting transformers and vllm backends."""
    if backend == "transformers":
        if config is not None and not isinstance(config, TransformersConfig):
            raise TypeError(f"Expected TransformersConfig for {backend} backend, got {type(config).__name__}")
        return spawn_transformers_server(config=config, adapter_manifest=adapter_manifest, lora_modules=lora_modules, **kwargs)
    elif backend == "vllm":
        if config is not None and not isinstance(config, VllmConfig):
            raise TypeError(f"Expected VllmConfig for vllm backend, got {type(config).__name__}")
        return spawn_vllm_server(config=config, adapter_manifest=adapter_manifest, lora_modules=lora_modules, **kwargs)
    else:
        raise ValueError(f"Unknown inference backend: {backend}. Expected transformers or vllm.")


__all__ = [
    "BaseInferenceServer",
    "TransformersConfig",
    "TransformersServer",
    "VllmConfig",
    "VllmServer",
    "spawn_server",
    "spawn_transformers_server",
    "spawn_vllm_server",
]
