"""Submission directory validation, adapter discovery, and skill discovery.

Provides utilities to validate submission directory structures against configured limits
(file counts, sizes, allowed extensions), and discover model adapters (LoRA weights)
and ADK skills (directories containing SKILL.md). Discovered assets are catalogued into
manifests that can be bulk-registered directly into closed registries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import LimitExceededError, PathTraversalError, SubmissionValidationError
from .limits import SubmissionLimits
from .paths import validate_sandboxed_path
from .registry import ModelRegistry, SkillRegistry
from .yaml_loader import set_active_limits_for_root

# Conventional root config filenames (checked in priority order).
_ROOT_CONFIG_NAMES: list[str] = ["agent.yaml", "root_agent.yaml"]


# ---------------------------------------------------------------------------
# Validated directory view
# ---------------------------------------------------------------------------


@dataclass
class SubmissionDirectory:
    """Validated, immutable view of a submission directory complying with submission limits.

    Instances of this class are produced by :func:`validate_directory` after successfully
    enforcing competition limits and verifying directory structure. All paths stored in this
    dataclass are resolved absolute :class:`pathlib.Path` objects.
    """

    root_dir: Path
    """The absolute path to the root of the unpacked submission directory."""

    config_path: Path
    """The absolute path to the primary root YAML configuration file (e.g., agent.yaml)."""

    prompt_files: list[Path] = field(default_factory=list)
    """List of absolute paths to all discovered markdown (.md) and text (.txt) prompt files."""

    adapter_files: list[Path] = field(default_factory=list)
    """List of absolute paths to all discovered model weight and adapter files (e.g., .safetensors)."""

    yaml_files: list[Path] = field(default_factory=list)
    """List of absolute paths to all discovered YAML configuration files (.yaml, .yml)."""

    all_files: list[Path] = field(default_factory=list)
    """List of absolute paths to every regular file discovered within the entire directory tree."""


def validate_directory(
    submission_dir: str | Path,
    limits: SubmissionLimits | None = None,
) -> SubmissionDirectory:
    """Validate and catalogue a submission directory against configured limits.

    Performs a recursive scan of the submission directory to ensure compliance with
    competition constraints and structural rules. Specifically, it:
    1. Resolves the root directory and rejects any symlinks to prevent path traversal.
    2. Enforces maximum file count and cumulative file size limits.
    3. Verifies that all file extensions match allowed whitelists.
    4. Categorizes files into YAML configs, prompts, and model adapters.
    5. Locates and validates the unambiguous root agent configuration file.

    Args:
        submission_dir: Absolute or relative path to the unpacked submission directory.
        limits: Optional limits to enforce on file counts, sizes, and extensions.
            If ``None``, default :class:`SubmissionLimits` are applied.

    Returns:
        A :class:`SubmissionDirectory` instance containing catalogued absolute file paths.

    Raises:
        SubmissionValidationError: If the path is not a directory, contains symlinks,
            has files with disallowed extensions, or lacks an unambiguous root config.
        LimitExceededError: If total file count, cumulative byte size, or YAML file count
            exceeds configured limits.
    """
    limits = limits or SubmissionLimits()
    root = Path(submission_dir).resolve()

    if not root.is_dir():
        raise SubmissionValidationError(
            f"Submission path is not a directory: {submission_dir}"
        )

    set_active_limits_for_root(root, limits)

    # Collect all files (recursive) — reject symlinks
    all_files: list[Path] = []
    total_size = 0
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise SubmissionValidationError(
                f"Symlinks are not permitted: {p.relative_to(root)}"
            )
        if p.is_file():
            all_files.append(p)
            total_size += p.stat().st_size

    # --- File-level limits ---
    if len(all_files) > limits.max_file_count:
        raise LimitExceededError(
            f"Too many files: {len(all_files)} (max {limits.max_file_count})"
        )
    if total_size > limits.max_total_size_bytes:
        raise LimitExceededError(
            f"Total submission size {total_size:,} bytes exceeds "
            f"limit of {limits.max_total_size_bytes:,} bytes"
        )

    # --- Extension check ---
    for p in all_files:
        if p.suffix.lower() not in limits.allowed_file_extensions:
            raise SubmissionValidationError(
                f"File has disallowed extension '{p.suffix}': "
                f"{p.relative_to(root)}"
            )

    # --- Categorise files ---
    adapters_dir = root / "adapters"
    yaml_files = [p for p in all_files if p.suffix.lower() in {".yaml", ".yml"}]
    prompt_files = [p for p in all_files if p.suffix.lower() in {".md", ".txt"}]
    adapter_files = [
        p
        for p in all_files
        if p.suffix.lower() in limits.adapter_extensions
        and adapters_dir.is_dir()
        and p.is_relative_to(adapters_dir)
    ]

    if len(yaml_files) > limits.max_yaml_files:
        raise LimitExceededError(
            f"Too many YAML files: {len(yaml_files)} "
            f"(max {limits.max_yaml_files})"
        )

    for yf in yaml_files:
        yf_size = yf.stat().st_size
        if yf_size > limits.max_yaml_size_bytes:
            raise LimitExceededError(
                f"YAML file '{yf.relative_to(root)}' size {yf_size:,} bytes "
                f"exceeds limit of {limits.max_yaml_size_bytes:,} bytes"
            )

    for sf in all_files:
        if sf.name == "SKILL.md":
            skill_dir = sf.parent
            skill_size = sum(
                f.stat().st_size for f in skill_dir.rglob("*") if f.is_file()
            )
            if skill_size > limits.max_skill_size_bytes:
                raise LimitExceededError(
                    f"Skill directory '{skill_dir.relative_to(root)}' size {skill_size:,} bytes "
                    f"exceeds limit of {limits.max_skill_size_bytes:,} bytes"
                )

    # --- Find root config ---
    config_path = find_root_config(root, yaml_files)

    return SubmissionDirectory(
        root_dir=root,
        config_path=config_path,
        prompt_files=prompt_files,
        adapter_files=adapter_files,
        yaml_files=yaml_files,
        all_files=all_files,
    )


def find_root_config(root: Path, yaml_files: list[Path] | None = None) -> Path:
    """Locate the primary root agent configuration file within the submission.

    Scans the root level of the submission directory in the following priority order:
    1. ``agent.yaml`` at the root level of the submission directory.
    2. ``root_agent.yaml`` at the root level of the submission directory.
    3. If exactly one YAML file (``.yaml`` or ``.yml``) exists at the root level, use it.

    Args:
        root: Absolute path to the submission root directory.
        yaml_files: Optional list of absolute paths to discovered YAML files in the submission.

    Returns:
        Absolute :class:`pathlib.Path` to the resolved root configuration file.

    Raises:
        SubmissionValidationError: If no root YAML config is found, or if multiple
            ambiguous YAML files exist at the root level without a conventional name.
    """
    for name in _ROOT_CONFIG_NAMES:
        candidate = root / name
        if candidate.is_symlink():
            raise PathTraversalError(
                f"Root config path escapes submission directory via symlink: {candidate}"
            )
        if candidate.exists() and candidate.is_file():
            return validate_sandboxed_path(
                path=candidate,
                base_dir=root,
                must_exist=True,
                allow_symlinks=False,
                allowed_extensions={".yaml", ".yml"},
                error_prefix="Root config",
            )

    if yaml_files is None:
        yaml_files = []
        for p in root.iterdir():
            if p.suffix.lower() in {".yaml", ".yml"}:
                if p.is_symlink():
                    raise PathTraversalError(
                        f"Root config path escapes submission directory via symlink: {p}"
                    )
                if p.is_file():
                    yaml_files.append(p)

    # Fall back: single yaml file at root level
    root_level_yamls = [
        p for p in yaml_files if p.parent == root
    ]
    if len(root_level_yamls) == 1:
        return validate_sandboxed_path(
            path=root_level_yamls[0],
            base_dir=root,
            must_exist=True,
            allow_symlinks=False,
            allowed_extensions={".yaml", ".yml"},
            error_prefix="Root config",
        )

    if len(root_level_yamls) == 0:
        raise SubmissionValidationError(
            "No root YAML config found. Expected 'agent.yaml' or "
            "'root_agent.yaml' at the top level of the submission directory."
        )

    names = [p.name for p in root_level_yamls]
    raise SubmissionValidationError(
        f"Ambiguous root config: found {len(root_level_yamls)} YAML files "
        f"at the root level ({', '.join(names)}). Please name the root "
        f"config 'agent.yaml' or 'root_agent.yaml'."
    )


_find_root_config = find_root_config


# ---------------------------------------------------------------------------
# Adapter discovery
# ---------------------------------------------------------------------------


@dataclass
class AdapterInfo:
    """Metadata describing a discovered model weight or adapter file.

    Generated during adapter discovery by :func:`discover_adapters`. If adapter files
    in subdirectories share the same filename stem as other adapters, their ``name``
    attribute is disambiguated using their relative directory path within ``adapters/``.
    """

    name: str
    """Identifier name for the adapter. Typically the filename stem (e.g., ``'my_lora'``), 
    or a disambiguated relative path (e.g., ``'subfolder_my_lora'``) if collisions occur."""

    path: Path
    """Absolute path to the adapter file on disk."""

    format: str
    """File extension without the leading dot (e.g., ``'safetensors'``, ``'gguf'``)."""

    size_bytes: int
    """Size of the adapter file in bytes."""

    rank: int | None = None
    """Detected LoRA rank ``r`` from ``adapter_config.json`` or ``.safetensors`` metadata, if available."""


def infer_adapter_rank(path: str | Path) -> int | None:
    """Infer the LoRA rank ``r`` from a PEFT checkpoint directory or ``.safetensors`` file."""
    import contextlib
    import json
    import struct

    p = Path(path)
    config_candidates: list[Path] = []
    if p.is_dir():
        config_candidates.append(p / "adapter_config.json")
    else:
        config_candidates.append(p.parent / "adapter_config.json")

    for cfg_path in config_candidates:
        if cfg_path.is_file() and not cfg_path.is_symlink():
            with contextlib.suppress(Exception):
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    r_val = data.get("r")
                    if isinstance(r_val, int) and not isinstance(r_val, bool) and r_val > 0:
                        return r_val

    safetensor_candidates: list[Path] = []
    if p.is_file() and p.suffix.lower() == ".safetensors":
        safetensor_candidates.append(p)
    elif p.is_dir():
        cand = p / "adapter_model.safetensors"
        if cand.is_file():
            safetensor_candidates.append(cand)

    for sf_path in safetensor_candidates:
        if sf_path.is_symlink():
            continue
        with contextlib.suppress(Exception):
            with open(sf_path, "rb") as f:
                raw_len = f.read(8)
                if len(raw_len) < 8:
                    continue
                (header_len,) = struct.unpack("<Q", raw_len)
                if header_len <= 0 or header_len > 10_000_000:
                    continue
                header = json.loads(f.read(header_len).decode("utf-8"))
            if isinstance(header, dict):
                ranks: list[int] = []
                for key, meta in header.items():
                    if key == "__metadata__" or not isinstance(meta, dict):
                        continue
                    key_lower = key.lower()
                    if "lora_a" in key_lower or "lora_b" in key_lower:
                        shape = meta.get("shape")
                        if (
                            isinstance(shape, list)
                            and len(shape) == 2
                            and all(isinstance(d, int) and d > 0 for d in shape)
                        ):
                            ranks.append(min(shape[0], shape[1]))
                if ranks:
                    return max(ranks)

    return None


@dataclass
class AdapterManifest:
    """Manifest of all model adapter files discovered within a submission directory.

    Contains a mapping of disambiguated adapter names to :class:`AdapterInfo` objects,
    providing a helper method to bulk-register discovered adapters into a :class:`ModelRegistry`.
    """

    adapters: dict[str, AdapterInfo]
    """Dictionary mapping disambiguated adapter names to their metadata."""

    def register_all(
        self,
        model_registry: ModelRegistry,
        model_id_fn: Callable[[AdapterInfo], str],
    ) -> None:
        """Register all discovered adapters into a :class:`ModelRegistry`."""
        for name, info in self.adapters.items():
            model_id = model_id_fn(info)
            model_registry.register(f"adapter:{name}", model_id)


_NON_ADAPTER_WEIGHT_STEMS: frozenset[str] = frozenset(
    {
        "training_args",
        "optimizer",
        "scheduler",
        "scaler",
        "rng_state",
        "trainer_state",
    }
)


def discover_adapters(
    submission_dir: str | Path,
    adapter_extensions: frozenset[str] | None = None,
) -> AdapterManifest:
    """Discover and catalogue adapter and model weight files within a submission's adapters/ directory."""
    if adapter_extensions is None:
        adapter_extensions = SubmissionLimits().adapter_extensions

    root = Path(submission_dir).resolve()
    adapters_dir = root / "adapters"
    if adapters_dir.is_symlink():
        raise PathTraversalError(
            f"Adapter path escapes submission directory: {adapters_dir}"
        )
    if not adapters_dir.exists() or not adapters_dir.is_dir():
        return AdapterManifest(adapters={})

    adapters: dict[str, AdapterInfo] = {}

    all_adapter_files: list[Path] = []
    for p in sorted(adapters_dir.rglob("*")):
        if p.is_symlink():
            raise PathTraversalError(
                f"Adapter path escapes submission directory: {p}"
            )
        validate_sandboxed_path(
            path=p,
            base_dir=root,
            must_exist=True,
            allow_symlinks=False,
            error_prefix="Adapter",
        )
        if (
            p.is_file()
            and p.suffix.lower() in adapter_extensions
            and p.stem.lower() not in _NON_ADAPTER_WEIGHT_STEMS
        ):
            all_adapter_files.append(p)

    root_adapters = [p for p in all_adapter_files if p.parent == adapters_dir]
    sub_adapters = [p for p in all_adapter_files if p.parent != adapters_dir]

    for p in root_adapters:
        info = AdapterInfo(
            name=p.stem,
            path=p,
            format=p.suffix.lstrip(".").lower(),
            size_bytes=p.stat().st_size,
            rank=infer_adapter_rank(p),
        )
        if info.name in adapters:
            raise SubmissionValidationError(
                f"Adapter name collision at root level: {info.name!r}"
            )
        adapters[info.name] = info

    # Group files inside PEFT checkpoint directories into a single adapter entry
    peft_dir_groups: dict[Path, list[Path]] = {}
    standalone_sub_adapters: list[Path] = []
    for p in sub_adapters:
        is_peft_dir = (
            (p.parent / "adapter_config.json").is_file()
            or p.stem == "adapter_model"
            or p.stem.startswith("adapter_model-")
        )
        if is_peft_dir:
            peft_dir_groups.setdefault(p.parent, []).append(p)
        else:
            standalone_sub_adapters.append(p)

    sub_entries: list[tuple[Path, str, int, bool]] = []
    for peft_dir, files in peft_dir_groups.items():
        # Prefer .safetensors primary file if available
        primary = next(
            (f for f in files if f.name == "adapter_model.safetensors"),
            next(
                (f for f in files if f.suffix.lower() == ".safetensors"),
                files[0],
            ),
        )
        total_size = sum(f.stat().st_size for f in files)
        default_name = str(peft_dir.relative_to(adapters_dir)).replace("/", "_")
        sub_entries.append((primary, default_name, total_size, True))

    for p in standalone_sub_adapters:
        sub_entries.append((p, p.stem, p.stat().st_size, False))

    sub_name_counts = Counter(name for _, name, _, _ in sub_entries)

    for p, default_name, total_size, is_peft in sub_entries:
        info = AdapterInfo(
            name=default_name,
            path=p,
            format=p.suffix.lstrip(".").lower(),
            size_bytes=total_size,
            rank=infer_adapter_rank(p),
        )
        if info.name in adapters or sub_name_counts[default_name] > 1:
            if is_peft:
                rel = p.parent.relative_to(adapters_dir)
                info.name = str(rel).replace("/", "_")
            else:
                rel = p.relative_to(adapters_dir)
                info.name = str(rel.with_suffix("")).replace("/", "_")
        if info.name in adapters:
            raise SubmissionValidationError(
                f"Adapter name collision after disambiguation: "
                f"{info.name!r} (from {p.relative_to(root)})"
            )
        adapters[info.name] = info

    return AdapterManifest(adapters=adapters)


# ---------------------------------------------------------------------------
# Skill discovery
# ---------------------------------------------------------------------------


@dataclass
class SkillInfo:
    """Metadata describing a discovered ADK skill directory.

    Produced by :func:`discover_skills`. A skill directory is identified by the presence
    of a ``SKILL.md`` file. If multiple skill directories share the same folder name, their
    ``name`` attribute is disambiguated using their relative path from the submission root.
    """

    name: str
    """Identifier name for the skill. Typically the directory name (e.g., ``'sql-analyst'``),
    or a disambiguated relative path (e.g., ``'subfolder-sql-analyst'``) if collisions occur."""

    path: Path
    """Absolute path to the skill directory containing the ``SKILL.md`` file."""


@dataclass
class SkillManifest:
    """Manifest of all ADK skill directories discovered within a submission directory.

    Contains a mapping of disambiguated skill names to :class:`SkillInfo` objects,
    providing a helper method to bulk-register discovered skills into a :class:`SkillRegistry`.
    """

    skills: dict[str, SkillInfo]
    """Dictionary mapping disambiguated skill names to their metadata."""

    def register_all(
        self,
        skill_registry: SkillRegistry,
        skill_loader_fn: Callable[[SkillInfo], Any],
    ) -> None:
        """Register all discovered skills into a :class:`SkillRegistry`.

        Iterates through all discovered skills, invokes the loader function to instantiate
        the ADK Skill object, and registers it in the provided skill registry. If a skill
        name already exists in the registry, it will be silently overwritten.

        Args:
            skill_registry: The :class:`SkillRegistry` instance to populate.
            skill_loader_fn: A callable that takes a :class:`SkillInfo` instance and
                returns a loaded ADK Skill object (e.g., ``lambda info: load_skill_from_dir(info.path)``).

        Returns:
            None

        Example::

            skill_manifest.register_all(
                skills,
                lambda info: load_skill_from_dir(info.path),
            )
        """
        for name, info in self.skills.items():
            skill = skill_loader_fn(info)
            skill_registry.register(name, skill)


def discover_skills(
    submission_dir: str | Path,
    limits: SubmissionLimits | None = None,
) -> SkillManifest:
    """Discover and catalogue ADK skill directories within a submission directory.

    Scans the submission directory tree for any directory containing a ``SKILL.md`` file.
    Skill names are derived from their directory names; if multiple skill directories share
    the same folder name, they are disambiguated by prefixing their relative path from the
    submission root using hyphens (e.g., ``subfolder-skillname``).

    Args:
        submission_dir: Absolute or relative path to the unpacked submission directory.
        limits: Optional limits to enforce on maximum skill counts and skill byte sizes.
            If ``None``, default :class:`SubmissionLimits` are applied.

    Returns:
        A :class:`SkillManifest` containing metadata for all discovered and disambiguated skills.

    Raises:
        PathTraversalError: If any symlink or path escaping the submission root is encountered.
        LimitExceededError: If the total number of discovered skills exceeds ``limits.max_skills``
            or if a skill directory exceeds ``limits.max_skill_size_bytes``.
        SubmissionValidationError: If a skill name collision persists even after relative path disambiguation.
    """
    limits = limits or SubmissionLimits()
    root = Path(submission_dir).resolve()
    skills: dict[str, SkillInfo] = {}

    for entry in sorted(root.rglob("*")):
        if entry.is_symlink():
            raise PathTraversalError(
                f"Skill path escapes submission directory: {entry}"
            )

    raw_skills: list[tuple[Path, str]] = []
    for p in sorted(root.rglob("SKILL.md")):
        if p.is_symlink() or p.is_file():
            validate_sandboxed_path(
                path=p,
                base_dir=root,
                must_exist=True,
                allow_symlinks=False,
                error_prefix="Skill",
            )
            skill_dir = p.parent
            validate_sandboxed_path(
                path=skill_dir,
                base_dir=root,
                must_exist=True,
                allow_symlinks=False,
                error_prefix="Skill",
            )
            skill_size = 0
            for child in sorted(skill_dir.rglob("*")):
                if child.is_symlink():
                    raise PathTraversalError(
                        f"Skill path escapes submission directory: {child}"
                    )
                validate_sandboxed_path(
                    path=child,
                    base_dir=root,
                    must_exist=True,
                    allow_symlinks=False,
                    error_prefix="Skill",
                )
                if child.is_file():
                    skill_size += child.stat().st_size
            if skill_size > limits.max_skill_size_bytes:
                raise LimitExceededError(
                    f"Skill directory '{skill_dir.relative_to(root)}' size {skill_size:,} bytes "
                    f"exceeds limit of {limits.max_skill_size_bytes:,} bytes"
                )
            raw_skills.append((skill_dir, skill_dir.name.replace("_", "-")))

    from collections import Counter

    name_counts = Counter(name for _, name in raw_skills)
    for skill_dir, name in raw_skills:
        if name_counts[name] > 1 or name in skills:
            rel = skill_dir.relative_to(root)
            name = str(rel).replace("/", "-").replace("_", "-")
        if name in skills:
            raise SubmissionValidationError(
                f"Skill name collision after disambiguation: "
                f"{name!r} (from {skill_dir.relative_to(root)})"
            )
        skills[name] = SkillInfo(name=name, path=skill_dir)

    if len(skills) > limits.max_skills:
        raise LimitExceededError(
            f"Too many skills discovered: {len(skills)} (max {limits.max_skills})"
        )

    return SkillManifest(skills=skills)


def discover_declared_models(
    submission_dir: str | Path,
    normalize_fn: Callable[[str], str] | None = None,
) -> set[str]:
    """Discover all model identifiers explicitly declared across an agent submission.

    Traverses the root agent configuration, referenced sub-agents, and agent tools,
    as well as standalone agent YAML files in ``submission_dir``.

    Args:
        submission_dir: Absolute or relative path to the submission directory.
        normalize_fn: Optional callable to normalize discovered model strings.

    Returns:
        Set of unique model identifiers declared across the submission.

    Raises:
        SubmissionValidationError: If the submission directory is missing or lacks a root config.
        PathTraversalError: If a configuration path or symlink escapes ``submission_dir``.
    """
    from .context import CompilationContext
    from .registry import ToolRegistry
    from .yaml_loader import load_yaml

    root = Path(submission_dir).resolve()
    if not root.is_dir():
        raise SubmissionValidationError(f"Agent directory not found: {submission_dir}")

    root_config_path = find_root_config(root)

    declared_models: set[str] = set()
    visited_paths: set[Path] = set()
    norm = normalize_fn if normalize_fn is not None else (lambda s: s.strip())

    def _safe_load(path: Path) -> Any:
        validate_sandboxed_path(
            path=path,
            base_dir=root,
            must_exist=True,
            allow_symlinks=False,
            error_prefix="Agent config",
        )
        return load_yaml(path.resolve(), root)

    def _resolve_sub(base_dir: Path, rel: str) -> Path:
        from .paths import ensure_no_traversal_components

        try:
            ensure_no_traversal_components(rel, context="in Sub-agent")
        except PathTraversalError as err:
            raise PathTraversalError(f"Path traversal detected: {err}") from err
        raw_candidate = base_dir / rel
        if raw_candidate.is_symlink() or not raw_candidate.resolve().is_relative_to(root):
            raise PathTraversalError(
                f"Path traversal detected: {raw_candidate} escapes {root}"
            )
        ctx = CompilationContext(
            root_dir=root,
            tool_registry=ToolRegistry(),
            model_registry=ModelRegistry(),
            callback_registry=None,
            limits=SubmissionLimits(),
            generation_constraints=None,
            current_dir=base_dir,
        )
        resolved_candidate = raw_candidate.resolve()
        if resolved_candidate.exists():
            return ctx.resolve_sub_config_path(rel)

        raw_root_cand = root / rel
        if raw_root_cand.is_symlink() or not raw_root_cand.resolve().is_relative_to(root):
            raise PathTraversalError(
                f"Path traversal detected: {raw_root_cand} escapes {root}"
            )
        if raw_root_cand.resolve().exists():
            ctx.current_dir = root
            return ctx.resolve_sub_config_path(rel)
        return raw_root_cand.resolve()

    def _traverse(cfg_path: Path) -> None:
        if cfg_path.is_symlink() or not cfg_path.resolve().is_relative_to(root):
            raise PathTraversalError(
                f"Path traversal detected: {cfg_path} escapes {root}"
            )
        if cfg_path in visited_paths or not cfg_path.is_file():
            return
        visited_paths.add(cfg_path)

        try:
            raw = _safe_load(cfg_path)
        except (PathTraversalError, LimitExceededError):
            raise
        except Exception as err:
            if "traversal" in str(err).lower() or "escapes" in str(err).lower():
                raise PathTraversalError(
                    f"Path traversal detected: {cfg_path} escapes {root}"
                ) from err
            return

        if not isinstance(raw, dict):
            return

        model_val = raw.get("model")
        if isinstance(model_val, str) and model_val.strip():
            declared_models.add(norm(model_val))

        sub_agents = raw.get("sub_agents")
        if isinstance(sub_agents, list):
            for entry in sub_agents:
                if isinstance(entry, dict) and "config_path" in entry:
                    sub_cfg = _resolve_sub(cfg_path.parent, str(entry["config_path"]))
                    _traverse(sub_cfg)

        tools = raw.get("tools")
        if isinstance(tools, list):
            for tool_entry in tools:
                if isinstance(tool_entry, dict):
                    agent_tool = tool_entry.get("agent_tool") or tool_entry.get("agent")
                    if isinstance(agent_tool, dict) and "config_path" in agent_tool:
                        tool_cfg = _resolve_sub(
                            cfg_path.parent, str(agent_tool["config_path"])
                        )
                        _traverse(tool_cfg)

    _traverse(root_config_path)

    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise PathTraversalError(
                f"Path traversal detected via symlink in submission directory: {p}"
            )
        if (
            p.is_file()
            and p.suffix.lower() in (".yaml", ".yml")
            and p not in visited_paths
        ):
            try:
                content = _safe_load(p)
                if isinstance(content, dict):
                    m = content.get("model")
                    if isinstance(m, str) and m.strip():
                        if any(
                            k in content
                            for k in ("agent_class", "instruction", "tools", "sub_agents")
                        ):
                            declared_models.add(norm(m))
            except (PathTraversalError, LimitExceededError):
                raise
            except Exception as err:
                if "traversal" in str(err).lower() or "escapes" in str(err).lower():
                    raise PathTraversalError(
                        f"Path traversal detected: {p} escapes {root}"
                    ) from err
                continue

    return declared_models



