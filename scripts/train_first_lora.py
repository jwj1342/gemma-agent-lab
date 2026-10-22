"""Train a bounded action-level LoRA pilot and save complete resumable state.

Run as python -m scripts.train_first_lora within a GPU allocation.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time

from src.sft_data import encode_example, read_jsonl


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--graph-data", type=Path, required=True)
    parser.add_argument("--terminal-data", type=Path)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training/first-lora.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training requires a Slurm allocation on Trillium")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForImageTextToText, AutoTokenizer, CompressedTensorsConfig
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()
    config = json.loads(args.config.read_text())
    if args.output.exists() and not args.resume:
        raise ValueError("Training output already exists; use a new directory or explicit resume")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(8)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    sources = [("swe", args.data), ("graph", args.graph_data)]
    if args.terminal_data:
        sources.append(("terminal", args.terminal_data))
    if config.get("terminal_fraction", 0) and not args.terminal_data:
        raise ValueError("Terminal sampling requires a grounded terminal dataset")
    buckets = {s: {k: [] for k, _ in sources} for s in ("train", "validation")}
    rejected = Counter()
    inputs = {}
    for split in buckets:
        for kind, directory in sources:
            path = directory / f"{split}.jsonl"
            inputs[str(path.resolve())] = digest(path)
            for sample in read_jsonl(path):
                encoded = encode_example(sample, tokenizer, config["max_length"])
                if encoded is None:
                    rejected[kind + ": complete target/context cannot fit"] += 1
                else:
                    buckets[split][kind].append(encoded)
    if any(not samples for split in buckets.values() for samples in split.values()):
        raise ValueError("Both training and validation need SWE and graph actions")
    for kind, _ in sources:
        train_groups = {x["metadata"]["group"] for x in buckets["train"][kind]}
        val_groups = {x["metadata"]["group"] for x in buckets["validation"][kind]}
        if not train_groups.isdisjoint(val_groups):
            raise ValueError("Train/validation snapshot leakage")
    all_train = {x["metadata"]["group"] for b in buckets["train"].values() for x in b}
    all_validation = {x["metadata"]["group"] for b in buckets["validation"].values() for x in b}
    if not all_train.isdisjoint(all_validation):
        raise ValueError("Cross-source train/validation snapshot leakage")
    # These full paths are saved only in ignored local run artifacts.
    contract = {"config_sha256": digest(args.config), "inputs": inputs,
                "source_sha256": {str(p): digest(p) for p in (Path("src/sft_data.py"), Path("scripts/train_first_lora.py"))},
                "base_config_sha256": digest(args.base / "config.json"),
                "base_weights_sha256": digest(args.base / "model.safetensors")}
    if args.init_adapter:
        contract["initial_adapter_sha256"] = {name: digest(args.init_adapter / name)
                                              for name in ("adapter_model.safetensors", "adapter_config.json")}
    run_manifest = {**contract, "config": config, "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "working_tree_modified": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                    "modules": os.environ.get("LOADEDMODULES", ""), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                    "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "compressed-tensors", "accelerate")},
                    "accepted_actions": {s: {k: len(v) for k, v in b.items()} for s, b in buckets.items()},
                    "rejected": dict(rejected), "scope": "SFT action pilot; capability and repair gains require independent evaluation"}
    if not args.resume:
        (args.output / "run-manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    else:
        old = json.loads((args.output / "run-manifest.json").read_text())
        if any(old[k] != contract[k] for k in contract):
            raise ValueError("Resume input/configuration hashes differ")
    print("accepted actions", run_manifest["accepted_actions"], flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, local_files_only=True, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        quantization_config=CompressedTensorsConfig(dequantize=True), attn_implementation="sdpa")
    targets = [name for name, module in model.named_modules()
               if name.endswith(tuple(config["target_suffixes"])) and "vision" not in name]
    if not targets:
        raise ValueError("No compatible text attention projections")
    if args.resume:
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True, local_files_only=True)
    elif args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(r=config["rank"], lora_alpha=config["lora_alpha"],
                                                target_modules=targets, lora_dropout=0, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    step = 0
    if args.resume:
        # This state was written by this program; never load arbitrary third-party pickle checkpoints.
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        step = state["step"]
        (args.output / "resume-check.json").write_text(json.dumps({"checkpoint": str(args.resume), "restored_step": step,
                            "input_hashes_match": True, "optimizer_and_rng_restored": True}, indent=2) + "\n")
    initial_adapter = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}

    def loss_for(sample):
        ids = torch.tensor([sample["input_ids"]], device="cuda")
        labels = torch.tensor(sample["labels"], device="cuda")
        positions = torch.nonzero(labels != -100, as_tuple=True)[0]
        # Select only assistant predictions before the large vocabulary projection.
        prediction_positions = positions - 1
        if prediction_positions.min() < 0:
            raise ValueError("Assistant target has no preceding prompt token")
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       logits_to_keep=prediction_positions, use_cache=False)
        return F.cross_entropy(output.logits[0].float(), labels[positions]), len(positions)

    validation = {kind: sorted(values, key=lambda s: hashlib.sha256(json.dumps(s["metadata"], sort_keys=True).encode()).hexdigest())[:config["validation_actions"]]
                  for kind, values in buckets["validation"].items()}

    def evaluate():
        model.eval()
        scores = {}
        with torch.no_grad():
            for kind, samples in validation.items():
                total, tokens = 0.0, 0
                for sample in samples:
                    loss, count = loss_for(sample)
                    total += float(loss) * count
                    tokens += count
                scores[kind] = {"assistant_nll": total / tokens, "tokens": tokens, "actions": len(samples)}
        return scores

    if step == 0:
        (args.output / "validation-before.json").write_text(json.dumps(evaluate(), indent=2) + "\n")
    target_step = min(config["steps"], args.stop_after or config["steps"])
    graph_per_step = round(config["gradient_accumulation"] * config["graph_fraction"])
    terminal_per_step = round(config["gradient_accumulation"] * config.get("terminal_fraction", 0))
    if graph_per_step + terminal_per_step >= config["gradient_accumulation"]:
        raise ValueError("Sampling mix must retain SWE actions")
    log = (args.output / "metrics.jsonl").open("a")
    while step < target_step:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for microstep in range(config["gradient_accumulation"]):
            kind = ("graph" if microstep < graph_per_step else
                    "terminal" if microstep < graph_per_step + terminal_per_step else "swe")
            sample = random.choice(buckets["train"][kind])
            loss, count = loss_for(sample)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            (loss / config["gradient_accumulation"]).backward()
            losses.append({"kind": kind, "loss": float(loss.detach()), "tokens": count, "group": sample["metadata"]["group"]})
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
            raise ValueError("Non-finite or zero LoRA gradient")
        optimizer.step()
        step += 1
        metric = {"step": step, "loss": sum(x["loss"] for x in losses) / len(losses), "microsteps": losses,
                  "gradient_norm": float(gradient_norm), "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "elapsed_s": time.monotonic() - started}
        log.write(json.dumps(metric) + "\n")
        log.flush()
        print(json.dumps({k: v for k, v in metric.items() if k != "microsteps"}), flush=True)
        if step % config["checkpoint_every"] == 0 or step == target_step:
            checkpoint = args.output / f"checkpoint-{step:04d}"
            temporary = args.output / f".checkpoint-{step:04d}.partial"
            if checkpoint.exists() or temporary.exists():
                raise ValueError("Checkpoint path already exists")
            temporary.mkdir()
            model.save_pretrained(temporary, safe_serialization=True, save_embedding_layers=False)
            torch.save({"step": step, "optimizer": optimizer.state_dict(), "python_rng": random.getstate(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temporary / "training_state.pt")
            (temporary / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
            temporary.rename(checkpoint)
    log.close()
    if step == config["steps"]:
        after = evaluate()
        (args.output / "validation-after.json").write_text(json.dumps(after, indent=2) + "\n")
        adapter = args.output / "adapter"
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        tensors = load_file(str(adapter / "adapter_model.safetensors"))
        changed = sum(not torch.equal(v, initial_adapter[k]) for k, v in tensors.items())
        nonzero_b = sum(bool(v.abs().sum() > 0) for k, v in tensors.items() if "lora_B" in k)
        if changed == 0 or nonzero_b == 0:
            raise ValueError("Adapter did not learn a nonzero update")
        sample = validation["graph"][0]
        model.eval()
        with torch.no_grad():
            enabled, _ = loss_for(sample)
            with model.disable_adapter():
                disabled, _ = loss_for(sample)
        effect = abs(float(enabled) - float(disabled))
        if effect == 0:
            raise ValueError("Adapter has no measurable fixed-input effect")
        report = {"optimizer_steps": step, "trainable_parameters": sum(p.numel() for p in parameters), "target_modules": targets,
                  "adapter_tensors": len(tensors), "changed_since_phase_start": changed, "nonzero_B_tensors": nonzero_b,
                  "adapter_bytes": (adapter / "adapter_model.safetensors").stat().st_size,
                  "adapter_sha256": digest(adapter / "adapter_model.safetensors"), "fixed_input_nll_difference": effect,
                  "validation_after": after, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "scope": "Training/export check. Official quantized-vLLM reload and matched repair evaluation are separate gates."}
        (args.output / "training-result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "target_modules"}), flush=True)


if __name__ == "__main__":
    main()
