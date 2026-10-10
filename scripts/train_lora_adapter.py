"""Train and export the competition LoRA adapter for the coder agent.

Two modes. The default is a CPU-only, offline dry run that prints the resolved plan and exits: it
needs neither the base weights nor a GPU, so it can validate the wiring anywhere. ``--train``
requires the base model weights, a CUDA device, and a training environment with ``torch``,
``transformers``, ``peft``, ``trl``, and ``datasets`` -- these are deliberately not part of
``configs/ci.requirements.txt``.

Training is not a claim. Nothing in this repository has trained an adapter yet, so treat any number
this script prints in dry-run mode as an estimate, and report a real comparison before promoting an
adapter into a submission. Keep the evaluation environment (official wheelhouse, patched vLLM,
ADK) separate from the training environment (see docs/training-plan.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / 'configs/experiments/lora-trajectory-v1'
TARGET_EXPANSION = ['q_proj', 'o_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj']


def load_adapter_config(experiment_dir: Path, adapter_name: str) -> dict:
    path = experiment_dir / 'adapters' / adapter_name / 'adapter_config.json'
    if not path.is_file():
        raise SystemExit(f'missing adapter config: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def resolve_targets(model, target_modules: list[str]) -> list[str]:
    """Return target modules that exist as nn.Linear layers, and fail if any requested name is absent."""
    import torch.nn as nn

    present = {
        name.rsplit('.', 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    missing = sorted(set(target_modules) - present)
    if missing:
        raise SystemExit(
            f'{missing} are not Linear modules on the loaded base model; '
            'widen adapter_config.json only after confirming the real module names'
        )
    return list(target_modules)


def dry_run(experiment_dir: Path, adapter_name: str, base_model: str | None, targets: list[str]) -> dict:
    config = load_adapter_config(experiment_dir, adapter_name)
    configured = config.get('target_modules')
    configured = [configured] if isinstance(configured, str) else list(configured or [])
    return {
        'mode': 'dry-run',
        'experiment_dir': str(experiment_dir),
        'adapter_name': adapter_name,
        'base_model': base_model or 'not supplied; pass --base-model for a real run',
        'rank': config.get('r'),
        'lora_alpha': config.get('lora_alpha'),
        'target_modules_configured': configured,
        'target_modules_requested': targets,
        'targets_beyond_config': sorted(set(targets) - set(configured)),
        'note': 'Dry run only: no model was loaded, no weights were written, and no adapter exists yet.',
    }


def train(args: argparse.Namespace) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit('--train requires a CUDA device; none is visible')
    if args.base_model is None or not Path(args.base_model).exists():
        raise SystemExit('--base-model must point at the local competition base weights')
    if args.dataset is None or not Path(args.dataset).exists():
        raise SystemExit('--dataset must point at a JSONL file with prompt and completion fields')

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if args.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={'': 0},
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    targets = resolve_targets(model, args.target_modules)
    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        task_type='CAUSAL_LM',
        target_modules=targets,
    ))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    dataset = load_dataset('json', data_files=str(args.dataset), split='train')
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(args.output_dir),
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            logging_steps=1,
            save_steps=args.save_steps,
            save_total_limit=2,
            bf16=True,
            report_to=[],
            seed=args.seed,
            # Overfit the completion, not the prompt: without this the loss falls on instruction
            # tokens too, which is what several public notebooks got wrong.
            completion_only_loss=True,
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    return {
        'mode': 'train',
        'output_dir': str(args.output_dir),
        'target_modules': targets,
        'max_steps': args.max_steps,
        'dataset': str(args.dataset),
        'note': 'Adapter exported. Validate with scripts/check_lora_adapter.py, then run a paired '
                'evaluation with valid empty and gold controls before any submission claims benefit.',
    }


def self_check(experiment_dir: Path, adapter_name: str, work_dir: Path,
               override_targets: list[str] | None = None) -> dict:
    """Exercise the whole export path on a tiny local model, with no GPU, download, or base weights.

    This catches the failure mode that costs a GPU session: a wrong PEFT or harness keyword, a target
    module that resolves differently than expected, or an export layout the discovery code cannot see.
    The real base model is still required for a real run; nothing here proves training quality.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    config = load_adapter_config(experiment_dir, adapter_name)
    targets = override_targets or config.get('target_modules')
    targets = [targets] if isinstance(targets, str) else list(targets or [])
    rank = int(config.get('r'))
    hidden, layers = 64, 2

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
            self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
            self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
            self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)
            self.gate_proj = torch.nn.Linear(hidden, hidden, bias=False)
            self.up_proj = torch.nn.Linear(hidden, hidden, bias=False)
            self.down_proj = torch.nn.Linear(hidden, hidden, bias=False)

        def forward(self, x):
            return self.down_proj(self.up_proj(self.o_proj(self.v_proj(self.k_proj(self.q_proj(x))))))

    class ModelConfig(dict):
        """Self-check stand-in for a transformers config: PEFT reads attributes and tests membership."""

        def __init__(self, **values: Any) -> None:
            super().__init__(values)
            self.__dict__ = self

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # PeftModel reads base_model.config.model_type and .use_cache, and looks up keys such as
            # "_name_or_path" while saving; a real transformers config behaves the same way.
            self.config = ModelConfig(model_type='tiny', use_cache=False)
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList([Block() for _ in range(layers)])
            self.head = torch.nn.Linear(hidden, hidden, bias=False)

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            # Mirror a causal-LM signature: PeftModel forwards these keyword arguments by name.
            x = input_ids if input_ids is not None else torch.randn(2, 4, hidden)
            if x.dim() == 2:
                x = x.unsqueeze(-1).expand(-1, -1, hidden)
            for block in self.model.language_model.layers:
                x = block(x) + x
            return self.head(x)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            # PeftModel for CAUSAL_LM reads this from the wrapped model; every real transformers
            # causal LM provides it, so the stand-in must too.
            return {}

    torch.manual_seed(0)
    base = Tiny()
    # Mirror resolve_targets: a target the base model does not expose must fail loudly here rather
    # than being silently ignored (PEFT warns and wraps nothing when a target is absent).
    available = {name.rsplit('.', 1)[-1] for name, module in base.named_modules()
                 if isinstance(module, torch.nn.Linear)}
    absent = sorted(set(targets) - available)
    if absent:
        raise SystemExit(f'self-check: target modules absent from the base model: {absent}')
    model = get_peft_model(base, LoraConfig(
        r=rank, lora_alpha=int(config.get('lora_alpha')), lora_dropout=float(config.get('lora_dropout', 0.0)),
        bias=str(config.get('bias', 'none')), task_type='CAUSAL_LM', target_modules=targets))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise SystemExit('self-check: no trainable LoRA parameters were created')
    # Verify the configured targets really got wrapped. PEFT parameter names look like
    # ...layers.0.q_proj.lora_A.default.weight, so match the module name as a path segment.
    unresolved = [target for target in targets
                  if not any(f'.{target}.' in f'.{name}.' for name in trainable)]
    if unresolved:
        raise SystemExit(f'self-check: configured target modules were not wrapped: {unresolved}')
    # Export into the exact layout the competition expects: adapters/<name>/adapter_model.safetensors,
    # which is how adk_submission.discovery groups a PEFT directory.
    export_dir = work_dir / 'adapters' / adapter_name
    export_dir.mkdir(parents=True, exist_ok=True)

    # A B=0 adapter makes the reload comparison meaningful: nonzero B with zero A would also match
    # while proving nothing about the exported weights.
    zeroed = 0
    for name, parameter in model.named_parameters():
        if 'lora_B' in name:
            with torch.no_grad():
                parameter.zero_()
            zeroed += 1
    loss = model(input_ids=torch.randn(2, 4, hidden)).pow(2).mean()
    loss.backward()
    gradients_present = any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for _, parameter in model.named_parameters() if parameter.requires_grad)
    if not gradients_present:
        raise SystemExit('self-check: LoRA parameters received no gradient; training would not learn')

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if 'lora_B' in name}
    model.save_pretrained(str(export_dir))
    (export_dir / 'adapter_config.json').write_text(
        json.dumps(config, indent=2) + '\n', encoding='utf-8')
    weights = export_dir / 'adapter_model.safetensors'

    fresh = PeftModel.from_pretrained(Tiny(), str(export_dir))
    after = {name: parameter.detach().clone() for name, parameter in fresh.named_parameters()
             if 'lora_B' in name}
    reload_matches = set(before) == set(after) and all(
        torch.equal(before[name], after[name]) for name in before)
    discovered = sorted(path.parent.name for path in work_dir.glob('adapters/*/adapter_model.safetensors'))
    return {
        'mode': 'self-check',
        'work_dir': str(work_dir),
        'target_modules_resolved': targets,
        'target_modules_verified_wrapped': not unresolved,
        'trainable_parameter_tensors': len(trainable),
        'trainable_parameters': sum(parameter.numel() for _, parameter in model.named_parameters()
                                    if parameter.requires_grad),
        'forward_backward_ok': gradients_present,
        'weights_filename_matches_discovery': weights.name == 'adapter_model.safetensors',
        'weights_written': weights.is_file(),
        'reload_matches_export': reload_matches,
        'lora_b_tensors_zeroed_for_roundtrip': zeroed,
        'adapter_names_discoverable': discovered,
        'note': 'PEFT mechanics and export layout only. The real base model is still required, and no '
                'claim about accuracy follows from this check.',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment-dir', type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument('--adapter-name', default='main_lora')
    parser.add_argument('--base-model', default=None,
                        help='Local path to the competition base weights')
    parser.add_argument('--dataset', type=Path, default=None,
                        help='JSONL with prompt and completion fields, built from verified trajectories')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/lora-trajectory-v1/adapter')
    parser.add_argument('--target-modules', nargs='+', default=None,
                        help=f'Defaults to adapter_config.json; canonical expansion is {TARGET_EXPANSION}')
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--lora-alpha', type=int, default=None)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument('--bias', default='none')
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--save-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load-in-4bit', action='store_true', default=True)
    parser.add_argument('--no-load-in-4bit', dest='load_in_4bit', action='store_false')
    parser.add_argument('--train', action='store_true',
                        help='Run real training; requires base weights, CUDA, and training dependencies')
    parser.add_argument('--self-check', action='store_true',
                        help='Exercise PEFT wiring and export layout on a tiny CPU model; needs torch and peft only')
    parser.add_argument('--self-check-dir', type=Path, default=ROOT / 'runs/lora-trajectory-v1-self-check',
                        help='Must be a writable location; the harness layout adapters/<name>/ is created here')
    args = parser.parse_args()
    args.experiment_dir = args.experiment_dir.resolve()

    config = load_adapter_config(args.experiment_dir, args.adapter_name)
    configured = config.get('target_modules')
    configured = [configured] if isinstance(configured, str) else list(configured or [])
    if args.target_modules is None:
        args.target_modules = configured
    if args.rank is None:
        args.rank = int(config.get('r'))
    if args.lora_alpha is None:
        args.lora_alpha = int(config.get('lora_alpha'))

    if args.self_check:
        print(json.dumps(self_check(args.experiment_dir, args.adapter_name,
                                    args.self_check_dir.resolve(),
                                    list(args.target_modules or [])), indent=2))
        return 0
    if not args.train:
        print(json.dumps(dry_run(args.experiment_dir, args.adapter_name, args.base_model,
                                 args.target_modules), indent=2))
        print('Dry run complete; pass --train with base weights and a GPU to produce adapter weights.',
              file=sys.stderr)
        return 0
    print(json.dumps(train(args), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
