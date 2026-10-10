# Prompt, context learning, and test-time adaptation

This workstream aims to improve repair success per unit of inference time without
updating model weights. It complements the LoRA workstream and uses the same frozen
benchmark and version 29 environment. Research findings below are author-reported;
they have not been reproduced with the competition's Gemma model.

## What can adapt?

| Method | Changed object | Gradient updates | Main cost |
| --- | --- | --- | --- |
| In-context learning (ICL) | Examples and evidence in the current prompt | None | Longer model input |
| Retrieval and playbooks | A small selected set of useful examples or strategies | None | Retrieval and context tokens |
| Execution-grounded reflection | Current hypothesis and subsequent actions | None | Tests and additional model turns |
| Offline prompt optimization | Instructions selected using training/development feedback | None | Preparation rollouts |
| Test-time search | Candidate patches or exploration branches | None | Extra inference and validation |
| Test-time training (TTT) | Model parameters or adapters for a test input | Required | Forward/backward passes, data generation and serving changes |

Fewer training examples do not necessarily mean lower total cost. Per-task TTT can
be more expensive than preparing one reusable LoRA, especially when synthetic data
generation and repeated adaptation are included. We will measure preparation cost
and deployment cost separately.

## Primary-source reading map

| Work | Finding or mechanism | Relevance and limitation |
| --- | --- | --- |
| [Agentless](https://arxiv.org/abs/2407.01489), 2024 | Separates localization, repair and validation in a simple SWE pipeline. | Supports a clear workflow before adding complex orchestration. Its models, benchmark and scaffold differ from ours. |
| [Reflexion](https://arxiv.org/abs/2303.11366), 2023 | Uses textual feedback and episodic memory to improve subsequent attempts without changing weights. | Ground each revision in actual execution output. HumanEval results do not establish repository-level repair gains. |
| [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952), 2025 | Curates reusable strategies at inference time. | Use compact, verified lessons. The paper's cross-query memory assumes persistence that our competition submission may not have. |
| [Agentic Context Engineering](https://arxiv.org/abs/2510.04618), 2025 | Generates, reflects on and incrementally curates structured playbooks. | Preserve useful evidence while controlling context growth. Its reported agent/domain results are not SWE results for Gemma. |
| [GEPA](https://arxiv.org/abs/2507.19457), 2025 | Evolves prompts using execution traces, reflective proposals and candidate selection. | A promising offline prompt search method; still requires rollout budget. The reported sample-efficiency comparison with RL is not a universal GPU-cost guarantee. |
| [SWE-Search](https://arxiv.org/abs/2410.20285), 2024 | Uses tree search and value/discriminator feedback for repository repair. | Consider bounded branching only after a cheap single path works. Multiple agents and noisy judgments can consume the shared deadline. |
| [SWE-Replay](https://arxiv.org/abs/2601.22129), 2026 | Reuses selected trajectory prefixes and restores environments before branching. | Suggests reusing localization work. A prompt that asks for a second attempt does not implement its replay/state-restoration algorithm. |
| [SWE-ContextBench](https://arxiv.org/abs/2602.08316), 2026 | Studies experience reuse in related coding tasks; useful summaries help while poor retrieval can hurt. | Retrieve a small relevant example set and test a no-example control. Protect evaluation tasks from the memory bank. |
| [Thinking Longer, Not Larger](https://arxiv.org/abs/2503.23803), 2025 | Combines trajectory training with development-process search and verification. | Distinguish its trained reasoning component from inference-only search; the full method is not training-free. |
| [TTT on Nearest Neighbors](https://arxiv.org/abs/2305.18466), 2023 | Fine-tunes on retrieved neighboring text, including experiments with a small number of updates. | Evidence concerns language modeling; successful bug repair and compatibility with the competition are separate questions. |
| [Self-Improving LLM Agents at Test-Time](https://arxiv.org/abs/2510.07841), 2025 | Detects uncertain cases, generates related examples and fine-tunes on them. | Useful inspiration for selective data generation. It still changes weights and requires trustworthy self-generated supervision. |
| [TTRL](https://arxiv.org/abs/2504.16084), 2025 | Performs RL updates using estimated rewards on unlabeled reasoning tasks. | Multiple plausible patches need execution evidence; majority agreement is not a reliable repair label. This is not our initial low-cost route. |

## Competition fit

The [official model and harness rules](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/model-selection-budget-and-harness-rules)
allow prompts, registered tools, sandboxed skills and packaged PEFT adapters. Every
agent and subagent must use `gemma-4-31b-it-qat-w4a16-ct`. The
[evaluation rules](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/evaluation)
give the agent a shared 12-hour patch-generation budget including setup.

Static examples, packaged read-only playbooks, task-local reflection and bounded
tool-driven verification fit those interfaces. Version 29 exposes skill resources
through the official skill toolset; it does not require adding an arbitrary new
Python function to the registered tool list.

We have not found a supported submission interface for running backward passes on
the served model, changing its live weights, or dynamically registering a newly
trained adapter during a task. This is an interface finding, not an assertion that
the rules explicitly ban every possible TTT method. Per-task gradient adaptation
requires organizer clarification and a demonstrated supported runtime path before
implementation. The initial route uses zero parameter updates.

The organizer's [LoRA-only clarification](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743383)
supports this distinction: the system is configured to accept LoRA artifacts, and
other submitted mechanisms can fail at the runtime interface even when the broad
idea is not forbidden in principle. The [custom-tools clarification](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743573)
directs custom scripts to sandboxed skills. These threads were refreshed on October
9, 2026; neither establishes a live gradient-update interface for our agent.

Keep memory within the current task. For cross-task experience, prepare and freeze
a playbook using allowed training data before evaluation. Do not assume evaluation
order, writable submission files, a persistent global service, network access, or
access to hidden grading tests. Generated checks belong in ephemeral shell commands;
the final patch contains the intended implementation changes.

## Implemented first ablations

| Variant | Intervention | Weight updates |
| --- | --- | --- |
| `workflow` | A concise locate–edit–verify–submit policy with source-grounded graph navigation | 0 |
| `protocol` | The workflow plus explicit editor arguments and recovery after repeated invocation errors | 0 |
| `icl` | The workflow plus three short team-authored decision examples | 0 |
| `reflect` | ICL plus failure-driven revision, task-local evidence summaries and an optional short playbook | 0 |

These are small research-inspired candidates. They do not reproduce GEPA, ACE or
SWE-Replay. The original base model, analyzer configuration, generation settings and
per-task limits remain the same across these candidates. The `reflect` stage adds
a skill toolset as part of its declared intervention. Compare `workflow` with the
original baseline separately, because its instructions have been rewritten.

The `protocol` variant was added after the exposed smoke task revealed repeated
missing-argument errors: 20 editor calls for workflow and 32 for ICL. It contains
general invocation guidance and no smoke-task fix. Treat this observation as a
debugging finding; compare the candidate on the frozen fast cohort before claiming
effectiveness. The model's native Gemma tool-call serialization differs from JSON;
the guidance focuses on argument semantics and allows a shell-based edit fallback.

The toy examples cover a falsey-value/default distinction, graph-query fallback,
and final submission. They contain no public benchmark solution or gold patch.
Their role is to demonstrate decisions; the current repository supplies the facts.

Build a variant from a normal clone, with no credentials or weights:

```bash
python scripts/build_prompt_candidate.py --variant reflect \
  --output runs/my-prompt-candidate/agent
python scripts/build_submission.py --agent-dir runs/my-prompt-candidate/agent \
  --output runs/my-prompt-candidate/submission.zip
```

Check the packaging CLI's `--help` if using another branch. The checked-in default
package is [configs/prompt-icl/](../configs/prompt-icl/README.md), an experimental
reflection candidate rather than a promoted submission policy. Use the existing
[Kaggle runtime workflow](kaggle-runtime.md) for team members without cluster access.

## Evaluation sequence and expansion

The first matched version 29 fast comparison is complete: baseline 2/11, workflow
2/11, ICL 3/11, reflection 2/11 and protocol 2/11, with 11/12 control coverage.
ICL gained two tasks and regressed on one; reflection/playbook retrieval did not
establish a benefit. The completed development expansion reversed the initial
ICL signal: baseline resolved 16/31 eligible tasks and ICL 11/31. On the registered
additional subset, the counts were 13/20 and 8/20. ICL took 15.4% more inference
time and had 92 missing-argument feedback events versus 53 for baseline. Keep this
package as a negative experiment and retain the baseline for the first LoRA
comparison. See the [execution report](experiments/2026-10-09-prompt-context.md)
for coverage, paired outcomes, clean completion, costs and trace diagnostics.
These are single-attempt public comparisons; the holdout remains sealed. Retrieval,
offline prompt search and the later expansion ideas below remain future work.

1. Compile each package with the version 29 official compiler and check that
   preparation from a clone works without local model/data paths or credentials.
2. Run valid empty/gold controls and a smoke task. A smoke pass verifies integration,
   not effectiveness. Freeze prompt/agent/task/archive hashes before inference.
3. Compare baseline, workflow and ICL on the same fast cohort and resource budget.
   Add reflection only if its extra calls justify their cost. Report task coverage,
   paired repaired/regressed tasks, wall time, tool calls, invalid calls, repeated
   submissions, context usage and timeout rate. Preserve raw traces.
4. Validate a promising candidate on the development cohort. Keep the holdout sealed
   until a milestone choice. A small public cohort can be noisy and overfit easily.
5. If hand-built examples help, curate compact successful-action summaries from the
   allowed training split. Record source, license, task group and actual verification.
   Reject holdout/development snapshots and test zero, one and two retrieved examples.
6. If prompt sensitivity is large, implement a bounded offline GEPA-style search:
   use a small training batch to propose revisions, use a distinct selection panel,
   freeze the selected text and report all rollout costs. This search is planned,
   not implemented by the candidate builder.
7. Consider a second patch hypothesis only for unresolved cases with enough remaining
   budget. Compare at equal total inference time; more samples alone are not a fair
   comparison. True trajectory replay additionally needs tested environment restoration.

Use the LoRA and prompt routes as separate controls before testing their combination.
The common lead owns split discipline and final integration; the prompt workstream
owns examples, failure analysis and inference efficiency.
