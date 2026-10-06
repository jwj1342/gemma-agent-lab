### Model Selection and LoRA Adapters ###

At present, we only support the [`gemma-4-31b-it-qat-w4a16-ct`](https://www.kaggle.com/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct) Gemma 4 model variant. You *must* choose this model for every agent and subagent.

You may, but are not required, to include one or more `.safetensors` format LoRA adapters with your submission. While you are restricted to a single base model, you may use different adapters for each agent in your submission:
1. Place PEFT LoRA directories (containing `adapter_config.json` and `adapter_model.safetensors`) inside
`adapters/<adapter_name>/`.
2. Reference `adapter: <adapter_name>` on any `LlmAgent` in `agent.yaml` or `sub_agents/*.yaml` (for example,
`adapter: main_lora` on the root coder agent and `adapter: tool_lora` on a read-only analyzer `AgentTool`).

### Tool, Prompt, and Skill Rules ###
* **`!include` Directives**: Resolves file paths relative to the directory of the file containing the tag. For instance, in `agent.yaml`, `!include prompts/system.md` loads `prompts/system.md`.
* **Sandboxing**: Path traversal outside the submission root (e.g., via `../` or symlinks) is not allowed.
* **Allowed Tools**: Your agent can only request tools provided by the competition harness or custom subagents defined via `agent_tool`.
* **Skill Structure & Sandboxing**: Each skill must be a directory containing a `SKILL.md` manifest with YAML frontmatter (`name: <skill-name>`). Scripts executed via `run_skill_script` run securely inside the competition's persistent Docker container, sharing filesystem access with `run_command` and debiting execution time against your central budget. Agents can inspect domain knowledge files using `load_skill_resource`.

The predefined tools available to your agent are:
1. `run_command(command: str) -> str` - Executes a shell command in `/bin/bash -c` inside `/workspace`.
2. `submit_patch() -> str` - Stages untracked file intents (`git add -N .`) and captures `git diff HEAD` from `/workspace`.
3. `get_status() -> str` - Returns live budget consumption and patch status.
4. `read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str` - Reads a file from `/workspace` with 1-indexed inclusive line slicing.
5. `edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str` - Replaces `old_string` with `new_string` in an existing non-empty file inside `/workspace`.
6. `write_file(filepath: str, content: str) -> str` - Creates or overwrites a file at `/workspace/<filepath>`, automatically creating parent directories (`mkdir -p`).
7. `get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str` - Finds incoming and outgoing neighbors of a symbol (`node`) in the repository call/dependency graph.
8. `search_similar_code(query: str, k: int = 10) -> str` - Finds top-`k` graph nodes with highest cosine similarity to `query` in the pre-computed embeddings.
9. `get_code_subgraph(nodes: list[str]) -> str` - Extracts the induced subgraph (all nodes and interconnecting edges) for a list of symbols.

See the `HARNESS_README.md` file in the dataset for detailed information about the submission format and execution environment.