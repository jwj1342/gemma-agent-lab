"""Convert SWE-smith actions without inventing tool observations or success labels."""
import hashlib
import json
import copy
import re
from pathlib import PurePosixPath


def tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
TOOLS = [
    tool("run_command", "Run a shell command in the offline repository workspace.", {"command": STRING}, ["command"]),
    tool("read_file", "Read a workspace file with optional inclusive line bounds.",
         {"filepath": STRING, "start_line": INTEGER, "end_line": INTEGER}, ["filepath"]),
    tool("edit_file", "Replace an exact, unique text block in a workspace file.",
         {"filepath": STRING, "old_string": STRING, "new_string": STRING}, ["filepath", "old_string", "new_string"]),
    tool("write_file", "Write the complete content of a workspace file.", {"filepath": STRING, "content": STRING}, ["filepath", "content"]),
    tool("submit_patch", "Capture the current Git diff for evaluation.", {}, []),
    tool("get_status", "Inspect the current submission and remaining tool budget.", {}, []),
    tool("get_code_neighbors", "Find incoming and outgoing neighbors of an exact graph symbol ID.",
         {"node": STRING, "edge_type": STRING, "max_neighbors": INTEGER}, ["node"]),
    tool("get_code_subgraph", "Return the induced graph on the supplied exact symbol IDs.",
         {"nodes": {"type": "array", "items": STRING}}, ["nodes"]),
    tool("search_similar_code", "Retrieve similar nodes using a cached symbol or source-text embedding.",
         {"query": STRING, "k": INTEGER}, ["query"]),
]
SYSTEM = ("You repair Python repositories using the available tools. Inspect relevant code, make a focused "
          "change, run a regression test, and submit the patch. Use exact argument names. "
          "After a successful submission, return a brief final completion message. "
          "Graph symbol IDs and edge types are case-sensitive. Use graph observations to guide source "
          "inspection. An empty graph result warrants a repository search. "
          "Some demonstration observations preserve an upstream tool's text format.")


class UnsupportedAction(ValueError):
    pass


def is_setup_command(command):
    """Conservatively identify explicit dependency/network setup at shell boundaries."""
    return bool(re.search(r"(?:^|[\n;&|])\s*(?:sudo\s+)?(?:"
        r"(?:python[\d.]*\s+-m\s+pip|pip[\d.]*)\s+(?:install|download)\b|"
        r"conda\s+(?:install|create|update)\b|(?:apt-get|apt|yum|dnf)\s+(?:install|update)\b|"
        r"git\s+clone\b|(?:curl|wget)\s)", command))


def read_jsonl(path):
    """JSONL separates records with LF; Unicode separators inside strings remain data."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(p.get("text", "") for p in value if p.get("type") == "text")
    return ""


def workspace_text(value):
    return value.replace("/testbed", "/workspace")


def relative_path(path):
    if path.startswith("/testbed/"):
        path = path[len("/testbed/"):]
    elif path.startswith("/workspace/"):
        path = path[len("/workspace/"):]
    if path.startswith("/") or ".." in PurePosixPath(path).parts or path in ("", "."):
        raise UnsupportedAction("File action escapes the workspace or targets a directory")
    return path


def convert_call(call):
    function = call["function"]
    args = function.get("arguments") or {}
    if isinstance(args, str):
        args = json.loads(args)
    name = function["name"]
    if name == "bash":
        name, args = "run_command", {"command": workspace_text(args["command"])}
    elif name == "submit":
        name, args = "submit_patch", {}
    elif name == "str_replace_editor":
        command = args["command"]
        path = relative_path(args["path"])
        if command == "view":
            name, mapped = "read_file", {"filepath": path}
            if args.get("view_range"):
                start, end = args["view_range"]
                mapped["start_line"] = start
                if end != -1:
                    mapped["end_line"] = end
            args = mapped
        elif command == "str_replace":
            name, args = "edit_file", {"filepath": path, "old_string": workspace_text(args["old_str"]),
                                       "new_string": workspace_text(args["new_str"])}
        elif command == "create":
            name, args = "write_file", {"filepath": path, "content": workspace_text(args["file_text"])}
        else:
            raise UnsupportedAction("Unsupported editor command: " + command)
    else:
        raise UnsupportedAction("Unsupported tool: " + name)
    return {"id": call["id"], "type": "function", "function": {"name": name, "arguments": args}}


def convert_trajectory(row, reject_setup_actions=False):
    """Keep successful upstream trajectories; preserve raw observations with path normalization."""
    if row.get("resolved") is not True:
        raise UnsupportedAction("Upstream trajectory is not marked resolved")
    messages = [{"role": "system", "content": SYSTEM}]
    calls = {}
    originals = json.loads(row["messages"])
    for source_index, original in enumerate(originals):
        role = original["role"]
        if role == "system":
            continue
        if role == "assistant":
            converted = [convert_call(c) for c in original.get("tool_calls") or []]
            if reject_setup_actions and any(c["function"]["name"] == "run_command"
                    and is_setup_command(c["function"]["arguments"]["command"]) for c in converted):
                raise UnsupportedAction("Explicit upstream dependency/network setup action")
            if not converted:
                # A real final answer after an observed submission is part of the protocol.
                if (source_index == len(originals) - 1 and messages[-1]["role"] == "tool"
                        and messages[-1].get("name") == "submit_patch"
                        and text_content(original.get("content")).strip()):
                    messages.append({"role": "assistant", "content": text_content(original["content"])})
                    break
                continue
            calls.update({c["id"]: c["function"]["name"] for c in converted})
            messages.append({"role": "assistant", "content": "", "tool_calls": converted})
        elif role == "tool":
            ids = original.get("tool_call_ids") or [original.get("tool_call_id")]
            if len(ids) != 1 or ids[0] not in calls:
                raise UnsupportedAction("Observation does not map to one known tool call")
            messages.append({"role": "tool", "tool_call_id": ids[0], "name": calls[ids[0]],
                             "content": workspace_text(text_content(original.get("content")))})
        elif role == "user":
            messages.append({"role": "user", "content": workspace_text(text_content(original.get("content")))})
        else:
            raise UnsupportedAction("Unsupported message role: " + role)
    if not any(m["role"] == "assistant" for m in messages):
        raise UnsupportedAction("No trainable assistant actions")
    return messages


def group_key(instance_id):
    """SWE-smith IDs use repo.base_commit.mutation; share a split for the whole snapshot."""
    parts = instance_id.split(".")
    if len(parts) < 2:
        raise ValueError("Missing repository/base-commit provenance")
    return parts[0].replace("__", "/") + "." + parts[1][:8]


def split_for_group(group, seed=42):
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16)
    return "validation" if value % 5 == 0 else "train"


def action_examples(messages, metadata):
    for index, message in enumerate(messages):
        if message["role"] == "assistant" and (message.get("tool_calls") or message.get("content")):
            yield {"messages": messages[:index + 1], "tools": TOOLS,
                   "metadata": {**metadata, "action_index": index,
                                "target_kind": "tool" if message.get("tool_calls") else "terminal"}}


def encode_example(example, tokenizer, max_length, history_actions=3, excerpt_chars=4000):
    """Mask the prompt; retain complete targets and skip actions that cannot fit."""
    messages = copy.deepcopy(example["messages"])
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if len(assistant_indices) > history_actions:
        messages = messages[:2] + messages[assistant_indices[-history_actions]:]
    for message in messages[:-1]:
        content = message.get("content", "")
        limit = 6000 if message["role"] == "user" else excerpt_chars
        if limit and message["role"] in ("user", "tool") and len(content) > limit:
            message["content"] = content[:limit // 2] + "\n[Excerpt shortened for training]\n" + content[-limit // 2:]
    while True:
        prefix = tokenizer.apply_chat_template(messages[:-1], tools=example["tools"], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
        whole = tokenizer.apply_chat_template(messages, tools=example["tools"], tokenize=False,
                                              add_generation_prompt=False, enable_thinking=True)
        # Gemma emits this token to hand control to the tool executor. It is
        # an inference stop token and must remain in the assistant's targets.
        # The observation body is absent because this example ends at the call.
        if messages[-1].get("tool_calls") and not whole.endswith("<|tool_response>"):
            raise ValueError("Gemma tool target must include its tool-response stop token")
        if len(tokenizer.encode(whole, add_special_tokens=False)) <= max_length:
            break
        # Remove complete earlier action/observation turns while retaining the issue and latest evidence.
        assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        if len(assistant_indices) <= 2:
            return None
        del messages[assistant_indices[0]:assistant_indices[1]]
    if not whole.startswith(prefix):
        raise ValueError("Chat template changes the prefix before the target action")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    whole_ids = tokenizer.encode(whole, add_special_tokens=False)
    if whole_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Tokenizer boundary crosses the prompt/assistant loss boundary")
    target = whole_ids[len(prefix_ids):]
    if not target or len(target) >= max_length // 2:
        return None
    return {"input_ids": prefix_ids + target, "labels": [-100] * len(prefix_ids) + target,
            "attention_mask": [1] * (len(prefix_ids) + len(target)), "metadata": example["metadata"]}
