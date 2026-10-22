---
name: repair-checkpoint
description: Inspect the current Git patch after editing and before submission. Detect an empty patch, protected test/config edits, Python syntax errors, merge-conflict markers, and debugger residue without modifying the repository.
---

# Repair checkpoint

Run this skill once after the targeted test and immediately before submission. It is a read-only,
standard-library-only inspection of the repository mounted at `/workspace`.

Call `run_skill_script` with:

- `skill_name`: `repair-checkpoint`
- `file_path`: `scripts/checkpoint.py`
- `args`: `{}`

The script emits one compact JSON object and returns successfully for both logical outcomes:

- `status: "ready"` means the patch is non-empty and no deterministic blocker was found. Submit it.
- `status: "repair"` means `reasons` names a blocker. Fix only that blocker, rerun the targeted test,
  and then run the checkpoint once more.

Treat the result as a structural guard, not proof that the issue is semantically resolved. The script
does not run tests, execute repository code, stage files, edit files, or inspect hidden tests.
