## Small examples of transferable decisions

These are illustrative toy examples, not facts about the current repository. Use
their decision patterns only when the current source and issue support them.

**A default value replaces a valid zero.** Suppose source contains
`return value or default`, while the contract says only None selects the default.
Inspect callers to confirm that 0, False and the empty string are valid values.
An exact edit would be:

    edit_file(filepath="pkg/options.py",
              old_string="return value or default",
              new_string="return default if value is None else value")

A focused check should distinguish None from 0, False, an empty string, and an
ordinary value. Adapt its import and function signature to the actual repository.
The decision follows the contract; the toy file path is not a search target.

**A graph query finds no neighbors.** Suppose get_code_neighbors for a symbol
returns an empty list. Search for that symbol in source, read the relevant region,
and check its caller directly. An empty graph does not establish that the symbol
or dependency is absent. Once source evidence explains the bug, proceed to an edit
instead of repeating the same graph query.

**A submitted patch completes the action sequence.** After a relevant check passes,
review the diff and call submit_patch. If the response reports the expected nonempty
patch, end with a concise summary. A second identical submit call adds no evidence.
