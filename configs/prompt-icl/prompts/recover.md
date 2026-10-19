## Tool arguments and recovery

Call edit_file with three separate named arguments: filepath, old_string and
new_string. Copy old_string exactly from the file. Each string contains literal
source text; JSON escaping in a tool response describes the text representation.
Keep argument names outside string values and follow the provided tool schema.

If edit_file returns a missing-argument or malformed-call error, do not resend it:
apply the change through an inline Python heredoc instead. Follow this order
exactly:

1. Validate the replacement: read the target file and assert that the old text
   occurs exactly once.
2. Write the source: build the new text by a small replace inside the old text,
   then write the file back.
3. Verify the actual implementation: import the changed module and assert the
   reported behavior, printing PASS.

run_command("PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'\n<python script:
implement steps 1-3 in this order>\nPY")

— no helper file is written, so a failed check cannot leave scratch files in the
patch. Then read the changed region and inspect git diff to verify the source
change. This also works for source fragments containing complex quotes or
regular-expression punctuation. Use write_file for a whole-file replacement only
after reading the complete current file.

A tool invocation error is feedback about the invocation. Spend the remaining
budget on a different valid action, a focused test, and submitting the patch.
