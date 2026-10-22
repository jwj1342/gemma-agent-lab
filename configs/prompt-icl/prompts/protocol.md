## Tool arguments and recovery

Call edit_file with three separate named arguments: filepath, old_string and
new_string. Copy old_string exactly from the file. Each string contains literal
source text; JSON escaping in a tool response describes the text representation.
Keep argument names outside string values and follow the provided tool schema.

After a missing-argument or malformed-call error, correct the argument structure.
If the same error occurs twice, switch to a short python3 heredoc through
run_command. Read the file, assert that the intended old fragment occurs exactly
once, apply the replacement, and write the file back. This also works for source
fragments containing complex quotes or regular-expression punctuation. Use
write_file for a whole-file replacement only after reading the complete current
file. Inspect git diff to verify the resulting source change.

A tool invocation error is feedback about the invocation. Spend the remaining
budget on a different valid action, a focused test, and submitting the patch.
