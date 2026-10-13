## Tool arguments and recovery

Call edit_file with three separate named arguments: filepath, old_string and
new_string. Copy old_string exactly from the file. Each string contains literal
source text; JSON escaping in a tool response describes the text representation.
Keep argument names outside string values and follow the provided tool schema.

If edit_file returns a missing-argument or malformed-call error, do not resend it:
apply the change through this heredoc form instead —

run_command("cat <<'EOF' > fix.py\n<python script: read the target file; build the
new text by a small replace inside the old text; assert the reported behavior now
works and print PASS; write the file back>\nEOF\npython3 fix.py && rm fix.py")

— then read the changed region and inspect git diff to verify the source change.
This also works for source fragments containing complex quotes or
regular-expression punctuation. Use write_file for a whole-file replacement only
after reading the complete current file.

A tool invocation error is feedback about the invocation. Spend the remaining
budget on a different valid action, a focused test, and submitting the patch.
