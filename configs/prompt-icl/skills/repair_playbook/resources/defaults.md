# Missing values and defaults

Check how the API represents a missing value: None, a sentinel, a missing dictionary
key, or another documented state. Compare that with values the caller may legitimately
pass, including 0, False, an empty string and an empty collection. A truthiness check
can collapse those cases. Choose an explicit predicate only when the contract calls
for it. Check both the missing case and the preserved explicit values. Inspect one
caller to catch accidental changes in coercion or validation order.
