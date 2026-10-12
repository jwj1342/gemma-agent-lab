# Graph evidence

Start with a symbol grounded in source or the issue. Query its neighbors to identify
candidate callers and dependencies. Read relevant source before changing behavior.
Use a small induced subgraph when several returned symbols interact. Search results
may use cached embeddings or lexical fallback, and the graph may omit relationships.
After an empty lookup, use a source search and inspect the actual call sites. Avoid
spending the remaining tool budget repeatedly querying the same unproductive symbol.
