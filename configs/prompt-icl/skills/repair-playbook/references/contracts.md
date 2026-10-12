# API contracts

Write down the input condition and expected observable result from the issue. Inspect
the implementation, a caller and a relevant existing test. For a changed parameter,
check the omitted case, an explicit ordinary value and a boundary value. Preserve
specified exception classes, status codes, and exact messages. Trace the value through
wrapper functions when the local implementation appears correct. A small change at
the wrong abstraction layer can pass a reproduction while breaking another caller.
