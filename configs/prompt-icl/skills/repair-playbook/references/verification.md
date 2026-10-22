# Execution evidence

A useful reproduction distinguishes the faulty behavior from the expected behavior.
If feasible, run it before and after the edit. Confirm that it imports the repository
code being changed. Then run a focused existing regression test for neighboring
behavior. A generated assertion may encode a mistaken assumption; reconcile it with
the issue and existing contracts. An import or fixture error is an environment result,
not evidence that the patch is correct or incorrect. Record what ran and what remains
unverified. Review the source diff, capture the patch, and finish the task.
