# Comment Content: state WHAT, not WHY or change-history

A code comment states a **non-obvious fact about the code as it currently
stands** — an invariant, a contract, a hidden constraint, a unit, a
precondition. Present tense, standalone, no reference to any edit that produced
it. If there is no such fact, write no comment.

## Never write a comment that

1. **Justifies or rationalizes a change.**
   Bad: `// keeps the task-record path uniform with the others`
   Bad: `// removed the barrier here for performance`
   Bad: `// no need to lock, it's safe`
2. **Narrates the modification or its history.**
   Bad: `// no longer needed`, `// now uses the engine`, `// previously wmb'd`,
   `// moved from switch_records_buffer`, `// was 16384 before`
3. **Restates mechanics the code already shows.**
   Bad: `// increment the count`
4. **Explains a deletion.** A removed line needs no epitaph. Do not add a
   comment *because* you deleted code. If the deletion changes an invariant,
   state the new invariant as a present-tense fact (rule below) — otherwise say
   nothing.

## A good comment states a fact that survives on its own

```cpp
// Publication barrier lives in enqueue_ready(): the host never reads this
// buffer before its ready-queue tail advances, so no per-record barrier here.
```

That is a WHAT: it describes the invariant that holds *now*. It reads correctly
to someone who never saw the diff. Contrast the rejected form — `// this keeps
the path uniform` — which only means something relative to the change.

Litmus test: **would the comment still make sense to a reader who has no idea
this line was ever edited?** If it only makes sense as a note about the change,
it belongs in the commit message, PR description, or `docs/investigations/` —
not in the code.

## Where change-context goes instead

Rationale, measurements, alternatives considered, and "why we did it this way"
are valuable — they go in the **commit message**, **PR description**, or a
**`docs/investigations/`** entry. Keeping them out of comments is what keeps
comments from rotting into lies when the next edit lands.

## Relation to the other rules

- Sharpens [`codestyle.md`](codestyle.md) §1 (no plan-specific
  `Phase 1 / Step 1 / Gap #3` comments): the same ban, generalized to all
  change-narration.
- Reconciles [`doc-consistency.md`](doc-consistency.md) §5 terminology: what §5
  calls a comment's "why" is the **non-obvious invariant / constraint** — a
  present-tense fact, i.e. a WHAT in this rule's terms. It does **not** license
  change-rationale or edit-history in comments.
