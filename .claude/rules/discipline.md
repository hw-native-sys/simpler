# Working Discipline

Repo-agnostic behavioral guidelines that complement the built-in simplicity
and surgical-edit rules already in the system prompt. These target gaps the
system prompt does not cover.

## 1. State assumptions on ambiguous requests

If a task has multiple reasonable interpretations, name them and pick one
explicitly, or ask. Do not silently choose one and implement it — when the
silent choice turns out wrong, the rework costs more than the clarification
would have.

- "Add validation" → ask which inputs and what failure mode (raise? return
  an error? log?) before writing any code.
- "Refactor X" → ask what the success condition is (same tests pass?
  benchmark unchanged? API preserved?).
- "Make it faster" → ask what "fast enough" means and how to measure.

## 2. Match local style when editing

Follow the conventions of the file you are touching, even if you would write
it differently elsewhere. Do not reformat, rename, or refactor adjacent code
that is not part of the request.

- If surrounding code uses snake_case and yours uses camelCase, match the
  file.
- If unrelated dead code is next to your edit, mention it — do not delete
  it as a drive-by.
- Every changed line should trace directly back to the user's request.

## 3. Bug fixes start with a failing repro

For any defect that can be reproduced in code, write a test that fails
against the current behavior first, confirm it fails, then make it pass.
This guarantees the fix actually addresses the reported bug and leaves a
regression barrier behind.

Exceptions: infrastructure/build breakage where a test is impractical, or
one-character typo fixes where the diff itself is the verification.
