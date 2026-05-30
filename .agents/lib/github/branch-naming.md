# Branch Naming

Generate branch names from commit subjects.

## Prefix

- `Add` or `Update`: `feat/`
- `Fix`: `fix/`
- `Refactor`: `refactor/`
- `Support` or `CI`: `support/`
- `Docs`: `docs/`
- Other: `support/`

## Slug

Take the subject after `Type:`, lowercase it, replace non-alphanumeric runs with `-`, trim leading
and trailing hyphens, and truncate to 50 characters.

Example:

`Refactor: simplify CUDA runtime builder` -> `refactor/simplify-cuda-runtime-builder`
