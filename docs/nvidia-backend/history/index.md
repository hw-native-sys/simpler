# CUDA Evaluation History

This directory keeps evaluation history out of the current review path.

## Organization

- `captures/` stores capture-specific reports and older accumulated notes.
- `../changelog/` stores human-written change reports for review slices.
- Raw generated JSON, Markdown, and SVG artifacts remain under `tmp/`.

## Current Historical Entries

- [Current-head layered-cross capture](captures/current-head-layered-cross-743709f3.md)
- [Legacy accumulated capture notes](captures/legacy-captures.md)

## Rule

Do not append long artifact inventories to `evaluation-current.md`. Add a
short current summary there, then put capture details in `history/captures/`
with the artifact root, command shape, validation command, and review signal.
