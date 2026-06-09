# Environment-Variable and Macro Gating

Environment variables and preprocessor macros that gate *behavior* (feature
flags, opt-in/opt-out switches, conditional code paths) are configuration
surface. They are easy to add and hard to remove: each one becomes a
permutation that must be documented, wired through CI, and reasoned about
forever. Keep them to a minimum.

## 1. Adding a behavior gate requires explicit user permission

**Before adding any new environment variable or macro that isolates or gates
behavior, you MUST explicitly ask the user for permission first.** State what
the flag would be, why you think it's needed, and what the default is — then
wait for an explicit yes.

This covers:

- New `std::getenv(...)` / `os.environ[...]` reads that change what the code
  does.
- New `#ifdef` / `#if defined(...)` blocks (or `-D` compile flags) that select
  between behaviors.
- New pytest CLI flags / env vars that toggle a code path.

It does **not** cover reading an *already-agreed* flag, or env/macros that are
purely diagnostic (a log-verbosity knob that changes no behavior) — though when
in doubt, ask.

Prefer alternatives that need no gate: do the thing unconditionally when it is
always correct; derive the decision from existing state; or rely on an existing
project invariant (e.g. onboard work always holds an exclusive `task-submit`
lock, so a device force-reset on the error path needs no opt-in flag).

## 2. Every env/macro touch is a chance to delete one

Whenever you modify code that reads an environment variable or branches on a
macro, check whether it can be **removed** — the gate may be dead, redundant,
or replaceable by unconditional behavior now that the surrounding code has
changed. If so, delete it (and its CI wiring, docs, and any include added only
for it). Leaving a stale gate is worse than never adding it: it implies a
choice that no longer exists.
