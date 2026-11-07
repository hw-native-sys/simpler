# Plan and board

## Check the starting point

Fetch the intended base without moving another worktree's branch. Inspect the
current Issue and all open/draft PRs relevant to the affected paths, including
pagination and overlapping diffs. Record exact base SHA, observation time,
related PR heads and disposition: reuse, integrate, wait, or independent scope.
Repeat before dispatch/rework and publication when those facts may have changed.
A failed lookup means unknown, not an empty list. Avoid dispatching potentially
duplicate writers until that overlap is resolved; safe read-only work can continue.

Write a concrete agreement: goal, exclusions, user-visible behavior, ownership and
lifetimes, failures/close, cost, acceptance and authorization. For material design
changes, prepare a recommendation and alternatives, review them internally, then
ask only for the unresolved decision. Bind the answer to the proposal and affected
items. Ordinary implementation choices and repairs within approval need no new
permission. Do not split a feature into a chain of unusable PR fragments just to
parallelize authors.

## Align the design

Use this only for a material design trade-off, not for routine implementation
choices.

### One-question protocol

Walk the decision tree one branch at a time. Ask one question at a time and wait
for the answer. For each question, give a recommended answer. Look up facts in
the repository or tools yourself; decisions belong to the user. Do not start
implementation until the user confirms shared understanding.

Record each resolved decision, the rejected alternatives, and the reason in the
local plan. Bind later work to the decision it depends on.

### Design brief

For a change that introduces or reshapes an interface, write a short brief:

- Problem and constraints.
- Interface at the seam: what callers must know (types, invariants, ordering,
  error modes), and what the implementation hides.
- Dependency category: in-process, local-substitutable, remote-owned, or true
  external.
- Test seam: where behavior can be observed without reaching inside.
- Deletion test: if this module were deleted, would the complexity vanish or
  reappear across callers?
- Alternatives considered.

Use the repository's vocabulary (`Worker`, `Chip`, `Core`, `runtime`,
`platform`). Do not import an external vocabulary wholesale.

### Durable decisions and terms

Record a decision only when it is hard to reverse, surprising without context,
and the result of a real trade-off. Keep it to one paragraph: decision,
alternatives, consequences.

Use the existing documentation surfaces:

- `docs/design/` for a durable design document.
- `docs/investigations/` for a considered-and-dropped proposal; add it to the
  index in `docs/investigations/README.md`.

Do not create `CONTEXT.md` or `docs/adr/` by default. Prefer updating an
existing document; create a new one only when the user asks or no suitable
document exists.

## Keep a useful local plan

Choose a lightweight format that fits the task and current tools: prose, a table,
a diagram or an existing planning facility. No particular filename, directory or
serialized schema is required. Keep one authoritative plan under main-agent control;
children provide their own reports. Store private execution details in an ignored
location or outside the checkout, and keep secrets out of shared artifacts.

For each meaningful item, make its outcome, owner, dependencies, write scope,
acceptance and supporting evidence clear. Distinguish planned, running, submitted
for review, accepted and blocked work in whatever presentation is appropriate.
The main agent accepts work only after review. A child's submitted result does not
satisfy a dependency. Independent accepted artifacts can unlock work immediately;
a join waits for every required predecessor.

Distinguish three kinds of prerequisite:

| Prerequisite | What satisfies it |
| ------------ | ----------------- |
| An artifact | Main agent reviews the actual artifact and checks |
| A design decision | User's existing or new approval of the concrete scope |
| An integrated dependency | GitHub merge observed and required base verified |

Connect pending decisions to the work they affect. An approved design is not a
merged base, and an accepted patch is not merged either. A cancelled approach does
not satisfy its old dependencies: revise the plan and preserve the reason.

Keep enough dispatch context to resume: exact worktree/base, inputs, approved
scope, allowed files/actions, hardware access, checks, author identity and report
location. Preserve current feedback and unknown operations. GitHub receives
business milestones and confirmed design, with discussion in comments only when
publication is authorized; local execution details do not become another queue.

## Slice the work

### Spec slice

For each meaningful delivery, state:

- Problem: what is wrong or missing, from the user's perspective.
- Solution: the chosen behavior.
- Implementation decisions: interfaces, contracts, state, ownership. Avoid
  file paths and code snippets that will go stale.
- Testing decisions: what makes a good test and at which seam.
- Out of scope.

Adapt user stories to the actual actors (runtime developer, workload owner,
operator); keep a small infrastructure change short. Link the slice to its
Issue; do not create a second tracker.

### Tracer-bullet tickets

Cut work into vertical slices, not horizontal layers:

- Each slice is a narrow but complete path through the affected layers and is
  independently verifiable.
- Record its blocking edges: the slices that must finish first. No blockers
  means it can start now.
- Prefactor first when the change requires it ("make the change easy, then make
  the easy change").
- For a wide mechanical refactor, use expand-contract: add the new form beside
  the old, migrate call sites in batches sized by blast radius, then remove the
  old form. Block each batch on the expand and the removal on every batch.

Map each slice to the existing plan fields: outcome, owner, dependencies, write
scope, acceptance, evidence. Work the frontier: items whose blockers are done.
Do not start a slice whose blocking edge is only an approved design or an
unmerged patch; those are not integrated dependencies.

## Present the board for the current task

For substantial parallel work, provide a readable overview suited to the user's
needs. When an HTML board is useful, have the agent design or update it using the
current plan and available tools. Prefer an existing suitable board over creating
another one. Choose the layout and implementation at that time; there is no fixed
template, field set or rendering script to follow.

The presentation should help the user answer:

- What is the complete goal, and which parts are finished or still active?
- Which work can run in parallel, and what depends on an artifact, decision or merge?
- Who owns each item, what is blocking it, and what happens next?
- Which concrete decisions need the user, and which PRs are ready to merge?
- What evidence supports the status, and when was it last checked?

Show a DAG when dependencies matter; preserve joins and shared prerequisites rather
than implying every task has only one parent. For a large graph, expandable nodes
and concise summaries on hover or selection can keep it readable. Adapt these
interactions to the current user preference and task size. Omit empty categories,
redundant metadata and controls that do not help a decision.

Derive the board from the current plan and verified GitHub observations. Do not
maintain conflicting handwritten status sources or present a stale snapshot as
live data. Check that dependencies are complete, acyclic and consistent with the
recorded acceptance. A useful-looking board is not itself evidence of completion.

Ready requires all agreed work accepted, review and required CI for the current PR
head. Withdraw it when the head changes. Source inspection, build, local tests,
hardware measurements and CI are distinct evidence levels; do not promote one into
another or extrapolate untested platforms. Keep merge observation distinct from
verification of the resulting baseline.
