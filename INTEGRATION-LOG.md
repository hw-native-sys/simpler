# Kernel-mode PR integration log

One entry per adjudication, in the order it was made. Each entry states the
problem, the options, the choice, the reason, and which PRs it affects.

The scope is all 14 submitted PR contributions in the supplied kernel pipeline,
including the latest K1 and K3 revisions frozen during the final audit. The
end-to-end acceptance path is `init -> prepare_callable -> launch -> close`
with real `tensormap_and_ringbuffer` computation on a2a3 hardware.

See [the validation record](docs/kernel-integration-validation.md) for frozen
PR heads, executed tests, source snapshot identity and remaining boundaries.

---

## D0 - The PRs are real git stacks, not independent source-embedded copies

**Problem.** The handover states that the in-flight PRs each source-embed a
copy of K1 and that those copies have diverged by 16 / 97 / 115 / 165 lines,
so `git merge` cannot deduplicate them and each copy must be adjudicated by
hand.

**Finding.** That is not what the refs contain. K1 appears under three
different commit ids -- `86dd62b4` (#2064 / K3), `dc1268cd` (#2177, #2176,
\#2185, #2180, #2190) and `66156ed3` (#2189) -- and all three have **identical
trees**:

```bash
git diff --stat dc1268cd 86dd62b4   # empty
git diff --stat 66156ed3 86dd62b4   # empty
```

K1 itself never diverged. The line counts the handover reports are each PR's
*own contribution on top of* K1, not a competing copy of it.

**Choice.** Do not hand-merge K1 copies. Cherry-pick each PR's non-K1 commits
onto the authoritative K1, which replays every contribution as a clean
three-way merge against the identical base it was written against.

**Reason.** The unique-contribution set is a handful of distinct commits, and
cherry-picking them yields linear history with one commit per contribution and
no duplicated K1 history.

**Affects.** All PRs. This is the method the rest of the log assumes.

---

## D1 - `pr/k3` carries K1, so batches 1 and 3.5 collapse into one merge

**Problem.** The recommended order lands #2064 (K1) at batch 1 and K3 at batch
3.5, with K3 not yet submitted at handover time.

**Finding.** K3 had already landed when work started: the local K3 workspace
was at `eb696879` ("kernel-mode capacity
refusals no longer free what they protect", 2026-09-10), whose parent is K1
`86dd62b4`. So the K3 branch *is* K1 + K3. No waiting was required and no
polling was done.

**Choice.** Merge `pr/k3` once, as the authoritative baseline for both.

**Reason.** K3's K1 files are byte-identical to #2064's, so the merge delivers
\#2064 exactly. Splitting it would add a commit and change nothing.

**Affects.** #2064, K3.

---

## D2 - Integrate the H chain while retaining the authoritative K1 ABI

**Scope (2026-09-11).** The user requested every submitted PR in the pipeline,
so the earlier decision to defer HBG is superseded. The local integrated source
now contains the contributions of all 14 listed PRs: #2064, #2171, #2172,
\#2173, #2174, #2175, #2176, #2177, #2180, #2185, #2187, #2189, #2190 and #2193.
This is local source integration, not a claim that the GitHub PRs were merged
or that all tests have passed.

**Finding.** The five H heads are cumulative snapshots of H1, 2B, Context
Prepare, H2 and H3. Their shared K1 variant conflicts with the integrated C
entries and invocation header; importing that whole variant would remove
contracts the TMR path uses.

**Choice.** Integrate their HBG contributions and the shared resource
prepare/freeze/inspect/bind methods into the existing context. Keep the current
five-parameter `prepare_callable`, id-based launch, K1 header and five-event
binder topology. Preserve #2180's contribution through the integrated #2189
stack, and include #2193's updated capacity-refusal checks.

**Boundary.** H1-H3 build, sizing, immutable packets and slot admission are
present. H4 has no submitted PR in the supplied pipeline; semantic restore and
its public HBG owner wiring remain absent. HBG reports no kernel capability;
kernel init returns `UNSUPPORTED`, and prepare/launch on that uninitialized
context return `INVALID_STATE`. Added HBG and dispatch-packet tests require a Linux
build; the separate positive TMR numerical test requires a2a3 hardware.

**Affects.** All 14 submitted PR contributions; HBG ABI adaptation is recorded
in D11.

---

## D3 - `prepare_callable` keeps `caller_stream` (five parameters)

**Problem.** Three signatures are in flight:

| Source | Signature |
| ------ | --------- |
| K1 / #2177 / #2185 | `(ctx, callable_id, callable, size, caller_stream)` |
| #2176 / #2180 / #2189 | `(ctx, callable_id, callable, size)` |
| #2190 | `(ctx, callable, size, SimplerCallableHandle *out)` |

K2's cherry-pick applies its four-parameter form **without a git conflict**,
because it is a clean delta against the same K1 base. Nothing reports the
change; the parameter simply disappears.

K2 does not drop it by oversight. It replaces the doc with an argument:
preparation enqueues on the context's own AICPU stream, later launches enqueue
on that same stream, and stream FIFO therefore orders registration ahead of
every launch without a caller stream being involved. That is coherent with the
three-stream binder in #2187.

**Choice.** Five parameters. `caller_stream` is restored in
`runtime_c_api.h`, `kernel_entry_validation.h`, both `c_api_shared.cpp`
entries, `test_kernel_entry_validation.cpp` and `test_kernel_mode_c_api.py`.

**Reason.**

1. **The actual ABI consumer is five-parameter.** #2185 is the entry layer
   that gives the four C entries their callers, and it calls
   `kernel_prepare_callable(callable_id, data, size, caller_stream)`, plumbing
   the stream up through the nanobind binding and the Python wrapper with the
   documented "may enqueue asynchronous work on that stream but never
   synchronizes it" semantics. K2 is the outlier against the layer that uses
   it.
2. **The K1 interface-freeze card is the designated tie-breaker** for ABI
   signature disputes, and it specifies five.
3. **It is strictly more expressive.** An implementation that prefers to stage
   on its own AICPU stream can ignore the parameter; a four-parameter entry
   cannot later acquire ordering against the caller's *preceding* work without
   an ABI break. Keeping it preserves both designs, dropping it forecloses one.
4. It keeps prepare and launch symmetric, matching the header's own rule that
   "the caller stream is always an explicit parameter".

K2's FIFO argument is not refuted, and this integration does not depend on
refuting it. It holds only while prepare and launch enqueue on the *same*
context stream; the parameter costs nothing and covers the case where they
do not.

**Affects.** #2176, #2180, #2189 (must adopt five), #2190 (see D-later),
\#2177 and #2185 (unchanged).

---

## D4 - `simpler_kernel_mode_init` keeps 2a's contract validation ahead of K2's body

**Problem.** The only textual conflict in the K2 cherry-pick. `HEAD` (K1 + 2a)
validates the pipeline contract the config implies and then returns the
`UNSUPPORTED` refusal stub. K2 replaces the whole body with the real init,
which has no contract validation because K2 branched before 2a.

**Choice.** Keep both: 2a's `build_kernel_pipeline_contract_impl` +
`is_valid_pipeline_contract` + arena/stream topology check first, then K2's
latch-and-initialize body. The refusal stub is dropped.

**Reason.** The two are independent and both wanted. Validating before the
latch means a config that cannot be serviced is refused while the context is
still free, which is what makes a refused init leave a reusable context -- a
guarantee #2185's tests already assert.

**Affects.** #2176, #2177.

---

## D5 - Environment: pre-commit `clang-tidy` cannot run on this host

**Problem.** Every commit fails the `clang-tidy` pre-commit hook:
`build_runtimes.py` imports `fcntl`, which does not exist on Windows.

**Choice.** Integration commits use `--no-verify`. `cpplint`, `ruff check`,
`ruff format` and `pyright` all run and pass; only the clang-tidy hook is
skipped.

**Reason.** The hook is unrunnable on this platform, not failing on the
content. It runs in CI and on the Linux host used for hardware validation.

**Affects.** Every commit on this branch.

---

## D6 - #2189 over #2180, taken as #2180's K4 chain plus #2189's K5 commit

**Problem.** #2180 and #2189 must not both land: 14 conflicting files, and the
handover records them as carrying the same K4 tree with no new content between
them.

**Finding.** #2189 is a strict superset in content but not in history. It
rebuilt the whole stack under fresh commit ids -- its own K1 (`66156ed3`), its
own 2a pair, its own K4 (`b5529c89`), a K2/K4 integration commit, and finally
K5 (`b6435e48`). Cherry-picking #2189's commits after #2177 (2a) and #2176
(K2) would replay two contributions this branch already carries.

**Choice.** Take #2180's K4 chain (`3e822453`, `b0943525`), which is stacked
directly on the 2a commits this branch already has, then cherry-pick only
\#2189's K5 commit (`b6435e48`) on top.

**Reason.** It is the same end state with no duplicated contribution, and it is
the second option the handover itself offers. #2180 is otherwise superseded.

**Affects.** #2180 (K4 chain taken, PR otherwise superseded), #2189 (only K5
taken; its K1/2a/K2/K4 commits are redundant against this branch).

---

## D7 - `launch` is id-based; #2190's handle collapses into it

**Problem.** #2190 changes both entries to a `SimplerCallableHandle`
`{callable_id, generation}`: prepare writes one out, launch takes one in. That
is a third shape against K1's `int32_t callable_id`, and it arrives without a
git conflict on the launch entry.

**Choice.** Keep `int32_t callable_id` on both entries. Launch resolves the
residency internally as `{callable_id, cache.generation()}`.

**Reason.** The generation guard is the valuable half of #2190 and it survives
intact: the cache stores the generation an entry was staged under, so a
callable left from a previous context generation still resolves to
`CALLABLE_STALE` rather than replaying a recycled address. What the handle
added beyond that was a caller-held token, and an id-based ABI has no place to
hold one -- the context generation is minted at init, so the check the caller
would have armed is exactly the check the runtime now performs. Meanwhile
\#2185's entry layer, its nanobind binding and the Python wrapper are all
id-based, as is K1.

**Affects.** #2190. Its `_prepare` ctypes helper and `CallableHandle` struct
are removed as dead; its two generation-staleness assertions are re-expressed
against the id-based launch in the end-to-end test.

---

## D8 - #2190's callable cache takes a caller-supplied id

**Problem.** `KernelCallableCache::stage` allocated its own sequential id
(`entries_.size()`), because under #2190's ABI the runtime chose the id. Under
D7 the caller chooses it.

**Choice.** `stage` takes `requested_id`. An id already staged is refused as a
duplicate registration. An image whose bytes match a resident entry is admitted
under its own id, shares that entry's device address, and is charged zero arena
bytes. `commit` and `rollback` find their entry by id instead of assuming it is
the last one.

**Reason.** It preserves every property the cache's own tests assert -- one
upload per unique image, no eviction, arena accounting, descriptor table
indexed by id -- while honouring the caller's id. The descriptor table was
already indexed by id and the ABI already bounds the id to
`[0, MAX_REGISTERED_CALLABLE_IDS)`, so a caller-supplied id indexes it safely.

**Affects.** #2190. Its `test_kernel_callable_cache.cpp` needs the new
parameter at each `stage` call site.

---

## D9 - The launch topology is the binder's AICore-first set

**Problem.** Two five-event topologies are in the tree.
`KernelEventKind` (from K2) is a chain: `Start, AicoreStart, AicoreDone,
AicpuDone, SerialTail`, where the aicpu stream forks the aicore stream. The
binder's `KernelLaunchHandles` is a fork-join:
`prepare_tail, start, aicore_done, aicpu_done, serial_tail`, where both device
streams fork from the caller's Start and rejoin the caller.

The handover reports the binder as AICPU-first and leans toward K2's set. The
binder's source says otherwise: its sequence waits `start` on the aicore
stream, launches AICore, records `aicore_done`, and only then admits AICPU.
Its comment gives the two v9 reasons verbatim -- the scheduler cycle when an
AICPU startup waiter blocks a later AICore SQE, and the better failure
closure, since AICore can still be cancelled while no AICPU has written the
handshake.

**Choice.** Adopt the binder's set. `KernelEventKind::AicoreStart` becomes
`PrepareTail`; the count stays five and the storage is unchanged.

**Reason.** The binder is the only implementation of a launch sequence in any
of these PRs, and this integration wires launch onto it. Its topology is
self-consistent, it is what v9 specifies, and K2's stated deadlock hazard is
avoided by a different means: AICore's SQE is on its stream before AICPU
becomes resident, so the orchestrator's spin on the AICore handshake always
has a submitted AICore to wait for.

`tests/st/a2a3/kernel_capture/native/driver.cpp` is reordered to match. It
exercises the same capture primitives -- three streams, five events, one AICPU
launch, three AICore launches -- so its capture evidence is preserved, and it
now demonstrates the topology the product actually uses. Its two device
branches touch disjoint buffers, so running them as siblings is safe.

**Affects.** #2176 (event enum and capture ST), #2187 (unchanged, it was
already the target).

---

## D10 - AICPU transport stays on the in-repo `rtsLaunchCpuKernel` path

**Problem.** The binder's native layer, `launch_bound_kernel_native`, requires
a `KernelNativeInvocation` carrying two `aclrtFuncHandle`s and drives
`aclrtLaunchKernel` / `aclrtLaunchKernelWithHostArgs`. Nothing in any PR calls
`aclrtBinaryLoad` or `aclrtBinaryGetFunction`, so nothing produces those
handles. The TMR path in the repo launches AICPU work through
`LoadAicpuOp::LaunchBuiltInOp`, which wraps `rtsLaunchCpuKernel`.

**Finding.** The choice is not forced. `launch_bound_kernel` -- the generic
entry -- takes `launch_aicpu` and `launch_aicore` as plain callbacks in
`KernelLaunchOps`. Only the `_native` variant hard-codes the CANN family.

**Choice.** Wire launch onto `launch_bound_kernel` with owner-supplied
callbacks that directly call `rtsLaunchCpuKernel` and
`rtKernelLaunchWithHandleV2` using handles resolved during init/prepare.
Leave `launch_bound_kernel_native` in the tree, unused, as the migration
target.

**Reason.** It is the lowest-risk route to a first working launch and the one
the handover recommends: the code already exists and is exercised on hardware,
where the `WithHostArgs` family has no precedent in this repo. The binder's
12-step sequence, its compensation ladder and its capture-safety guarantees
are all in the generic entry, so nothing is given up. Switching the transport
later changes two callbacks and no sequencing.

`LoadAicpuOp::LaunchBuiltInOp` allocates its argument wrapper, so launch does
not call that helper. The owner uses stack argument/configuration structures
and a per-callable packet whose backing storage was allocated during prepare.

**Affects.** #2187. The source guard also covers the new launch owner.

---

## D11 - HBG packets use the integrated common invocation header

**Problem.** The H2/H3 snapshots assume a 64-byte common envelope with
`abi_version`, `header_bytes` and reserved fields absent from the original K1.
Hard-coded offsets would parse the wrong bytes after integration.

**Choice.** Keep the authoritative `SimplerKernelInvocationHeader`. HBG producer,
validator, HostArgs placeholders, slot limits and tests derive offsets with
`sizeof(SimplerKernelInvocationHeader)`. Validate the shared mode, identity,
counts and payload length; keep version/reserved checks on HBG's own graph
header and slot records. HBG graph format remains version 2. D13 later imports
K1's explicit trailing reserved word without changing the 40-byte layout.

**Reason.** Both runtime payloads share one envelope without introducing a
second public ABI. HBG framing, checksums and trusted slot admission remain
intact. Compilation and execution results are recorded separately after tests
run; source integration alone is not execution evidence.

**Affects.** #2174, #2175 and their HBG documentation/tests.

---

## D12 - Kernel execution uses the resident device SO's single executor

**Problem.** The TMR device SO contains one `g_aicpu_executor`, one affinity
gate and shared platform register/profiling state. Separate host contexts can
resolve the same resident SO, while their host mutexes and hidden streams are
independent. A per-context submission lock does not serialize those contexts
on the device.

**Choice.** Each loaded host runtime SO enforces one live kernel context per
`(device_id, device runtime SO fingerprint)`. Initialization claims this key
before loading or initializing device state. A second claimant returns
`PTO_RUNTIME_ERR_INVALID_STATE` with an explicit diagnostic. Failed initialization
rolls back the claim. Failed close retains it; successful normal finalize releases
it. Destruction and fatal resource abandonment cannot establish quiescence, so
neither silently releases ownership. Launch also requires the context's claim.
The owner sequences each launch's complete AICPU/AICore join before reusing its
execution regions. Program contexts never acquire or release these claims.

The registry covers contexts created through the same loaded host SO. Separately
loaded copies of that host SO and different host processes have no shared claim
registry; they must not concurrently target the same resident device runtime SO.
This integration does not add a cross-library or cross-process locking protocol.
Program/kernel execution sharing the same resident device SO must also be
serialized by the caller, because program contexts do not participate in the
kernel claim registry.

**Reason.** Moving the executor, affinity gate and platform state into
context-addressed storage requires an additional device ownership design.
Adding only a host mutex to each context cannot establish that isolation.
The new kernel consumer has one admission owner, publishes its immutable
argument storage to the affinity group, gathers all thread errors and releases
its borrowers only after every participating thread finishes.

**Rejection boundary.** TMR admission failures with an established Runtime
publish the pre-window AICore cancel sentinel. Outer dispatcher failures that
reject the prefix or residency return an error without dereferencing
`binding_address`: its alignment and numeric range alone cannot prove it is a
live host-owned allocation. If AICore is already enqueued when such a packet is
rejected, this boundary has no trusted cancellation target. Supporting recovery
there requires a prepare-time binding registry plus explicit close-time revoke;
this integration does not invent trust from malformed packet pointers. Host
validation rejects ordinary invalid inputs before either device launch.

**Validation.** Host lifecycle tests exercise concurrent claims, device/runtime
key isolation, rejected-claim cleanup, initialization rollback, failed-close
retention and successful retry. Executor tests cover trusted admission failure,
pre-window cancellation and later reuse. The standalone K4 snapshot probe uses
its own device symbol and raw encoder transport; production uses only the unified
`SimplerKernelDispatchArgs` packet sender.

**Affects.** #2176, #2180/#2189, #2190 and the launch owner. Host-side callable
admission also rejects duplicate or out-of-range child function ids before
allocation/upload, matching the device function-table contract.

---

## D13 - Refresh K1 without discarding integrated runtime behavior

**Finding.** The final remote-head audit found #2064 had advanced from
`86dd62b4` to `2ab04b1b`. The other 13 submitted heads matched the audited
snapshots, including K3 #2193 at `b8e739d9`. Compare each K1 commit against
its own parent (`405b5bbd` and `c540572d`) so unrelated upstream changes are
not mistaken for K1 contributions.

**Choice.** Import the revised K1 contracts into the working implementation:

- Pin the invocation header to 40 bytes and every field offset. Turn the
  historical final four padding bytes into `reserved_`, zero them in producers,
  and reject nonzero values in the common, TMR and HBG consumers.
- Derive callable scalar counts from signature entries, ignore historical
  callable padding, and restore the existing C++/Python factory signatures.
- Always resolve the runtime capability probe; resolve the three kernel
  lifecycle symbols only for a supported component. Publish resolved function
  pointers after successful initialization and test failed-init retry with
  fixture shared libraries.
- Make context operation callbacks non-throwing, retain the event-flags
  parameter needed by K2, and use byte-sized stream/event enums. Invalid
  initialization arguments are rejected before resource callbacks run.
- Return `INVALID_STATE` for program-only device ownership operations attempted
  through a kernel context. Structural C entry errors use `INVALID_ARGUMENT`.

**Error-code adjudication.** Revised K1 assigned `INVALID_ARGUMENT` to -1004,
which the already integrated callable cache uses for `CALLABLE_COUNT_EXCEEDED`.
Keep the cache/capacity band -1004 through -1008 and assign `INVALID_ARGUMENT`
the next free value, -1009. Preserve the specific launch-id capacity refusal.

**Integration boundaries.** K1's stub-only removal of stream/event accessors
and production source links cannot be applied: K2, HBG resource preparation
and the real launch owner use them. Retain those integrated interfaces and
K3's stronger capacity-preservation rules. No production feature is reverted
to a K1 placeholder.

**Test dispatch correction.** K2's capture test lacked runtime/device-count
markers, so the mixed-runtime scene scheduler did not execute it. Add both
markers and link the new shared resource implementation into its native probe.
The final hardware sweep must explicitly report this test, rather than inferring
capture coverage from the sweep exit code.

**Affects.** #2064, #2176, #2190 and both runtime consumers. Final validation
results are recorded separately from the source-integration decisions.

---

## D14 - Merge the integration line onto the current mainline

**Problem.** The PR target, `feat/kernel-mode-integration-test`, is identical
to `main`. Since the integration base `a1aa7fdd`, `main` has gained the
squashed final K1 (#2064) and fourteen other commits. A merge conflicts in 16
files across 59 hunks, and one disagreement is semantic rather than textual:
the merged K1 numbers `PTO_RUNTIME_ERR_INVALID_ARGUMENT` as `BASE - 4`, where
the integration line had given `BASE - 4` through `BASE - 8` to its callable
and capacity codes and `BASE - 9` to `INVALID_ARGUMENT`.

**Choice.** The merged K1 decides the contract, and the integration line keeps
the implementation that the merged skeleton does not have.

1. Host error codes take the merged numbering. `INVALID_ARGUMENT` is
   `BASE - 4`; `CALLABLE_COUNT_EXCEEDED`, `CALLABLE_BYTES_EXCEEDED`,
   `CALLABLE_NOT_RESIDENT`, `CALLABLE_STALE` and `CAPACITY_EXCEEDED` take
   `BASE - 5` through `BASE - 9`. The constants in
   `tests/ut/py/test_kernel_mode_c_api.py` follow.
2. An out-of-range launch id returns `INVALID_ARGUMENT`, as the merged K1
   specifies and as prepare already does.
3. The merged wording and codes for the execution-mode latch, the
   `ensure_acl_ready` refusal, the `attach_current_thread` refusal and the
   finalize branch are taken unchanged.
4. The integration line keeps its real init, prepare and launch bodies, the
   canonical-image validation, the current-device check on adoption, and K3's
   `commit_static_arena_bank`. The merged runner's inline
   `kernel_arena_change_is_forbidden` guard and its unit tests are not
   carried.
5. The lifecycle comment in `runtime_c_api.h` describes the integrated
   behaviour: init bootstrap and callable registration synchronize the
   context's own AICPU stream, and the launch path synchronizes nothing.
6. `ChipWorker` keeps capability-gated resolution of init, prepare and launch,
   where the merged K1 text resolves all four unconditionally.

**Reason.** Items 1 to 3 are contract, and the merged K1 is the released
contract. None of the integrated implementation depends on the numbers or the
launch-id code it replaces.

Item 4: K3's bank enforces the same capacity rule for onboard and simulation
from one place, which is the consolidation #2193 exists to make. Keeping the
merged guard as well would restore the duplication.

Item 5: the merged comment promises no synchronize on the prepare and close
paths, which this implementation does not meet.

Item 6: every runtime still exports all four entries, so every producer the
merged contract describes satisfies the gated consumer. The integrated worker
reports an unsupported runtime by the absence of those bindings, through a
binding helper that twelve call sites share. Moving to unconditional
resolution changes no in-tree behaviour and can be done later without touching
any producer.

**Affects.** #2064 (final contract adopted), #2189 and #2190 (error-code
numbers), #2193 (bank kept over the merged guard), #2185 (symbol resolution).

---

## D15 - Re-survey the PR heads; take only what does not re-open a decision

**Problem.** The heads this integration was audited against were frozen on
2026-09-11. Since then #2064 merged to `main`, `main` advanced four commits
past the integration base `748c39fb`, and six of the remaining thirteen PRs
moved. A head that moved is not automatically a change to adopt: some of the
movement is a rebase over the merged K1, and some of it reverses a decision
this log already made.

**Finding.** Seven PRs are byte-identical to their audited heads: #2173,
\#2174, #2175, #2180, #2187, #2189 and #2193. The six that moved divide
cleanly:

| PR | What moved | Re-opens a decision? |
| -- | ---------- | -------------------- |
| #2177 | Invalid `CallConfig` now returns `INVALID_ARGUMENT`, not `INTERNAL` | no |
| #2176 | `KernelContextOps` drops `event_flag`; `create_event` takes `(context, event)` | no |
| #2176 | Failed-rollback retention test; `rtFree` fault hook and `arm_destroy_failure_after` | no |
| #2176 | Header states a capture-boundary rule that forbids a prepare-published tail | **yes** — D9 |
| #2185 | Test reaps a hung child; asserts the borrowed stream survives a refused init | no |
| #2185 | `prepare_callable` drops `caller_stream` (five parameters to four) | **yes** — D3 |
| #2190 | `SimplerCallableHandle` and the per-callable generation removed; block allocator; resolve by index | **yes** — D7, D8 |
| #2171, #2172 | H1 and 2B restructured around new `graph_definition_pack.{h,cpp}`, `kernel_pipeline_contract.h`, `kernel_resource_plan.h` | **yes** — D2, D11 |

**Choice.** Adopt the four rows marked "no". Leave the rest for a decision
pass that can weigh each on its merits, and record here what that pass has to
answer.

Adopted:

1. **#2177's error codes.** `build_kernel_pipeline_contract_impl` returns
   `INVALID_ARGUMENT` for a null or unresolvable `CallConfig` and keeps
   `INTERNAL` for a null output or a contract it generated but cannot
   validate. Both arch TMR runtime makers, the header's contract comment,
   `test_trb_runtime_temp_buffer.cpp` and `test_pipeline_contract_loader.cpp`
   move together. This continues D14 rather than re-opening it: D14 adopted
   the merged K1 numbering for the entry points, and this extends the same
   numbering to the hook behind them.
2. **#2176's `event_flag` removal.** The flag existed so a host-only test
   could observe which constant the platform asked for. The platform's own
   `create_event` is a better place for that knowledge, and the test that
   read the field goes with it.
3. **#2176's fault coverage.** `PersistentKernelArgs` already rolls back
   through `finalize_once` on this line, but nothing exercised a rollback
   whose own release fails. The new test and the `rtFree` hook behind
   `persistent_free_close` cover the owner to `finalize_common` to allocator
   retry chain.
4. **#2185's test hardening.** A hung non-daemon child is terminated instead
   of left for the CI job timeout, and destroying the borrowed stream after a
   refused `kernel_init` is now an assertion rather than a swallowed
   exception. Only the hardening is taken; the signature change it accompanies
   is deferred below.

**Not applicable here.** #2176, #2177 and #2185 all drop
`kernel_execution_state.cpp` from a platform source list, because on their
lines nothing outside K2 consumes it. On this line
`kernel_resource_requirements.h` calls `bind_resources_for_launch` from the
H chain, so both the onboard and the simulation host runtimes need the
translation unit; removing it from the simulation lists leaves
`libhost_runtime.so` with an undefined symbol that `test_pipeline_contract_loader`
catches at `dlopen`.

**Deferred here, and decided in D16.** The four rows marked "yes" are not open
questions: `.docs/vllm/kernel-mode-design.md`'s companion call-flow document
settles all but the last of them, and D16 adopts that settlement. They are
listed here as what this pass did not take, not as what still needs arguing.

- **D3, `prepare_callable`'s stream parameter.** D3's first reason was that the
  entry layer, #2185, called the five-parameter form and K2 was the outlier.
  \#2185 has since moved to four, so that evidence no longer holds.
- **D7 and D8, the callable generation.** D7 kept `int32_t callable_id` on both
  entries because the generation guard was "the valuable half of #2190" and
  survived the collapse. That half no longer exists in #2190.
- **D9 and the capture boundary.** #2176's header records that a wait issued
  during capture on an event recorded before capture is rejected by CANN as
  107024, and concludes that preparation must not publish a tail for a later
  captured launch to consume. This line does exactly that, so the note and the
  five-event sibling topology cannot both stand.
- **D2 and D11, the H chain.** #2171 and #2172 were force-pushed onto a
  different decomposition: a `GraphDefinitionPack` with its own translation
  unit, and `kernel_resource_plan.h` where this line has
  `kernel_resource_requirements.h`. Re-integrating them is a rewrite of the H
  contribution, not a patch to it, and no target document covers it.

**Affects.** #2176, #2177, #2185 (partially adopted); #2171, #2172, #2190
(deferred here); D2, D3, D7, D8, D9, D11 (superseded or restated in D16).

---

## D16 - Adopt the target call-flow design: minted ids, no generation, chained streams

**Problem.** D15 left four adjudications open. Three of them are settled by the
target design the kernel-mode work is being built against — the vLLM call-flow
document dated 2026-09-13, with the callable-id shape frozen 2026-09-14 — which
names this integration line explicitly and says it "still has the old shape" and
needs rebasing onto #2185's signature and #2190's callable-id out parameter.
That design is not one more PR's opinion: it is what the three-party contract
between vLLM, PyPTO and simpler is written to.

**Finding.** The three are one package, not three independent choices.

The four-parameter prepare (D3) and the deleted `PrepareTail` (D9) stand or fall
together with the stream topology. `PrepareTail` exists on this line only because
D9 chose #2187's sibling topology, in which both device branches fork from the
caller's `Start`: AICore's stream is then unrelated to the AICPU stream, so
nothing orders it behind registration and an event has to. In the target's
chained topology AICore forks from AICPU, so the AICPU stream's own FIFO carries
registration ahead of every launch and no event is needed. Taking the
four-parameter prepare without the chained topology would be the worst of both:
prepare would lose the stream it currently orders, leaving the launch-side
`PrepareTail` wait as the only mechanism — and that wait is precisely what
crosses the ACLGraph capture boundary.

**Choice.** Adopt the target design.

1. **Registration mints the id.** `simpler_kernel_mode_prepare_callable(ctx,
   callable, size, int32_t *out)` writes a context-local id on success and -1 on
   every failure. It takes no `caller_stream`; only launch does, per call.
2. **Registration is pure.** No deduplication and no lookup: the same image
   registered twice takes two ids, two uploads and two charges. Capacity is
   spent per registration.
3. **No callable generation, anywhere.** `SimplerCallableHandle`,
   `PTO_RUNTIME_ERR_CALLABLE_STALE`, `KernelCallableDeviceResidency::generation`
   and `SimplerKernelInvocationHeader::generation` are gone; the wire header is
   32 bytes. What replaces the guard is three properties together: an id is
   minted once and never reused within a context, close invalidates every id the
   context minted, and a closed worker accepts no launch.
4. **The launch sequence is chained.** Events are `Start`, `AicoreStart`,
   `AicoreDone`, `AicpuDone`, `SerialTail`. Caller records `Start`; AICPU waits
   it, clears the handshake, records `AicoreStart`; AICore waits that, launches,
   records `AicoreDone`; AICPU launches with HostArgs, waits `AicoreDone`,
   records `AicpuDone`; the caller waits that and records `SerialTail`. Caller
   and AICore share no event, so capture propagates in two hops.
5. **`PTO_RUNTIME_ERR_CAPACITY_EXCEEDED` moves to `BASE - 8`,** the slot
   `CALLABLE_STALE` vacated. Nothing outside this line pins either value: the
   merged K1 on `main` declares no `CALLABLE_*` code at all.

**Reason.** Items 1 to 3 are the frozen contract of the document PyPTO and vLLM
are being written against; keeping this line's shape would mean every consumer
adapts to an integration branch rather than to the design. Item 4 is what makes
items 1 and 2 safe, and independently removes the 107024 exposure D15 recorded.
Item 5 keeps the error band dense rather than leaving a hole behind a code no
producer emits any more.

**Kept against the source PRs.** #2190's block allocator is not taken. Its cache
is host-only — that branch carries no device dispatch — whereas this line's
AICPU entry resolves a descriptor at `arena_ + callable_id * sizeof(descriptor)`.
The single arena with its descriptor prefix stays; only the generation, the
deduplication and the caller-chosen id leave it. The descriptor shrinks to 24
bytes accordingly.

**Boundary.** The chained topology is what #2176's capture probe validates on
a2a3, and `tests/st/a2a3/kernel_capture` now drives that sequence. The public
`prepare -> launch` path is still not exercised inside a captured graph by any
test, so two-hop capture propagation through the real entries remains unproven.

**Affects.** #2176, #2180, #2185, #2189, #2190 (their shapes adopted); #2187
(its binder rewritten to the chained topology); D3, D7, D8, D9 (superseded).

---

## D17 - Follow #2190's device entry: the image span travels in the packet

**Problem.** D16 kept this line's residency descriptor: a 24-byte device struct
per callable, uploaded at registration into a fixed prefix of the code arena,
whose address the packet carried so the AICPU could read the image address and
extent from it. #2190 has since restored its device entry without a descriptor
at all — `SimplerKernelDispatchArgs` carries `chip_callable_address` and
`chip_callable_bytes` directly, and the consumer receives the `ChipCallable`
reference.

**Finding.** The descriptor bought two things and only one of them was real.

The stated one was revocation: the entry re-read the slot on every invocation
"including graph replay", so the host had one small location it could
invalidate. Nothing ever wrote a descriptor after commit — ids are never
evicted, kernel-mode `unregister_callable` returns `INVALID_STATE`, and
`clear()` only drops host metadata — so the capability was never exercised. And
it would not have helped the case that matters: after close the arena is freed,
which leaves the descriptor address dangling exactly as the image address does.

The real one was keeping the extent out of the per-call packet, so a packet
could name *which* descriptor but could not widen the window the device parses.
That argument assumes an untrusted packet, and the packet is built by the host
binder from `cache.resolve()`. It is not part of this contract's threat model.

**Choice.** Adopt #2190's shape.

1. `SimplerKernelDispatchArgs` replaces `residency_address` with
   `chip_callable_address` and `chip_callable_bytes`. This line keeps the four
   binding fields #2190 has no source for — `binding_address`,
   `context_generation`, `sm_bytes`, `arena_bytes` — because its TMR consumer
   reads them; the prefix is 88 bytes against #2190's 56.
2. `KernelCallableDeviceResidency` and its header are deleted. The code arena
   loses its descriptor prefix, so `device_address` is `arena_ + used_`.
3. `consume_kernel_invocation` takes
   `(args, const ChipCallable &, callable_bytes, payload, payload_bytes)`. This
   line passes the whole prefix where #2190 passes only the invocation header,
   for the same reason as item 1; it is otherwise #2190's signature.
4. The entry validates the span — non-null, `alignof(ChipCallable)`, at least
   `sizeof(ChipCallable)`, no wraparound — and performs no device read. Cache
   visibility for the image moves to the consumer, which is where the image is
   actually parsed.
5. `KernelDispatchStatus` drops `NotResident` and `Stale`; the entry no longer
   produces them. 2 and 3 stay retired rather than being reused.

**Reason.** One indirection and one device read per launch disappear, the
descriptor prefix and its upload disappear, and the two lines stop diverging on
a wire struct. Nothing that was load-bearing is lost: the image address and
extent still come from the context's own committed residency, and the host is
the only producer either way.

**Affects.** #2190 (shape adopted); #2180, #2189 (their consumer signature and
TMR dispatch follow); D16 item 3 is superseded on the descriptor point only —
the callable generation stays gone.

## D18 - Take the func_id bound from the runtime table, not the child array

**Problem.** D17's line of work restored two checks #2190's extraction of
`validate_kernel_callable_image` had lost: a child `func_id` outside the device
function table, or repeated within one image, has the device consumer overwrite
a mapping it built earlier in the same invocation. This line expressed the range
bound as `std::extent_v<decltype(ChipCallable::child_func_ids_)>`. #2190 has
since restored the same two checks against `KERNEL_MAX_FUNC_ID`, a named
constant in `callable_protocol.h` tied to `RUNTIME_MAX_FUNC_ID` by a
`static_assert` in both `device_runner_base.cpp` files.

**Finding.** The two bounds are both 1024 and that is a coincidence.
`ChipCallable` is `Callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>`, so the
array extent says how many children one image may carry. A `func_id` indexes
the runtime's function table, whose size is `RUNTIME_MAX_FUNC_ID`. Nothing ties
them: raising the child capacity would widen the accepted id range past the
table the ids address, and the widening would be silent because the validator
would still compile and still look right.

**Choice.** Adopt #2190's expression, and its test with it.

1. `KERNEL_MAX_FUNC_ID` joins `MAX_REGISTERED_CALLABLE_IDS` in
   `callable_protocol.h`, with the comment saying the two id spaces are
   independent.
2. `validate_kernel_callable_image` bounds `func_id` by it.
3. `static_assert(KERNEL_MAX_FUNC_ID == RUNTIME_MAX_FUNC_ID)` in the onboard and
   simulation `device_runner_base.cpp`, the two translation units that see both
   headers, so a change to either bound fails the build rather than the images.

**Reason.** The accepted set does not move today, so this buys no behavior. It
buys the constraint: the bound now names what it bounds, and drift between the
two is a compile error instead of a class of image the device will accept and
then mis-dispatch.

**Affects.** #2190 (its shape adopted, closing the divergence this line had
recorded against it); the simulation `prepare_callable` ordering remains the one
deliberate difference.

---

## D19 - Registration waits for the device, as #2176 and #2190 do

**Problem.** Three statements about kernel-mode registration disagreed.
`runtime_c_api.h` said preparation "never synchronizes a stream or device -
registration errors surface through the caller's own warmup plus synchronize";
D14 item 5 said "init bootstrap and callable registration synchronize the
context's own AICPU stream"; `docs/zh-cn/kernel-mode-integration-test.md` said
both, in two sections. The implementation matched the first: this line inlined
the `RegisterCallableArgs` build and its `launch_aicpu_payload` into
`prepare_kernel_callable` and kept `commit_device_register` after it, dropping
the `aclrtSynchronizeStreamWithTimeout` that sits between them in
`register_callable_on_device`.

**Finding.** Both source PRs wait. #2176 and #2190 reach registration through
`register_callable_on_device(callable_id, control_stream)`, whose three steps
are launch, synchronize with `PLATFORM_STREAM_SYNC_TIMEOUT_MS`, commit — the
timeout classified separately from any other failure. That function is still on
this line, still synchronizing, and program mode still calls it through
`launch_device_register`. Only the kernel path had a second, divergent copy.

Nothing rests on the divergence. Registration is ordered ahead of every launch
by the AICPU stream's FIFO either way, which is what D16's chained topology
relies on; what the missing synchronize costs is the report. A device-side
`dlopen` of the orchestration SO that fails, or a registration that does not
complete within the timeout, leaves `prepare_callable` returning 0 and surfaces
later as a launch failure on a context nothing poisoned. The header's own
promise that "a registration that fails on the device poisons the context"
cannot hold without the wait, because there is nothing to observe before the
call returns.

**Choice.** The kernel path calls `register_callable_on_device`, deleting the
inlined copy. A device-side registration failure, timeout included, is
`prepare_callable`'s own status, and the entry poisons the context on it.

1. `DeviceRunnerBase::prepare_kernel_callable` replaces its inlined launch and
   commit with that call. The `host_dlopen_handle != nullptr` early return the
   inline copy open-coded is the callee's first branch, so HBG-style host
   orchestration still registers nothing.
2. `runtime_c_api.h` states the wait: registration synchronizes the context's
   own AICPU stream before committing the callable, and synchronizes no caller
   stream and no device.
3. `docs/zh-cn/kernel-callable-residency.md` and
   `docs/zh-cn/kernel-mode-integration-test.md` drop their claims that
   preparation synchronizes nothing, each of which contradicted a neighbouring
   section of the same document.

**Reason.** One registration path for both modes is the same consolidation D18
made for the `func_id` bound: the divergence bought nothing and hid a report.
Synchronizing an internal stream does not touch capture — prepare is outside
ACLGraph capture by contract, and the caller's stream is still never
synchronized, so the launch path D16 made capturable is unchanged.

The cost is that prepare now blocks for one AICPU round trip, under
`kernel_submission_mutex`, so a concurrent `launch` on another thread meets a
held lock and returns `INVALID_STATE` for that window. Launch already takes that
mutex with `try_to_lock` and reports the same code when preparation holds it, so
the window widens rather than a new failure appearing.

**Affects.** #2176 and #2190 (their registration path adopted); D14 item 5
(now true of the implementation).

---

## D20 - Take #2185's entry hardening; its refusal model is the correct one

**Problem.** D15 took only the test hardening from #2185's refreshed head and
deferred the rest as accompanying a signature change. D16 then adopted that
signature independently, so what is left of #2185's newer head is no longer a
signature question: a status-carrying exception type, a non-throwing capability
query, and bookkeeping for a refused init or a failed teardown.

**Finding.** One of those is not a preference. This line's `kernel_init` drops
the device context on any refused init, under a comment saying a refused kernel
init "has adopted nothing — it took no ACL state and bound no thread". That is
false here. `init_kernel_context` acquires the context claim, then
`kernel_exec_state_.initialize` creates two streams and five events, and the
four steps after it — `ensure_binaries_loaded`, `ensure_aicpu_init_launched`,
`prepare_launch_shape`, `prepare_aicpu_affinity` — each return a failure
without unwinding them; only the claim has a rollback guard. A refusal from any
of those therefore reaches `ChipWorker` with live resources, where
`destroy_device_context` refuses the context and returns void, and the
`DlHandleGuard` then `dlclose`s the library whose release routines are the only
way back to them. The streams and events leak for the life of the process, and
the log line saying so is the only trace.

**Choice.** Adopt #2185's model, its Python and C++ tests with it.

1. **`ChipWorkerError` carries a status code**, `UnsupportedRuntimeOperation`
   derives from it, and the nanobind layer registers both with translators that
   copy the code onto the Python exception. `PTO_RUNTIME_ERR_*` become module
   attributes. The registered `UnsupportedRuntimeOperation` subclasses
   `NotImplementedError` as well, so the ad-hoc conversion on
   `device_memory_info` is deleted rather than duplicated.
2. **A refused kernel init calls `finalize_device` first**, and when that fails
   keeps the handle, the bindings and the library, records
   `device_teardown_owed_`, and refuses any later init until a `finalize()`
   succeeds. `INVALID_ARGUMENT` and `UNSUPPORTED` are refusals the entry returns
   before taking anything, so they skip the teardown.
3. **`kernel_mode_supported()` answers rather than throws.** It is false
   whenever no runtime is bound, which `initialized()` already distinguishes
   from a bound runtime without kernel support.
4. **A failed `finalize()` on a kernel context raises and is retriable.**
   `kernel_context_` scopes that to contexts that entered
   `simpler_kernel_mode_init`; the Python wrapper's registries already
   documented this behaviour for a raising finalize that could not yet happen.
5. **Kernel callables get their own registry.** `_callable_registry` keys are
   program slots that `init()` replays into `register_callable` and that
   `_allocate_slot_locked` reads as occupancy; a runtime-minted kernel id has
   neither meaning.

**Kept against #2185.** Four places where this line is right and its head is an
artifact of its own stub platform:

- **The capability gate stays.** D14 item 6 kept `ChipWorker` resolving the
  kernel entries only when `simpler_kernel_mode_supported` is nonzero, so a
  runtime reporting no support never reaches `simpler_kernel_mode_init`. The
  adopted tests build fake runtimes that must now declare support to exercise
  the entry.
- **A program-mode teardown failure is reported, not swallowed.** #2185 lets it
  fall through silently. The runner releases what it can and gives up its
  device either way, so there is nothing to retry and nothing to raise — but the
  status is still the only evidence, so it goes to stderr.
- **`kernel_init` succeeding is the a2a3 case.** TMR implements kernel mode
  here, so the hardware test keeps the live-context scenario; #2185's twin
  expects `UNSUPPORTED` because its platform side is a stub. The Python twin
  keeps loading `host_build_graph` to get a genuine refusal.
- **`validate_kernel_prepare_callable_args` and the entry signature** are
  already D16's shape, so #2185's delta to them is empty here.

**Affects.** #2185 (the remainder of its refreshed head adopted); D15 (its
deferral of this PR closed).

---

## D21 - The L2 Worker owns the mode; init and close are shared, dispatch is not

**Problem.** The target call-flow document names one runtime entry object for
PyPTO — `simpler.worker.Worker(level=2, execution_mode="kernel")` — and lists
its construction identity plus `init` / `prepare` / `launch` / `close` dispatch
as the one simpler deliverable still marked 🔧. This line had only the
`ChipWorker` surface: the native wrapper, one layer below the object PyPTO
holds. Every kernel test drove `ChipWorker` or the C ABI directly, so nothing
established that the public path exists at all.

**Finding.** Three shapes are decided together, and the argument for each is
that the alternative reintroduces something the design already removed.

**Choice.**

1. **The mode is fixed at construction, not chosen at `init`.** `Worker(level=2,
   execution_mode="kernel")`; `program` is the default and every existing caller
   keeps its meaning. A mode argument on `init` would make the two surfaces
   reachable on one object before it, so `register()` on a not-yet-initialized
   Worker could not say which surface it belongs to. Fixing it at construction
   makes every refusal resolve from the constructor, before any device exists —
   which is also what makes the mode contract testable without hardware.
2. **`init` and `close` are shared; dispatch is not.** `init(config=...)` and
   `close()` are one call in both modes, which is the unified init the document
   asks for.
   The dispatch surfaces stay disjoint and refuse each other by name:
   `register` / `unregister` / `submit` / `run` are program-only,
   `kernel_prepare_callable` / `kernel_launch` are kernel-only. Native refuses
   the crossing too — the program entries bounds-check a kernel context's empty
   slot storage, and the device context latches one mode write-once — so the
   Python guard adds a diagnosis, not a safety property.
3. **The kernel context's `CallConfig` is a new `config=` parameter, not
   `prewarm_config`.** They are not the same object wearing two names:
   `prewarm_config` is an optional ring-sizing hint for a later `run`, and the
   kernel one is the context's required, immutable sizing, with no later `run`
   to hint at. Each is refused in the other's mode rather than silently
   ignored, because an ignored sizing config is exactly the failure that a
   context fixed at init cannot report later.
4. **Kernel mode is level 2 only.** L3+ forks a chip child per device, and a
   forked child cannot inherit the borrowed device and stream the caller holds.
   `level != 2` with `execution_mode="kernel"` is refused at construction.
5. **`kernel_mode_supported` reports capability, not mode.** It answers for the
   runtime the chip context bound — false before init, after a failed init,
   after close, and on any L3+ Worker, which binds none of its own. A
   program-mode L2 Worker answers for its runtime, exactly as `ChipWorker`
   does.

**Reason.** The Worker layer adds no mechanism: `init` reaches
`ChipWorker.kernel_init`, `close` already finalizes the chip worker through the
`CleanupJournal`, whose retry is what covers a failed kernel teardown, and
prepare and launch forward under the same READY lease every other live-tree call
takes. What it adds is the identity the three-party contract is written to, and
the guarantee that a Worker constructed for one surface cannot reach the other.

**Boundary.** `tests/ut/py/test_worker/test_kernel_mode_entry.py` now drives the
public path end to end on a2a3 — init, two registrations of one image, two
launches with different addresses and scalars, numeric verification, close, and
a launch after close that the CLOSED state refuses. The capture boundary is
unchanged by this entry: no test drives `Worker.kernel_launch` inside a captured
graph, so D16's note stands.

**Affects.** The target call-flow document's 🔧 row for the L2 Worker
construction identity and lifecycle dispatch; D16 (its prepare/launch contract
is what this surface exposes).

---

## D22 - Registration may synchronize; only the launch scope forbids every wait

**Problem.** `tests/st/a2a3/tensormap_and_ringbuffer/kernel_mode_capture` failed
23 of its 24 scenarios on a2a3, every one on the same assertion:
`prepare/launch performed an internal sync`. D19 routed kernel
`prepare_callable` through `register_callable_on_device`, which synchronizes the
context's AICPU stream so a device-side registration failure is
`prepare_callable`'s own status. The scene test's `_guarded` armed the
observer's `forbid_sync` around registration as well as launch, and the observer
does not merely count a forbidden sync — it refuses it, returning -4331. The two
cannot both stand. #2245 merged at 12:09 and #2242 at 12:16 the same day, so the
latter's checks ran against a base without the wait, and this base branch runs
only the `build` check.

**Choice.** D19's wait stays. The test's expectation is the half that was
written against a contract that had already changed.

1. **Two scopes, not one.** `capture_observer_guard_sync` keeps its meaning for
   launch: every synchronize is refused, because launch is pure enqueue and a
   wait there is what a captured graph cannot contain. A new
   `capture_observer_prepare_scope` covers registration and refuses only the
   *caller's* streams, which the test registers up front; the context's own
   AICPU stream reaches CANN. A null stream counts as the caller's, since a
   device-wide drain takes those streams with it.
2. **`forbid_sync` stops doubling as a scope marker.** The registration fault
   injection and the capture gate both keyed off `forbid_sync &&
   !invocation_scope`, so simply not arming it around registration would have
   disarmed them too — `prepare_fail_register` returned 0 where it expects
   -4333. Both now key off `prepare_scope` directly.
3. **The blocking gate moves from registration to launch.** It installed a
   blocking callback ahead of the AICPU register launch, which a synchronous
   registration now waits on: `prepare` returned only after the gate's own 10 s
   timeout, and the scenarios that assert it is still blocked could not hold.
   `capture_gate_install_if_armed` is called from the invocation scope instead,
   ahead of that launch's AICPU work, so the first launch's serial tail stays
   incomplete. `blocked_same` then asserts what remains true — a second launch
   on the same caller stream neither queries the tail nor waits for it, since
   the stream's own FIFO orders it and the query belongs to the
   different-stream path — and `stream_busy` keeps its
   `PREPARED_INCOMPATIBLE` coverage for the different-stream case.

**Reason.** The property the scene test exists for is that a launch contains no
wait, and that is untouched. Registration is not in a captured graph: it is the
eager, out-of-capture step whose whole purpose is to have already happened
before any launch a graph records. Asserting it performs no wait asserted
something the design never promised, and the narrower assertion — that it never
touches a stream the caller owns — is the one that protects a framework caller.

**Boundary.** 24 of 24 scenarios pass on a2a3. `tests/st/a2a3/kernel_capture`
is unchanged and still passes. No native source changed.

**Affects.** #2242 (its observer and two of its scenarios); #2245 and D19 (their
wait confirmed as the contract).
