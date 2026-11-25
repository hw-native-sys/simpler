# Codestyle Rules

1. **Avoid plan-specific and process-narrative comments.** Do not write
   comments that describe the planning or the editing process rather than
   the code as it now stands — neither plan markers (`Phase 1`, `Step 1`,
   `Gap #3`) nor narration of the change that produced the line (`now uses
   the engine`, `removed the barrier`, `previously wmb'd`, `was 16384
   before`). A comment must state a present-tense fact about the code; see
   [comments.md](comments.md) for the full WHAT-not-history rule.

   **The same applies to commit messages.** A commit message describes what
   the change does and why it is correct — not the plan or step sequence
   that produced it. No `Phase 1 / Step 2 / Gap #3` framing, no "first did
   X, then changed to Y" working-process narration.
2. Use `enum class` preferentially for basic enumeration usage. Use `enum` only when implementing bitmask patterns or when bitwise operations are required.

    **Good:**

    ```cpp
    enum class CoreType : int { AIC = 0, AIV = 1 };
    CoreType type = CoreType::AIC;
    ```

    **Bad (unless implementing bitmask):**

    ```cpp
    enum CoreType { AIC = 0, AIV = 1 };  // Avoid this for basic enums
    ```

3. Prefer `volatile` decorator on struct members rather than volatile pointer casts unless necessary.
4. Avoid using pointer arithmetic with hardcoded offsets when `offsetof` is available.
5. **No sleeping on the dispatch path — at any tier.** A wait that a task's
   latency passes through may only ever *spin* or *block on a wakeup
   primitive* (semaphore, futex, condition variable, pipe/eventfd read).
   `sleep` and OS scheduler yields are forbidden there. Backoff timers
   are also forbidden except for the AICore polling case below. An OS
   sleep can delay a dispatch that becomes ready before the sleep ends.

   **AICore polling may use bounded cycle-based staggering.** Use it when
   an on-device A/B comparison on a representative workload shows a
   performance gain from reducing contention for a shared resource such
   as a SoC SRAM counter. The synchronization protocol must remain correct
   under any polling order.

   **Initialization and teardown paths are exempt** and may poll with a
   sleep — a startup handshake or a shutdown drain costs a one-off wait,
   not per-task latency. `_STARTUP_POLL_INTERVAL_S` in `worker.py` is the
   sanctioned shape.

   The boundary is *"does a task wait behind this?"*, not which file it is
   in. A mailbox poll waiting for `TASK_READY`, a completion poll waiting
   for `TASK_DONE`, a ready-queue wait, an AICPU ticket lock or CAS retry:
   all dispatch path. Registration, device bring-up, `init()` readiness
   barriers, `close()` reaping: all exempt.

   On the AICPU specifically, **never** `std::this_thread::yield()` or
   `sched_yield()` — yielding to the OS scheduler costs more than the wait
   it replaces. Use an empty loop body or a bare architecture hint
   (`__asm__ volatile("yield")`).

   When an idle spin is genuinely too expensive to leave running (a forked
   child that would hold a core for its whole lifetime), the answer is a
   blocking wakeup primitive, **not** a sleep — blocking costs no CPU and
   wakes in single-digit microseconds, where a sleep-based compromise pays
   latency on every dispatch to save CPU while idle.
6. **For cross-platform/platform-isolation preprocessor blocks, place the `__aarch64__` branch first.** Use this ordering pattern:

    ```cpp
    #if defined(__aarch64__)
    // aarch64 path (must be first)
    #elif defined(__x86_64__)
    // x86_64 path
    #else
    // other platforms
    #endif
    ```

7. **Never log on AICPU hot paths** (orchestrator / scheduler inner loops,
   per-task or per-scope code such as `submit_task` / `begin_scope` / the
   dispatch loop). AICPU `device_log` writes are expensive and serialize on the
   single AICPU op; flooding them — e.g. one `LOG_*` per scope_begin or per task
   — slows the op enough to trip the **op-execute timeout** (STARS/tsdaemon
   `HandleTaskTimeout` kills `aicpu-sd`), which *masks the very behavior you were
   trying to observe* and looks like a runtime hang. Gate any diagnostic to a
   high-water-mark (log only on a new max), a sample interval, or the
   cold/stall path — never unconditionally per iteration.

8. **Host code is C++; the host–device boundary is C/POD.** Host-side code
   (everything running on the host CPU — `src/{arch}/**/host/`, host-side
   runtime maker / orchestration, `simpler_setup/`, and the Python-adjacent
   tooling) must be written in modern C++, not C style. Use the STL
   (`std::vector`, `std::string`, `std::optional`, RAII, algorithms) to
   express intent; do not hand-roll C idioms where a standard facility
   exists. **Do not size fixed / static arrays to a worst case** — allocate
   to the actual size with a container. A multi-MB global dimensioned by a
   `MAX_*` constant is a defect, not a safety margin.

   The host↔device boundary is the sole exception, and it goes the other
   way: anything copied to the device, placed in shared memory, or uploaded
   (task descriptors, graph/definition images, ring/SM structs) must use **C
   data types in POD, contiguous storage** — trivially `memcpy`-able and
   position-independent. **Intra-image references must be offsets or indices
   from a block base, never raw pointers**: a wire struct has to survive a
   single `memcpy` to the device with zero pointer fix-up, so a `T*` that
   points inside the image is a defect even inside an otherwise trivially
   copyable struct (an absolute device address is acceptable only as an
   integer-typed field, not a host pointer). Back each wire struct with a
   compile-time guard — `static_assert(std::is_trivially_copyable_v<T> &&
   std::is_standard_layout_v<T>)` — as done for the device-copied structs in
   `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/runtime.h`. Keep the
   rich C++/STL representation host-only and compact it into the POD wire
   form at the boundary.

9. **`PTO2` is a legacy prefix — do not propagate it.** New code must not
   introduce the `PTO2` prefix on any identifier — types, functions,
   variables, macros, or file names; it is historical and carries no
   meaning. When you modify internal legacy code that uses it, remove the
   prefix as part of the change (rename the variable / macro / type / file)
   rather than leaving a mixed `PTO2Foo` / `Foo` surface behind, replacing
   the disambiguation it provided with **clear names or a `namespace`**.
   **Exempt externally-consumed names** where a blind rename would break a
   contract — public API, ABI / linked symbols, serialized or on-wire names,
   include paths, and host↔device struct layouts — unless you also ship a
   compatibility alias or a full migration. The ban on *new* `PTO2`
   identifiers is unconditional; removal of *existing* ones is scoped to
   what can be renamed safely. Rule 10 sets the *migration policy* for the
   existing surface — read it before starting any rename.

10. **Retire the legacy `PTO` naming incrementally — never in one sweep.**
    The project is **`simpler`** ("Simple Runtime"); its two runtimes are
    `host_build_graph` and `tensormap_and_ringbuffer`. **"PTO Runtime",
    "PTO Runtime2", and a bare "Runtime2" are not names of anything** — the
    `2` never denoted a version any reader can identify.

    **No `PTO2` identifier is left.** The last of them were the `PTO2_RING_*`
    ring-sizing environment variables, retired in favour of per-task
    `CallConfig.runtime_env`; `git grep -Io -E "PTO2|pto2_"` now finds only this
    rule's own examples, the warning that fires if a caller still exports a
    retired ring variable, prose recording that these names are gone, and
    `PTO2_MANUAL_MAX_SEQ` — a pypto-lib knob two example READMEs document and this
    repo never reads. **So a new `PTO2` match is a defect.**

    **The brand spellings are held by a hook, not by a count.**
    `tests/lint/check_retired_names.py` fails pre-commit on any new
    `PTO Runtime`, `PTO Runtime2`, or bare `Runtime2`. That hook exists because
    this paragraph once asserted the whole surface was empty while the rule's own
    grep found 38 banners across 36 files: the assertion had been measured with
    `PTO2|pto2_`, which cannot see a brand name with a space in it. Do not put a
    number here again — put it in the hook.

    Find the surface with the rule's own pattern, which is wider than the hook's
    on purpose (it also reaches the tiers below):

    ```bash
    grep -rIn -E "PTO Runtime2?|PTO2|pto2_|pto_runtime2" --include=* . | grep -v '^\./\.git/'
    ```

    Each match falls in exactly one tier, and the tier decides what you owe. Note
    that **the first two tiers share the `pto_*` namespace and have opposite
    risk** — that is the trap this table exists to spell out:

    | Tier | What | Policy |
    | ---- | ---- | ------ |
    | **A1 — brand prose** | The brand itself in docs, headings, skill descriptions, issue templates, and code comments (`# PTO Runtime2 Profiling Levels`, "the PTO Runtime consists of…") | **Fix on sight, unconditionally**, and the hook will make you. No compile risk, no contract, and no Tier-C name can hide in a file banner. Say `simpler`, or name the actual component ("the AICPU orchestrator", "the `tensormap_and_ringbuffer` runtime"). Follow the banner style already in the tree: `host_build_graph/host/orchestrator.cpp` opens `* host_build_graph orchestrator implementation`. |
    | **A2 — comments naming a `pto_*` that does not exist** | e.g. `pto_submit_task` (the entry is `rt_submit_task`), `pto_runtime2_init` (the work is `SchedulerState::init_data_from_layout`) | **Fix on sight**, but this is a [`doc-consistency.md`](doc-consistency.md) §3 defect, not a naming preference: a reader who greps the name finds nothing. Confirm the symbol is absent before renaming it — `git grep -w <name>` returning only comments is the test. |
    | **B — internal identifiers** | `PTO2Foo` types, `pto2_*` functions, internal `PTO2_*` macros and enumerators, `pto_*.h` / `pto_runtime2*.cpp` file names | Rename **only when you are already modifying that code**, per rule 9, and finish the identifier you started (below). |
    | **C — external contracts** | `runtime_env` knobs, `extern "C"` symbols in `runtime_c_api.h`, on-wire / serialized names, and **every live `pto_cpu_sim_*` / `pto_sim_*` hook** | **Exempt until a migration ships.** Renaming these breaks callers in other repos. The sim hooks are the worst case and the reason A2 above demands a grep first: `cpu_sim_context.cpp` exports them for "pto-isa via `dlsym(RTLD_DEFAULT)`" and `device_runner_base.cpp` fetches `pto_sim_register_hooks` by `dlsym`, so a rename produces **no compile error** — just a hook that is never found at run time. Ask the user before touching one; land it only with a compatibility alias or a coordinated cross-repo change. |

    An environment variable is the worst case in Tier C and the reason it
    exists: an unrecognised name is *ignored*, so the rename fails silently
    in production instead of at compile time. Three knobs proved the point.
    Two were found already dead — a benchmark setting `PTO2_SERIAL_ORCH_SCHED`
    against a runtime reading `SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE`, and an
    error hint naming `PTO2_SCHEDULER_TIMEOUT_MS` instead of
    `SIMPLER_SCHEDULER_TIMEOUT_MS`. The third case, `PTO2_RING_*`, is the one
    to copy: the knob was not renamed but **removed**, because
    `CallConfig.runtime_env` already carried the same sizing per task and was
    strictly more expressive. Retiring an env knob outright beats renaming it —
    and when you do, the runtime should say so on sight rather than silently
    fall back, which is what `warn_on_retired_ring_env()` in each
    `host/runtime_maker.cpp` is for.

    Three constraints make the difference between progress and churn:

    - **One identifier, all its occurrences, one commit.** A rename that
      leaves both spellings alive is worse than no rename — the reader now
      has to know both. The `src/{a2a3,a5}/runtime/{host_build_graph,
      tensormap_and_ringbuffer}/` trees are near-duplicates, so a Tier-B
      rename in one of them must land in its siblings in the same commit.
      Verify with the grep above returning zero hits for that identifier.
    - **The count only goes down**, and for the brand spellings
      `tests/lint/check_retired_names.py` enforces it. Introducing a new
      `PTO2`/`PTO Runtime` spelling anywhere — including by copying an existing
      `pto_*.h` to a new arch — is a defect, not neutral. Rule 9 makes this
      unconditional.
    - **Check the target name against three things, not one.** The bare name
      may be taken by the host orchestrator under `src/common/hierarchical/`
      (`TensorMap`, `ReadyQueue`, `TaskState`, `HeapRing`, `MAX_SCOPE_DEPTH`
      — rule 13 decides who keeps it, and the chip runtime takes `Chip` /
      `CHIP_`); by an external header the same translation units include
      (CANN defines `ALIGN_UP`); or, for an **object-like macro**, by any
      declaration anywhere, because a macro rewrites its identifier textually
      rather than colliding where the compiler can say so. Macros therefore
      keep a prefix even when the bare name looks free.

    Do **not** open a repo-wide mechanical rename PR: it collides with every
    in-flight branch, and a 6k-line diff cannot be reviewed for the handful
    of Tier-C names hiding in it.

11. **Prefer `pto::`-qualified names in kernels; let the tree converge, never
    sweep it.** Two conventions coexist in kernel sources today. The
    collectives kernels qualify (`pto::Stride<…>`) and carry no
    using-directive; the older scene-test kernels open with `using namespace
    pto;` and use bare names. Both are internally consistent, so neither is a
    defect on its own.

    **Qualification is the direction.** It is the only spelling that compiles
    regardless of whether a using-directive is in scope, and a file-scope
    `using namespace` is what most style guides steer away from. So:

    - **New kernels qualify, and do not add `using namespace pto;`.** Adding
      the directive to a file that does not have it is a step backwards.
    - **When you already edit a kernel, you may qualify it** — but see the
      constraint below, which is not optional.
    - **Existing files that use bare names are not defects.** Leave them
      unless you have another reason to touch them.

    **The constraint: qualify a file completely or not at all.** `Shape`,
    `Stride`, `Tile` and `GlobalTensor` all live in `namespace pto`, so
    qualifying one of them and leaving its siblings bare produces adjacent
    lines like

    ```cpp
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kTCols_, 1>;
    ```

    which reads worse than either consistent state. A half-qualified file is
    a defect where a fully bare one is not.

    **No repo-wide sweep.** As of 2026-07-28 the 142 files carrying `using
    namespace pto;` contain roughly 3,200 bare uses — `Tile` 2080,
    `GlobalTensor` 796, `Shape` 211, `Stride` 137. Rewriting them is a purely
    stylistic change that would touch every kernel in the repo, collide with
    every in-flight branch, and risk regex damage to member aliases and
    comments for no functional gain. If the tree is ever to be unified in one
    go, it needs its own decision plus a lint rule to hold the line — not a
    hand-edited PR.

12. **Use `#pragma once` for header guards, not `#ifndef`/`#define`/`#endif`.**

    ```cpp
    // Good
    #pragma once

    // Bad
    #ifndef SRC_A5_RUNTIME_FOO_H_
    #define SRC_A5_RUNTIME_FOO_H_
    ...
    #endif  // SRC_A5_RUNTIME_FOO_H_
    ```

    There is no compiler this project targets (gcc, clang, ccec, all C++17)
    that does not support it.  `#pragma once` is one line rather than three,
    cannot have the guard name disagree with the file's actual path, and does
    not leave a trailing `#endif` that can detach from its `#if` when the
    convertor ojects the `#ifndef` but misses the `#endif`.  Existing files
    are converted on sight during routine edits; a dedicated sweep is not
    required.

13. **Name runtime domain objects by role (`Worker` / `Chip` / `Core`), not
    by numeric topology level.** The hierarchy has three software naming
    contexts, and a type that crosses one of their boundaries must say which
    context owns its semantics:

    | Context | Naming | Examples |
    | ------- | ------ | -------- |
    | L3+ recursive scheduling | Unprefixed public domain name | `Worker`, `Buffer`, `Tensor`, `TaskArgs` |
    | L2 chip runtime | `Chip` prefix | `ChipWorker`, `ChipTensor`, `ChipTaskArgs`, `ChipCallable` |
    | L0 AIC/AIV task construction | `Core` prefix | `CoreCallable`, `CoreTaskArgs`, `CoreTaskPredicate` |

    Numeric levels belong in architecture prose, diagrams, and topology tables,
    not in software identifiers. Do not introduce names such as `L2Tensor`,
    `L2TaskArgs`, `L0TaskArgs`, `L3L2Queue`, or `L2Swimlane`: use the owning
    software entities instead (`ChipTensor`, `ChipTaskArgs`, `CoreTaskArgs`,
    `WorkerChipQueue`, `ChipSwimlane`). `L2` and `Chip` are not two naming
    dimensions — L2 is the topology position occupied by the Chip context, so
    retaining both spellings makes one concept look like two.

    A representation change at a boundary is also a type-name change. For
    example, the address-free `Tensor` carried by an L3+ `TaskArgs` is
    materialized by a `ChipWorker` into a GM-address-bearing `ChipTensor`; the
    two must not both be named `Tensor` merely because they share shape and
    dtype fields. A core task consumes that `ChipTensor` and therefore does not
    create a third `CoreTensor` concept.

    **Public Python and C++ types use the same canonical name and semantics.**
    Binding a C++ `Tensor` as Python `ChipTensor`, or maintaining a separate
    Python dataclass with the same public name as a C++ ABI type, is a naming
    defect. Bind the canonical C++ type directly where practical; codecs and
    compatibility aliases stay internal and must not create a second public
    meaning for the name.

    Existing numeric or mismatched identifiers are migrated when their API or
    module is already being changed. Finish each identifier across architecture
    siblings in one change; retain a compatibility alias only where an external
    source, ABI, or wire contract requires it. Do not open a repository-wide
    rename sweep.
