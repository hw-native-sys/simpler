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
5. **Never use `std::this_thread::yield()` or `sched_yield()` in AICPU spin-wait loops.** On the Ascend AICPU, yielding to the OS scheduler introduces unacceptable latency for tight spin-waits (ticket locks, CAS retries, etc.). Use an empty loop body or a bare architecture hint (`__asm__ volatile("yield")`) instead.
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
   what can be renamed safely.
