# a5-ir-check

The A5 device compiler has a stack-realignment defect
([simpler#2973](https://github.com/hw-native-sys/simpler/issues/2973)): a stack
object with a large alignment can get inconsistent address fixups once its
address escapes into an outlined call, so one `alloca` ends up with two runtime
addresses a few bytes apart and a predicate reads a shifted object. The source
looks correct, the build is clean, unit tests pass; only an A5 run shows it.

This tool finds that configuration without a device:

```bash
./a5_ir_check.py <build-dir>
```

Point it at the build directory of the configuration under review; the generator
is detected (Ninja via `ninja -t commands`, otherwise the verbatim commands in
`CMakeFiles/*/build.make`). AICore targets are `add_custom_command`, so they
never reach a `compile_commands.json` and the build tree is the only source of
their real command lines. Each translation unit is re-emitted as LLVM bitcode
and every stack object with `align >= 16` whose escaping address it can trace to
a call that is not `alwaysinline` is reported — including calls through a
function pointer, where `alwaysinline` cannot be proven. Exit status: 0 clean,
1 findings, 2 missing input/tool or an incomplete scan.

Two readings that matter:

- **A hit is a precondition, not a defect.** The object may never cross an
  outlined boundary at that build's level. Removing it or justifying it is fine;
  silence is not.
- **A clean run covers only the scanned configuration.** With upstream `inline`
  nothing escapes and nothing is reported, so run the scanner against the build
  the configuration under review actually produces.

## Why `-flto`

`ccec` refuses `-emit-llvm` and every `-mllvm -print-*` option on the device
side. LTO, however, must consume bitcode, so `-c -flto -o x.bc` produces valid
device bitcode (`target triple = "hiipu64-hisilicon-cce"`, `__gm__` as
`ptr addrspace(1)`), and a newer `llvm-dis` (17 reads clang-15 bitcode) turns it
back into text. No host replay, no synthetic TU.

## Verified behaviour

| input | result |
|---|---|
| A/B fixture, 5 variants (`alignas` 8/16/32 × ref/by-value/inline) | flags exactly the 3 that fail on hardware, clears the 2 that pass |
| real A5 `aicore` target, `--opt-level 0` | 48 findings in `aicore_executor.cpp`, all on `SchedulerTaskMetadata` (align 16) escaping into `scheduler_prepare_dispatch_slot`, `scheduler_publish_dispatch_slot`, `scheduler_task_requires_admission`, `scheduler_load_dispatch_metadata` |
| same target, `--opt-level 1` | 0 findings: at `-O1` everything is inlined, nothing crosses a boundary — which is why the shipped build is safe and the outlined build is not |

Runtime: `-O0` ≈ 2 s for the whole AICore target, `-O1` ≈ 6 min.

## Limits

- Escapes are traced through `getelementptr`, `bitcast` and `addrspacecast`
  only. A value routed through `phi` or `select` is not followed, so a hit is
  evidence and a clean run is not proof.
- A TU that fails to compile or disassemble is reported as `SKIP`, and any skip
  makes the exit status 2 rather than 0: a partial scan must not read as clean.
  Findings still win (status 1).
- Largest TU (`kernel.cpp`) at `-O1` makes system `llvm-dis` 17 crash while
  reading the vendor bitcode (crash frame only, no diagnostic); `-O0` works, and
  a vendor-matched `llvm-dis` 15 would be the fix. The other five TUs parse.
- The scan is over-approximate at `-O0`: escapes that `-O1` would promote or
  inline are all reported.
- The scanner cannot see the miscompile itself, only the shape that exposes it.
  A device functional claim still needs an A5 run.
## Knobs

Rarely needed — the defaults are what a review uses: `--opt-level {0,1,2,3}`
(override the build's own), `--extra-arg -D…`, `--min-align`, `--jobs`,
`--keep-ir`/`--ir-dir`, `--keep-debug`, `--ccec`, `--llvm-dis`, `--list`, `-v`.
