# a5 AICore SIMT Launch

How the a5 onboard path runs AICore kernels that contain SIMT vector
intrinsics (e.g. `mscatter`), and the two problems that had to be solved to
get there. SIMT is currently an a5-only path. For the chip tiers and the
three-program model this builds on, see
[a5 hardware.md](../src/a5/docs/hardware.md) and
[chip-level-arch.md](chip-level-arch.md).

## The setup

On a5 the registered AICore kernel is a thin **SU dispatcher**:
`KERNEL_ENTRY(aicore_kernel)` in
[onboard/aicore/kernel.cpp](../src/a5/platform/onboard/aicore/kernel.cpp) is a poll
loop that calls `aicore_execute`, which jumps into the real per-task kernel
code. Those per-task kernels — the ones that actually issue `mscatter` and
other SIMT ops — are compiled separately (one `.cpp` per `func_id`, via
`kernel_compiler.py`) into `CoreCallable` binaries and uploaded to device as
children of a `ChipCallable` buffer.

That split is what makes both problems below hard: the launch-time machinery
(the dispatcher ELF, the host launch call) never sees a SIMT instruction —
those live in the child binaries that are linked in only at runtime.

## Problem 1 — the compiler won't tag the dispatcher as SIMT

### What the hardware needs

The legacy launch path (`rtRegisterAllKernel` + `rtKernelLaunchWithHandleV2`)
reads two TLV records out of the kernel ELF's `.ascend.meta.<func>` NOTE
section at register time:

| TLV type | meaning | consumer |
| -------- | ------- | -------- |
| 7 (`COMPILER_ALLOC_UB_SIZE`) | UB (unified buffer) share-mem size | `Kernel::shareMemSize_` |
| 12 (`AIV_TYPE_FLAG`) | SIMD / SIMT classification | `Kernel::kernelVfType_` |

bisheng emits these **only when it statically sees a SIMT launch inside the
kernel**. Our dispatcher entry has none (the SIMT ops are in the task `.o`
files reached through `aicore_execute`), so left alone bisheng tags it
`NO_VF` / `shareMemSize=0` and the SIMT path won't run.

### The approach we rejected

The first attempt hand-wrote the two TLV records into the section and added
`-mllvm -cce-dyn-kernel-stack-size=false` to *suppress* bisheng's own
auto-emission so the hand-written bytes survived runtime parse. It worked but
was brittle: it couples us to the exact `.ascend.meta` byte layout, so any
bisheng change to that layout would silently desync the record. It also forced
us to manually pin `cfg.localMemorySize` to balance the UB budget (see below).

### The approach we took — a compiler-visible VF anchor

Reintroduce vector-unit usage the compiler *can* see, but never execute it, so
bisheng classifies and emits the record itself. The anchor deliberately uses
BOTH a SIMT launch and a SIMD op (see below for why). See
[onboard/aicore/simt_anchor.h](../src/a5/platform/onboard/aicore/simt_anchor.h):

```cpp
// `static` is load-bearing — see the "no extra global entry" property below.
static __simt_vf__ LAUNCH_BOUND(1024) __aicore__ void simt_meta_anchor_kernel(__gm__ uint32_t *sink) {
    sink[threadIdx.x] = threadIdx.x;
}
__simd_vf__ __aicore__ void simd_meta_anchor_kernel(__ubuf__ uint32_t *ub) {
    ub[0] = ub[0] + 1;
}
__attribute__((always_inline)) inline __aicore__ void simt_meta_anchor(__gm__ uint32_t *sink) {
    cce::async_invoke<simt_meta_anchor_kernel>(cce::dim3{1, 1, 1}, sink);
    simd_meta_anchor_kernel((__ubuf__ uint32_t *)0);
}
```

`KERNEL_ENTRY` calls it under an always-false guard (AIV only):

```cpp
#ifdef __DAV_VEC__
    if (k_args->force_simt_anchor) {           // always 0 at runtime
        simt_meta_anchor(reinterpret_cast<__gm__ uint32_t *>(k_args));
    }
#endif
```

These properties make it work and keep it cheap:

- **Built from cce compiler builtins, not pto-isa.** `cce::async_invoke`,
  `cce::dim3`, `__simt_vf__`, `LAUNCH_BOUND`, `threadIdx` come from bisheng's
  bundled `__clang_cce_*.h` headers (auto-included for `dav-c310-vec`).
  `mscatter` itself lowers to exactly these builtins — pto-isa is only a
  wrapper. So the a5 **platform runtime stays pto-isa-free**; only per-example
  incore kernels pull pto-isa (via `kernel_compiler.py`).
- **Survives dead-code elimination.** `force_simt_anchor` is a trailing field
  in the a5 `KernelArgs` ([kernel_args.h](../src/a5/platform/include/common/kernel_args.h))
  that the host always leaves 0. Because it is loaded from `__gm__` memory the
  compiler cannot fold the branch away, so the SIMT launch reaches codegen.
- **Tags the entry, not a helper.** The final link is `ld -r` (relocatable, not
  LTO), so cross-TU inlining needs the definition visible at the call site —
  hence a header with `always_inline`. Inlining lands the `async_invoke` inside
  `KERNEL_ENTRY`, so the SIMT tag attaches to the entry's `.ascend.meta`
  section, which is the one runtime reads.
- **AIV only.** The anchor is `#ifdef __DAV_VEC__`; AIC carries no SIMT and
  stays `NO_VF` naturally.
- **Must classify as MIX, not SIMT-only.** A SIMT launch alone makes bisheng
  emit `AIV_TYPE = SIMT_VF_ONLY (3)`. The shared SU dispatcher routes *both*
  SIMD and SIMT task `.o` files, and a SIMT-only tag makes runtime reject every
  SIMD launch through it — the whole a5 ST suite fails `107000` (param-invalid).
  The `__simd_vf__` companion adds a SIMD VF op so bisheng emits
  `SIMD_SIMT_MIX_VF (4)` instead. (`__simd_vf__` requires `__ubuf__` pointer
  args — SIMD runs on UB — so the companion takes a never-dereferenced UB
  pointer.)
- **No extra global entry.** Without `static`, `__simt_vf__` (cce_simt_entry)
  exports a GLOBAL `..._simt_entry` symbol. `rtRegisterAllKernel` registers the
  whole ELF, so that symbol becomes a *second* launchable kernel beside the real
  dispatcher entry — and that extra registration also fails the launch path with
  `107000`. `static` gives the anchor kernel internal linkage; it never needs to
  be launched (the branch is dead), only to exist at compile time, so the global
  symbol drops while the meta tag stays. This is the one structural difference
  from the old hand-written-bytes approach, which added no kernel code at all.

### Why this also removes the manual UB-budget tuning

Register-time check: `shareMemSize_ (TLV7) + cfg.localMemorySize ≤ 224 KB`
(256 KB UB − 32 KB dcache). With the record auto-emitted, bisheng computes a
shareMemSize (floor 8 KB) and the runtime auto-allocates AICore local memory
from what's left. The host launch path therefore sets **nothing** — `cfg` is
zero-initialized and `cfg.localMemorySize` stays 0, identical to a2a3
([device_runner_base.cpp](../src/common/platform/onboard/host/device_runner_base.cpp)
`launch_aicore_kernel`). No `aicore_local_memory_size()` virtual, no
`PLATFORM_AICORE_LOCAL_MEMORY_SIZE` constant.

### Verified

`readelf` of the linked AIV binary shows bisheng auto-emits
`.ascend.meta.aicore_kernel_0_mix_aiv` with **TLV7 = 0x2000 (8 KB)** and
**TLV12 = 4 (`SIMD_SIMT_MIX_VF`)**, and the only GLOBAL kernel entry symbols are
the two real dispatcher entries (no `..._simt_entry`); the AIC entry has no VF
type tag. On hardware the full `st-onboard-a5` suite passes — both `simt_basic`
and the non-SIMT workloads (bgemm, spmd, paged_attention, …) that the earlier
SIMT-only / extra-global-symbol states broke with `107000`.

## Problem 2 — the 16-byte alignment failure

This is the subtle one, and the one that caused intermittent, confusing
breakage.

### Where the kernel binary lands on device

A child kernel's binary code address on device is:

```text
chip_dev + offsetof(ChipCallable, storage_) + child_offset(i) + CoreCallable::binary_data_offset()
```

(see [chip_callable_layout.h](../src/common/task_interface/chip_callable_layout.h)
`patch_chip_callable_scratch_for_device`). Of the four terms:

- `chip_dev` — `rtMalloc` result, ≥512 B aligned (asserted ≥ `CALLABLE_ALIGN`).
- `child_offset(i)` — `callable_align_up` → multiple of `CALLABLE_ALIGN` (64).
- `binary_data_offset()` — rounds up to 64 → 128.

Three of the four are already multiples of 64, hence 16-aligned. **Only
`offsetof(ChipCallable, storage_)` decides the binary's final alignment.**

### Why it was misaligned

`ChipCallable`'s header is all `int32`/`uint32`/`char[]` fields
([callable.h](../src/common/task_interface/callable.h)), so the natural offset
of `storage_` is **8852 = 4 mod 8**. A SIMT instruction requires its operand /
code address aligned more strictly than the 4 bytes the all-uint32 header
yields — SIMT needs ≥8 B (and the next generation needs 16 B), versus SIMD's
4 B. Binding a `const CoreCallable&` to a 4-mod-8 address is also undefined
behaviour (the struct has a `uint64`), which UBSan flagged independently.

This is exactly why the SIMT path "used to work, then silently broke": its
correctness rode on whatever alignment `offsetof(storage_)` happened to have,
and nothing pinned it.

### How #979 "accidentally" fixed it — and why that wasn't enough

PR #979 (`align Callable.storage_ for 8-byte child structs`) added
`alignas(alignof(Child))` to `storage_` to fix the UBSan reference-binding
report. `alignof(CoreCallable)` is 8 (it has a `uint64`), so `offsetof`
moved **8852 → 8856 (8-aligned)**. That happened to satisfy the 8-byte SIMT
requirement of the time — so our SIMT change started passing again without us
touching it. But **8856 = 8 mod 16**: still only 8-byte aligned. The next SIMT
instruction generation needs 16-byte alignment, which 8856 would again fail.

### The fix — make it explicit and 16-byte

Replace the implicit `alignof(Child)` with a named constant
([arg_direction.h](../src/common/task_interface/arg_direction.h)):

```cpp
inline constexpr uint32_t CALLABLE_CHILD_ALIGN = 16;
...
alignas(CALLABLE_CHILD_ALIGN) char storage_[];   // callable.h
```

This moves `offsetof(storage_)` **8856 → 8864 (16-aligned)**, so the child
kernel binary lands 16-byte aligned. Two `static_assert`s lock the invariant:
`CALLABLE_CHILD_ALIGN >= alignof(CoreCallable)` (never under-align the child's
`uint64` reference — keeps #979's fix) and
`offsetof(ChipCallable, storage_) % CALLABLE_CHILD_ALIGN == 0`.

Cost is +8 bytes per `ChipCallable` (header padding only; children are already
64-B aligned, so no per-child padding is added), deduped once each — negligible.

### `offsetof(ChipCallable, storage_)` across versions

| state | alignment of `storage_` | `offsetof` | mod 16 | SIMT result |
| ----- | ----------------------- | ---------- | ------ | ----------- |
| pre-#979 | none | 8852 | 4 | 4-byte → misaligned, fails (and UB) |
| #979 | `alignof(CoreCallable)` = 8 | 8856 | 8 | 8-byte → passes by luck; 16-byte SIMT would fail |
| now | `CALLABLE_CHILD_ALIGN` = 16 | 8864 | 0 | 16-byte → robust |

## What we deliberately did not change

- **Host launch path** — unchanged from main; no manual `localMemorySize`.
- **No bisheng suppression flag** — `CMakeLists.txt` is back to the default
  build; bisheng's auto-emission is now what we rely on.
- **No pto-isa dependency in the platform runtime** — the anchor uses only cce
  compiler builtins.
