# L2 Cache Bypass: Native CANN and simpler Delivery Paths

This document explains how an A2/A3 load marked for L2 bypass obtains the
device-specific nocache-alias offset, why the native AscendC/CANN mechanism
does not work for a simpler incore kernel, and how simpler transports the same
driver-owned value instead.

The runtime transport described here is implemented by
[PR #2323](https://github.com/hw-native-sys/simpler/pull/2323). The compiler
side must still consume `get_l2_cache_offset(args)` before
`pl.CachePolicy.BYPASS` is end-to-end usable on simpler; see
[PTOAS #1537](https://github.com/hw-native-sys/PTOAS/issues/1537).

## The two independent decisions

L2 bypass has two inputs with different owners:

| Input | Meaning | Owner | Lifetime |
| ----- | ------- | ----- | -------- |
| Load policy | Which individual GM loads should not allocate in L2 | frontend and PTOAS lowering | compiled into the kernel's `.text` |
| Nocache offset | How this device reaches the nocache alias of a GM address | driver and runtime | one value per device, fixed for a Worker lifetime |

The offset does **not** decide whether a load bypasses L2. PTOAS makes that
decision per load. The offset only supplies the target-specific address
transformation needed by loads already lowered as bypass loads.

It also does not vary per task or per incore kernel. Native CANN stores one
copy in every registered kernel image, but those copies contain the same
per-device value. simpler stores one copy in each core's `GlobalContext` and
shares it across every incore kernel dispatched to that core.

## Why adding an offset changes the cache policy

On devices that expose the A2/A3 mechanism, the driver maps the same physical
GM pages through two virtual-address windows:

```text
ordinary window: [base,          base + size)
nocache window:  [base + offset, base + offset + size)
```

`addr + offset` therefore names the same bytes through the nocache alias. It
is not an access to different physical data. A load through that alias uses the
not-allocate cache behavior, which avoids filling L2 with data that will not be
reused.

The offset is a driver-owned virtual-address-layout value. The runtime API is
deliberately keyed by physical device ID:

```cpp
rtError_t rtGetL2CacheOffset(uint32_t device_id, uint64_t *offset);
```

Code must query it rather than infer it from an architecture name or hardcode
a value. For example, the value observed during PR #2323 validation was
`0x80000000000`, while a pto-isa comment named `0x100000000000`. A static
constant would silently address the wrong window on at least one environment.

An offset of zero means that no alias is available. Adding zero preserves the
ordinary address, so lack of support or a failed query safely loses only the
optimization and leaves the load correct and cached.

### A2/A3 only

The offset exists in the A2/A3 trees alone. A2/A3's DMA load instruction does
not carry the cache policy as an operand, so pto-isa reaches the uncached
mapping by offsetting the address instead. A5's `TLOAD` does carry `l2Control`
as an instruction operand, and pto-isa's `TLoadSrcAddrWithL2Hint` returns the
address unchanged on every non-A2/A3 architecture — an A5 kernel could not
consume an offset it was handed. So `InitArgs`, `GlobalContext`, and the
accessor all stay A2/A3-only rather than mirroring a field no kernel reads.

## Native AscendC and CANN path

The native path is a three-party contract between PTOAS, pto-isa, and the CANN
runtime loader.

### Compile time

PTOAS lowers a frontend policy such as
`#pto.load_cache_policy<l2_bypass>` to a load resembling:

```cpp
TLOAD<pto::TLoadL2Hint::NotAllocKeep>(dst, src);
```

On A2/A3, pto-isa then emits three related pieces into the kernel ELF:

| ELF section | Content | Purpose |
| ----------- | ------- | ------- |
| `.text` | A bypass load reads `g_opL2CacheHintCfg.l2Cacheoffset` and adds it to its source address | Records which loads use the alias |
| `.data` | `g_opL2CacheHintCfg`, initialized to zero | Provides a slot for the loader-supplied value |
| `.ascend.meta` | Runtime implicit-information record `{4, 4, 3}`, where feature `3` is `L2CACHE` | Tells the CANN loader that the slot must be processed |

The metadata record does not mean "all loads in this kernel bypass L2." The
per-load lowering in `.text` carries that choice. The metadata only advertises
the presence of a runtime configuration slot. It can be emitted even when no
load ultimately uses the bypass policy.

### Binary registration

When CANN registers the complete ELF, its host-side loader:

1. parses `.ascend.meta` and recognizes the `L2CACHE` feature;
2. locates `g_opL2CacheHintCfg` through the ELF symbol table;
3. calls `rtGetL2CacheOffset(device_id, &offset)`;
4. writes the offset into the kernel image's `.data` slot in the host buffer;
5. uploads the complete patched image.

The query and patch occur before the AICore starts. The AICore cannot call the
driver: it has no host operating system, system-call path, or access to the
host runtime API.

### Kernel execution

At execution time a bypass load performs an ordinary device-memory read of the
already-patched slot, adds the value to its source address, and issues the load
against the nocache alias. A regular load leaves the address unchanged.

```text
PTOAS policy
    -> pto-isa emits .text + .data slot + .ascend.meta
    -> CANN loader parses, queries the driver, and patches the slot
    -> complete ELF is uploaded
    -> AICore load reads the slot and adds the offset
```

## Why that ABI breaks for simpler incore kernels

simpler's outer dispatcher and its dynamically dispatched incore kernels use
different loading mechanisms. The native CANN contract applies to the former,
not the latter.

| Property | CANN-registered kernel | simpler incore kernel |
| -------- | ---------------------- | --------------------- |
| Loader input | Complete linked ELF | Linked `.text` bytes only |
| ELF registration | CANN sees and registers the image | No CANN binary registration |
| `.ascend.meta` | Parsed | Discarded |
| `.data` | Patched and uploaded | Discarded |
| Entry | CANN launch | Runtime casts the payload address to `kernel(args)` |

[`extract_text_section`](../simpler_setup/elf_parser.py) deliberately extracts
only the linked `.text` payload. The broader loader contract is described in
[AICore Kernel Programming](aicore-kernel-programming.md#4-the-aicore-loader-runs-a-linked-text).

For `g_opL2CacheHintCfg`, this has two consequences:

1. CANN never sees the incore ELF, so it cannot parse the metadata, query the
   driver for this binary, or patch its symbol.
2. The initialized `__gm__` variable lives in `.data`, which is not copied to
   the device. The linked `.text` still contains a PC-relative reference to
   the expected location, so it reads unrelated device memory rather than the
   safe zero initializer.

The resulting address is `valid_addr + garbage`, which explains both observed
failure modes: silent wrong data when the accidental address is readable and
an AICore fault when it is not. Moving the metadata to the dispatcher would
not help; the dispatcher and each incore payload are separate images, and
CANN would patch the dispatcher's slot rather than storage belonging to the
incore payload.

This is also why the current design does not claim to fix initialized `__gm__`
globals in general. Full data-section loading is a separate loader problem.

## simpler runtime delivery path

simpler keeps the authoritative source, `rtGetL2CacheOffset`, but replaces
CANN's ELF-patching transport with the existing runtime context transport used
for other device configuration such as async-DMA workspaces.

```text
host: rtGetL2CacheOffset(device_id)
  -> InitArgs.l2_cache_offset
  -> resident AICPU configuration
  -> scheduler cold start
  -> per-core GlobalContext.l2_cache_offset
  -> args[SPMD_GLOBAL_CONTEXT_INDEX]
  -> get_l2_cache_offset(args)
  -> generated incore kernel's bypass-load lowering
```

The stages are:

1. [`DeviceRunner::fill_init_arch_fields`](../src/a2a3/platform/onboard/host/device_runner.cpp)
   queries the driver immediately before the one-time AICPU initialization.
   Unsupported devices and query errors are logged and represented as zero. It
   sits on the A2/A3 runner rather than the shared base because A5's `InitArgs`
   has no such field — the same split `kernel_args_init_ffts_base_addr` uses.
2. [`InitArgs`](../src/a2a3/platform/include/common/kernel_args.h) carries the
   value into `simpler_aicpu_init` once per Worker/device initialization.
3. The resident AICPU SO stores it through
   [`set_dev_l2_cache_offset`](../src/common/platform/include/aicpu/aicpu_device_config.h).
4. Each runtime's scheduler cold path copies the resident value into every
   core's `GlobalContext`, for AIC as well as AIV cores.
5. Each dispatch already passes that context through the fixed tail of
   `args[]`. The incore accessor
   [`get_l2_cache_offset(args)`](../src/a2a3/runtime/tensormap_and_ringbuffer/common/intrinsic.h)
   reads the field.

`args` is used because it is the existing channel that reaches a dynamically
loaded incore kernel, not because the value changes on every dispatch. The
query occurs once, the cold-start copy occurs once per core, and all later
tasks on that Worker read the same value.

Simulation needs no extra wiring. The shared resident configuration is
zero-initialized, no device query overwrites it, and `addr + 0` remains an
ordinary cached access.

### What holds the chain together

`tests/st/a2a3/tensormap_and_ringbuffer/spmd_basic` reads the accessor from all
three cores of a MIX task and records, per core, whether the offset it saw was
nonzero. Its onboard case expects every one of them — including the AIC — to see
one, which is what catches the cold-start copy being narrowed to the AIV pair
that `sub_block_id` walks. The sim case expects zero, pinning the `args` index
and the `GlobalContext` layout agreeing between the AICore-side and AICPU-side
headers. Neither case can assert the offset's value: it is whatever the driver
reports for that device.

## Compiler handoff still required

PR #2323 ends at `get_l2_cache_offset(args)`. It intentionally does not change
PTOAS or pto-isa, so the existing A2/A3 lowering still reads
`g_opL2CacheHintCfg` until the compiler stack adopts this contract.

The generated incore kernel must:

1. read `get_l2_cache_offset(args)` once at `kernel_entry`;
2. make that value available to the A2/A3 address-hint lowering, either by
   threading the local value through generated code or by initializing a
   block-local slot at entry;
3. add it only for loads whose compiled policy is a not-allocate policy; and
4. stop depending on the initialized `__gm__` global and its CANN-only
   `.ascend.meta` patch path for simpler incore payloads.

Conceptually, the generated code becomes:

```cpp
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    const uint64_t l2_cache_offset = get_l2_cache_offset(args);
    // ...
    // Only a load compiled with the bypass policy uses base + l2_cache_offset.
}
```

Reading once at entry matters: calling the accessor at every load would
re-read `GlobalContext` through GM even though the value cannot change during
the dispatch. A block-local handoff is compatible with simpler's text-only
payload because block-local storage is allocated by the AICore environment; it
is not an initialized `.data` section that the incore loader must upload.

Until this compiler handoff lands, the runtime accessor is available and has
been validated, but the original `pl.CachePolicy.BYPASS` lowering remains on
the incompatible CANN path.
