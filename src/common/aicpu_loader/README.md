# Simpler AICPU Dispatcher SO

Source for `libsimpler_aicpu_dispatcher.so` — a transient bootstrap-only helper
loaded by CANN's preinstalled `libaicpu_extend_kernels.so`. Its only job is to
write the bundled runtime SO bytes to the main `aicpu_scheduler`'s preinstall
process directory under a content-fingerprint + device-id filename:

```text
/usr/lib64/aicpu_kernels/<vf>/aicpu_kernels_device/<hostPid>_<localDeviceId>/simpler_inner_<fp>_<device_id>.so
```

The `<device_id>` suffix isolates the paired dies of one a2a3 chip (which share
the preinstall filesystem) so they never write/rename/execute one shared file —
concurrent bootstrap on a shared file corrupted the mmap'd image and faulted
`simpler_aicpu_exec` (507018 → chip fault → 507899 cascade).

CANN manages loading the dispatcher SO during bootstrap; the dispatcher does
not run at per-task launch time. After bootstrap, the host registers the preinstall
file via `rtsBinaryLoadFromFile` (JSON load, cpuKernelMode=0) and
resolves every symbol the runtime declares it exports via
`rtsFuncGetByName` (a miss is a build/registration error, not an optional
gap); per-task launches go through `rtsLaunchCpuKernel` on the
cached `rtFuncHandle`. The main `aicpu_scheduler` owns the dlopen of the
preinstall file; the dispatcher is out of the picture once bootstrap returns.

The source is runtime-agnostic. It is built per-arch under
`build/lib/<arch>/dispatcher/libsimpler_aicpu_dispatcher.so`, one per arch and
shared across every runtime. A single process binding multiple runtimes
therefore shares one dispatcher SO on disk. Each runner uploads into CANN's
`<hostPid>_<localDeviceId>` directory, including after device reset. CANN removes
that directory at process-session teardown.

Historical files in the shared root remain untouched. CANN can still select
those files first; recovering incorrect historical content is outside this
change.

## Argument ABI boundaries

The dispatcher bootstrap uses a private `DeviceArgs` ABI. Host-side
`LoadAicpuOp::BootstrapDispatcher()` writes a `device_args_ptr` into the
bootstrap `KernelArgs` front slot at offset 40. The device-side dispatcher
reads that pointer as an extended `DeviceArgs` object:

```text
DeviceArgs + 96:  aicpu_so_bin
DeviceArgs + 104: aicpu_so_len
DeviceArgs + 112: device_id
DeviceArgs + 120: inner_so_bin
DeviceArgs + 128: inner_so_len
```

This private bootstrap structure is distinct from the platform runtime
`KernelArgs` payload used by per-task launches. Per-task AICPU launch passes
the front-less `KernelArgs` payload directly to `rtsLaunchCpuKernel` (no CANN
launch front): `runtime_args` is at offset 0, followed by profiling/logging
fields and register tables. `device_id` is not in this payload — it reaches
the device once via `InitArgs`. AICore receives only the same
front-less `KernelArgs` via a host-owned device copy.

## Exported entry points

Three C-style symbols are exposed; `libaicpu_extend_kernels.so::SetTileFwkKernelMap`
dlsym's all three at load time, but only DynInit does real work:

1. `StaticTileFwkBackendKernelServer`       — stub
2. `DynTileFwkBackendKernelServerInit`      — bootstrap upload (real work)
3. `DynTileFwkBackendKernelServer`          — stub

See `aicpu_dispatcher.h` for the bootstrap protocol details (extended DeviceArgs
with `inner_so_bin`/`inner_so_len`, Build-ID-derived fingerprint).

## See also

- [`docs/aicpu-kernel-launch-mechanisms.md`](../../../docs/aicpu-kernel-launch-mechanisms.md) —
  this dispatcher is the heart of "Method 2 (Path A)". That doc compares
  it against the older tar.gz method and the broken Path B
  (`KERNEL_TYPE_AICPU_CUSTOM`), and records the four failed user-space
  workarounds from issue #822 so future readers don't re-derive them.
- [`tools/cann-examples/aicpu-kernel-launch/`](../../../tools/cann-examples/aicpu-kernel-launch/) —
  standalone reference tool implementing the dispatcher bootstrap with
  the minimum possible inner kernel; copy that as a template for new
  AICPU work.
