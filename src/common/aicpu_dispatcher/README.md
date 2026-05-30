# Simpler AICPU Dispatcher SO

Source for `libsimpler_aicpu_dispatcher.so` — a transient bootstrap-only helper
loaded by CANN's preinstalled `libaicpu_extend_kernels.so`. Its only job is to
write the bundled runtime SO bytes to the main `aicpu_scheduler`'s preinstall
path under a content-fingerprint + device-id filename:

```text
/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<device_id>.so
```

The `<device_id>` suffix isolates the paired dies of one a2a3 chip (which share
the preinstall filesystem) so they never write/rename/execute one shared file —
concurrent bootstrap on a shared file corrupted the mmap'd image and faulted
`simpler_aicpu_exec` (507018 → chip fault → 507899 cascade).

The dispatcher SO itself is **never** persisted to disk and **never** dispatches
at per-task launch time. After bootstrap, the host registers the preinstall
file via `rtsBinaryLoadFromFile` (JSON load, cpuKernelMode=0) and
resolves `simpler_aicpu_init` / `simpler_aicpu_exec` once via
`rtsFuncGetByName`; per-task launches go through `rtsLaunchCpuKernel` on the
cached `rtFuncHandle`s. The main `aicpu_scheduler` owns the dlopen of the
preinstall file; the dispatcher is out of the picture once bootstrap returns.

The source is runtime-agnostic. It is built per-arch under
`build/lib/<arch>/onboard/<runtime>/libsimpler_aicpu_dispatcher.so` as a
sibling of each runtime's host_runtime.so. A single process binding multiple
runtimes can share one dispatcher SO on disk; the host process-level
fingerprint cache deduplicates bootstrap calls by inner-SO Build-ID.

## Exported entry points

Three C-style symbols are exposed; `libaicpu_extend_kernels.so::SetTileFwkKernelMap`
dlsym's all three at load time, but only DynInit does real work:

1. `StaticTileFwkBackendKernelServer`       — stub
2. `DynTileFwkBackendKernelServerInit`      — bootstrap upload (real work)
3. `DynTileFwkBackendKernelServer`          — stub

See `aicpu_dispatcher.h` for the bootstrap protocol details (extended DeviceArgs
with `inner_so_bin`/`inner_so_len`, FNV-1a content fingerprint).
