# Simpler AICPU Dispatcher SO

Source for `libsimpler_aicpu_dispatcher.so` — a transient bootstrap-only helper
loaded by CANN's preinstalled `libaicpu_extend_kernels.so`. Its only job is to
write the bundled runtime SO bytes to the main `aicpu_scheduler`'s preinstall
path under a content-fingerprint filename:

```text
/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>.so
```

The dispatcher SO itself is **never** persisted to disk and **never** dispatches
at per-task launch time. After bootstrap, the host launches the runtime SO
directly via `rtAicpuKernelLaunchExWithArgs` (kernel_type = `KERNEL_TYPE_AICPU`),
which routes through the main `aicpu_scheduler` and dlopens the preinstall file.

The source is runtime-agnostic, so it is built once and installed at
`build/lib/<arch>/onboard/<runtime>/libsimpler_aicpu_dispatcher.so` (a sibling
of each runtime's host_runtime.so). A single process binding multiple runtimes
shares one dispatcher SO on disk.

## Exported entry points

Three C-style symbols are exposed; `libaicpu_extend_kernels.so::SetTileFwkKernelMap`
dlsym's all three at load time, but only DynInit does real work:

1. `StaticTileFwkBackendKernelServer`       — stub
2. `DynTileFwkBackendKernelServerInit`      — bootstrap upload (real work)
3. `DynTileFwkBackendKernelServer`          — stub

See `aicpu_dispatcher.h` for the bootstrap protocol details (extended DeviceArgs
with `inner_so_bin`/`inner_so_len`, FNV-1a content fingerprint).
