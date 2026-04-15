# AICPU Dispatcher SO

Two-layer architecture for runtime-specific AICPU kernels.

## Architecture

The dispatcher SO provides a two-layer architecture where:

- **Outer layer (this SO)** is fixed and handles dynamic SO loading
- **Inner layer (runtime-specific SO)** can be different for each runtime

This allows different runtimes (tensormap, ringbuffer, etc.) to load their own AICPU kernel implementations at runtime without recompiling the dispatcher.

## Exported Functions

Three C-style exported functions (AICPU entry points):

1. `DynTileFwkKernelServerNull` - Load phase: receives inner SO binary, saves to filesystem
2. `DynTileFwkKernelServerInit` - Init phase: delegates to inner SO's initialization
3. `DynTileFwkKernelServer` - Run phase: delegates to inner SO's execution

## BackendServerHandleManager

Internal class that manages the lifecycle of the inner SO:

- `SaveSoFile()` - Saves inner SO binary to `/tmp/aicpu_kernels/`
- `SetTileFwkKernelMap()` - Loads init and run functions from inner SO using dlopen/dlsym
- `ExecuteFunc()` - Executes inner SO functions with provided arguments

## Function Key Mapping

- `dyInitFuncKey = 2` - Initialization function
- `dyExecFuncKey = 3` - Execution function

## Reference

Based on pypto's implementation:
`/data/fangjingzhi/pypto/framework/src/machine/device/machine_interface/pypto_aicpu_interface.{h,cpp}`
