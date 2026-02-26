# Troubleshooting Guide

## Compilation Errors

### ccec compiler not found
```
Error: ccec: command not found
```
- You need the CANN toolkit installed for hardware platform (`a2a3`)
- Set up the CANN environment: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- For simulation (`a2a3sim`), `ccec` is not needed — only `g++`

### aarch64 cross-compiler not found
```
Error: aarch64-linux-gnu-g++: command not found
```
- Install the aarch64 cross-compilation toolchain for AICPU compilation on hardware
- For simulation (`a2a3sim`), host `g++` is used instead — no cross-compiler needed

### PTO_ISA_ROOT not set
```
Error: PTO_ISA_ROOT environment variable is not set
```
- The test framework auto-clones pto-isa on first run to `examples/scripts/_deps/pto-isa`
- If auto-clone fails, clone manually:
  ```bash
  mkdir -p examples/scripts/_deps
  git clone --branch master https://gitcode.com/cann/pto-isa.git examples/scripts/_deps/pto-isa
  ```
- Or set the path explicitly: `export PTO_ISA_ROOT=/path/to/pto-isa`

### Linker errors with undefined symbols
- Ensure all three components (host, aicpu, aicore) are compiled with matching runtime variant
- Check that `RUNTIME_CONFIG.runtime` in `kernel_config.py` matches the runtime directory

## Test Failures

### Golden test mismatch
```
FAIL: Output mismatch for tensor 'out_c'
```
- Check `compute_golden()` implements the correct reference computation
- Verify tolerance settings: add `RTOL` and `ATOL` to `golden.py` if needed (default: `1e-5`)
- Ensure data types match between golden and kernel (e.g., `float16` vs `float32`)

### Simulation hangs or deadlocks
- Check handshake buffer logic — AICPU and AICore must follow the handshake protocol
- Verify task dependencies are acyclic (no circular `fanout` references)
- Check `aicpu_thread_num` and `block_dim` in `RUNTIME_CONFIG` — they must be consistent

### Device test fails but simulation passes
- Hardware has stricter memory alignment requirements
- Check for uninitialized memory reads (simulation may zero-fill, hardware won't)
- Verify tensor sizes are aligned to hardware requirements

## Profiling Issues

### No perf_swimlane_*.json generated
- Ensure `--enable-profiling` flag was passed to `run_example.py`
- Check that the test passed — profiling data is only written on success
- Verify the `outputs/` directory exists

### Perfetto shows empty trace
- Ensure `merged_swimlane_*.json` was generated (check `outputs/`)
- Try regenerating: `python3 tools/swimlane_converter.py outputs/perf_swimlane_*.json`
- Verify the JSON file is valid (not truncated or corrupt)

## CI Issues

### Simulation CI fails on macOS
- Ensure `g++` is installed (Xcode command line tools or Homebrew)
- macOS uses `clang++` by default — verify `g++` points to a real GNU compiler if needed

### Device CI fails with "device not found"
- Verify device IDs are correct: `./ci.sh -p a2a3 -d 4-7`
- Check that the Ascend driver is loaded: `npu-smi info`
- Ensure the CANN toolkit version matches the driver version

## Common Mistakes

### func_id mismatch
The `func_id` in `kernel_config.py` must match what the orchestration code uses to reference kernels. A mismatch causes the wrong kernel to execute.

### Missing output tensor declaration
Output tensors must be declared either by:
- Adding `__outputs__ = ["tensor_name"]` to `golden.py`, or
- Using the `out_` prefix convention (any tensor named `out_*` is treated as output)

### Wrong runtime variant
If `RUNTIME_CONFIG.runtime` doesn't match the runtime directory where the example lives, compilation will fail or produce incorrect results.
