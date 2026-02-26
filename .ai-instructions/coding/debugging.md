# Debugging Guide

## Simulation Debugging

The `a2a3sim` platform runs all three programs (Host, AICPU, AICore) as threads in a single process, making it the easiest target for debugging.

### Enable verbose logging

```bash
python examples/scripts/run_example.py \
    -k <kernels> -g <golden.py> -p a2a3sim \
    --log-level debug
```

Log levels: `error`, `warn`, `info`, `debug`

### Using GDB with simulation

Since `a2a3sim` compiles to native code, you can attach GDB:

1. Build with debug symbols (add `-g` to compiler flags in `kernel_config.py` or runtime build)
2. Run under GDB:
   ```bash
   gdb --args python examples/scripts/run_example.py -k <kernels> -g <golden.py> -p a2a3sim
   ```
3. Set breakpoints in kernel or orchestration code

### Print debugging in kernels

For `a2a3sim`, standard `printf` works in AICPU and AICore code since they run as host threads. On real hardware, use the device logging macros instead.

## Hardware Debugging

### Host logs

Host logs are printed directly to the terminal. Control the log level with:

```bash
python examples/scripts/run_example.py \
    -k <kernels> -g <golden.py> -p a2a3 \
    --log-level debug
```

### Device logs

On Ascend hardware, device logs are written to:
```
~/ascend/log/debug/device-<id>/
```

Control the log level with:
```bash
export ASCEND_GLOBAL_LOG_LEVEL=0   # 0=error, 1=warn, 2=info, 3=debug
```

Use the device logging macros in AICPU/AICore code:
- `DEV_INFO` — Informational messages
- `DEV_DEBUG` — Debug messages
- `DEV_WARN` — Warnings
- `DEV_ERROR` — Error messages

## Golden Test Debugging

### Inspect tensor values

Add print statements to `compute_golden()` in `golden.py` to inspect intermediate values:

```python
def compute_golden(tensors, params):
    print(f"Input a shape: {tensors['a'].shape}, dtype: {tensors['a'].dtype}")
    print(f"Input a sample: {tensors['a'][:5]}")
    tensors["out_c"].copy_(tensors["a"] + tensors["b"])
    print(f"Output c sample: {tensors['out_c'][:5]}")
```

### Tolerance tuning

If golden comparison fails with small numerical differences, adjust tolerances in `golden.py`:

```python
RTOL = 1e-3   # Relative tolerance (default: 1e-5)
ATOL = 1e-3   # Absolute tolerance (default: 1e-5)
```

## Build Debugging

### Inspect compiled output

```bash
# Check ELF sections of compiled objects
readelf -S <output.o>

# Check symbol table
nm <output.so>

# Disassemble
objdump -d <output.o>
```

### Compilation pipeline trace

The Python build modules (`python/runtime_builder.py`, `python/kernel_compiler.py`) print compilation commands when verbose logging is enabled. Check these commands to verify correct flags and paths.
