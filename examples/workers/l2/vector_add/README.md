# `vector_add/` — one AIV kernel, end-to-end

Adds two 128×128 float32 tensors on the NPU and verifies the result against a
numpy reference. This is the smallest example that exercises the **full**
L2 Worker API: kernel compilation, `ChipCallable` assembly, device memory
management, `TaskArgs` construction, synchronous `run()`, and result readback.

If the equivalent `@scene_test` version feels magical, this is what the magic
unpacks into.

## Layout

```text
vector_add/
  main.py
  kernels/
    aiv/
      vector_add_kernel.cpp         # element-wise add kernel (reused verbatim
                                    #   from a2a3/../vector_example/kernels)
    orchestration/
      vector_add_orch.cpp           # simplified orchestration: single submit
  README.md
```

The kernel source is a verbatim copy of
`examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add.cpp`.
The orchestration is intentionally simpler than the `vector_example` one — it
calls `rt_submit_aiv_task` exactly once and has no nested scopes.

## The six steps in `main.py`

Open `main.py` and follow along.

### 1. Compile kernel sources → binary bytes

```python
kc = KernelCompiler(platform=args.platform)
kernel_bytes = kc.compile_incore(
    source_path=".../vector_add_kernel.cpp",
    core_type="aiv",
    pto_isa_root=ensure_pto_isa_root(),
    extra_include_dirs=kc.get_orchestration_include_dirs(runtime),
)
orch_bytes = kc.compile_orchestration(runtime, ".../vector_add_orch.cpp")
```

`compile_incore` picks the right toolchain for your platform (ccec on real
hardware, g++-15 on sim) and returns the `.o` / `.so` bytes. `pto_isa_root`
is the sibling header repo, auto-cloned on first call.

### 2. Wrap kernel bytes → `CoreCallable`

```python
core_callable = CoreCallable.build(
    signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
    binary=kernel_bytes,
)
```

The `signature` matches the kernel's I/O: two inputs, one output.

### 3. Wrap orchestration + children → `ChipCallable`

```python
chip_callable = ChipCallable.build(
    signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
    func_name="vector_add_orchestration",
    binary=orch_bytes,
    children=[(0, core_callable)],   # func_id=0 → our one kernel
)
```

`func_name` must match the `__attribute__((visibility("default")))` symbol in
the orchestration `.cpp`. The `children` list maps `func_id` integers used in
`rt_submit_aiv_task(func_id, ...)` to the corresponding `CoreCallable`.
In our orch `func_id=0` is the only value, so one child.

### 4. Allocate device memory, push inputs

```python
dev_a   = worker.malloc(NBYTES)
dev_b   = worker.malloc(NBYTES)
dev_out = worker.malloc(NBYTES)
worker.copy_to(dev_a, host_a.ctypes.data, NBYTES)
worker.copy_to(dev_b, host_b.ctypes.data, NBYTES)
```

`malloc` returns a device pointer (uint64). `copy_to(dst_dev, src_host, n)`
does host→device DMA.

### 5. Build `ChipStorageTaskArgs`, run

```python
args = ChipStorageTaskArgs()
args.add_tensor(ContinuousTensor.make(dev_a,   shape, DataType.FLOAT32))
args.add_tensor(ContinuousTensor.make(dev_b,   shape, DataType.FLOAT32))
args.add_tensor(ContinuousTensor.make(dev_out, shape, DataType.FLOAT32))

worker.run(chip_cid, args, CallConfig())  # chip_cid = worker.register(chip_callable) before init()
```

The tensor order must match `signature` order on the `ChipCallable`. `run()`
blocks until the kernel completes.

### 6. Pull result back, verify, free

```python
worker.copy_from(host_out.ctypes.data, dev_out, NBYTES)
worker.free(dev_a); worker.free(dev_b); worker.free(dev_out)
np.testing.assert_allclose(host_out, expected, rtol=1e-5, atol=1e-5)
```

## Run

```bash
python examples/workers/l2/vector_add/main.py -p a2a3sim -d 0
```

Expected output (sim):

```text
[vector_add] compiling kernels for a2a3sim...
[vector_add] compiled. binary_size=... bytes
[vector_add] init worker (device=0)...
[vector_add] running on device...
[vector_add] max |host_out - expected| = ...e-07
[vector_add] golden check PASSED ✅
```

First run will be slower because `ensure_pto_isa_root()` clones the PTO-ISA
header repo into `build/pto-isa/` (~few seconds on a fast connection).

## Why `@scene_test` still wins for real tests

The plumbing in `main.py` is ~100 lines. `@scene_test` does all of this for
you, plus:

- Session-level compile cache (kernel not recompiled per case)
- Parametrized `CASES` and auto-golden comparison
- Parallel device dispatch via pytest-xdist
- Integration with `--enable-l2-swimlane`, `--rounds`, `--dump-tensor`

Use the raw API when you're **learning** or **embedding** the runtime in a
larger Python program; use `@scene_test` for shippable test code.

## Next step

- Try `-p a2a3` if you have hardware access
- Move on to [`../../l3/multi_chip_dispatch/`](../../l3/multi_chip_dispatch/)
  to see how the same primitives compose into a multi-chip DAG
