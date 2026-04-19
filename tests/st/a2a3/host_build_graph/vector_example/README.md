# Vector Example — Task Dependency Graph (host_build_graph runtime)

Demonstrates building and executing a task dependency graph on Ascend hardware
(`a2a3`) and its simulation backend (`a2a3sim`).

The kernel computes `(a + b + 1)(a + b + 2)` as a four-task graph:

| Task | Expression |
| ---- | ---------- |
| 0 | `c = a + b` |
| 1 | `d = c + 1` |
| 2 | `e = c + 2` |
| 3 | `f = d * e` |

With `a=2.0`, `b=3.0`, `f = (2+3+1)(2+3+2) = 42.0`.

## Run it

```bash
# Simulation (no hardware)
python tests/st/a2a3/host_build_graph/vector_example/test_vector_example.py -p a2a3sim

# Hardware
python tests/st/a2a3/host_build_graph/vector_example/test_vector_example.py -p a2a3 -d 0

# Batch via pytest
pytest tests/st/a2a3/host_build_graph/vector_example --platform a2a3sim
```

`--log-level debug` enables verbose compile / runtime logs. `--build`
recompiles runtime C++ before running.

## Layout

```text
vector_example/
├── README.md
├── test_vector_example.py         # @scene_test class + __main__ entry
└── kernels/
    ├── aiv/
    │   ├── kernel_add.cpp         # element-wise addition
    │   ├── kernel_add_scalar.cpp  # add scalar to each element
    │   └── kernel_mul.cpp         # element-wise multiplication
    └── orchestration/
        └── example_orch.cpp       # task graph builder
```

Each kernel source is listed in `CALLABLE["incores"]` inside the test; the
orchestration source is listed in `CALLABLE["orchestration"]`.

## Simulation architecture

The `a2a3sim` backend emulates the AICPU/AICore execution model in-process:

- Kernel `.text` sections are `mmap`-ed executable
- Host threads emulate AICPU scheduling and AICore computation
- Memory uses regular host `malloc`/`free`
- Same C API as the real `a2a3` platform, so kernels are cross-compiled from
  the same sources (ccec → PTO ISA on hardware, g++ → plain C++ on sim)

## Troubleshooting

- **Kernel compile fails on `a2a3`:** PTO-ISA is auto-cloned on first run. If
  that fails, clone it manually to `build/pto-isa` or set
  `PTO_ISA_ROOT=/path/to/pto-isa` (see [docs/getting-started.md](../../../../docs/getting-started.md)).
- **Device init fails on `a2a3`:** check CANN env (`source
  $ASCEND_HOME_PATH/bin/setenv.bash`) and that the chosen `-d <id>` is free.
- **`"binary_data cannot be empty"`:** wrong `--platform`, missing kernel
  source, or compile error; rerun with `--log-level debug`.
- **Compile errors on `a2a3sim`:** gcc/g++ 15 is required; ensure it's on `PATH`.
