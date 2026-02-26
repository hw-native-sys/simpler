# Runtime Layer

Three runtime variants provide different graph-building strategies:

| Variant | Graph built by | Use case |
|---------|---------------|----------|
| `host_build_graph` | Host CPU | Simple examples, full host control |
| `aicpu_build_graph` | AICPU on-device | Reduced host involvement |
| `tensormap_and_ringbuffer` | AICPU with shared memory | Advanced: tensor maps, ring buffers, multi-core orchestration |

## Structure

Each variant contains `host/`, `aicpu/`, and `aicore/` subdirectories with a `build_config.py` declaring include/source paths for the three compiled programs.

## Rules

1. Changes to one variant do not automatically apply to others — verify each independently
2. The runtime is selected via `RUNTIME_CONFIG.runtime` in each example's `kernel_config.py`
3. AICPU-AICore communication uses handshake buffers — see `.ai-instructions/coding/architecture.md`
4. Format with `clang-format -i <file>` before committing
