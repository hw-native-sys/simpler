# PTO Runtime - Task Runtime Execution Framework

Modular runtime for building and executing task dependency graphs on Ascend devices with coordinated AICPU and AICore execution. Three independently compiled programs (Host `.so`, AICPU `.so`, AICore `.o`) work together through clearly defined APIs.

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd simpler

# Install (venv recommended — see `.claude/rules/venv-isolation.md`)
pip install --no-build-isolation -e '.[test]'

# Run the vector example (simulation, no hardware required)
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py -p a2a3sim
```

PTO ISA headers are automatically cloned on first run. See [Getting Started](docs/getting-started.md) for manual setup and troubleshooting.

## Platforms

| Platform | Description | Requirements |
| -------- | ----------- | ------------ |
| `a2a3` | Real Ascend A2/A3 hardware | CANN toolkit (ccec, aarch64 cross-compiler) |
| `a2a3sim` | Thread-based A2/A3 simulation | gcc/g++ only (no Ascend SDK needed) |
| `a5` | Real Ascend A5 hardware | CANN toolkit (ccec, aarch64 cross-compiler) |
| `a5sim` | Thread-based A5 simulation | gcc/g++ only (no Ascend SDK needed) |

## Runtime Variants

Two runtimes under `src/{arch}/runtime/`, each with a different graph-building strategy:

| Runtime | Graph built on | Use case |
| ------- | -------------- | -------- |
| `host_build_graph` | Host CPU | Development, debugging |
| `tensormap_and_ringbuffer` | AICPU (device) | Production workloads |

See runtime docs per arch: [a2a3](src/a2a3/docs/runtimes.md), [a5](src/a5/docs/runtimes.md).

## Testing

```bash
# Simulation scene tests (no hardware)
pytest examples tests/st --platform a2a3sim

# Hardware scene tests (requires Ascend device)
pytest examples tests/st --platform a2a3 --device 4-7

# Python unit tests
pytest tests/ut -m "not requires_hardware" -v

# C++ unit tests
cmake -B tests/ut/cpp/build -S tests/ut/cpp && cmake --build tests/ut/cpp/build && ctest --test-dir tests/ut/cpp/build --output-on-failure
```

See [Testing Guide](docs/testing.md) for details.

## Environment Setup

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## Documentation

| Document | Description |
| -------- | ----------- |
| [Chip-Level Architecture](docs/chip-level-arch.md) | L2 single-chip: three-program model (host/AICPU/AICore), API layers, handshake protocol |
| [Hierarchical Level Runtime](docs/hierarchical_level_runtime.md) | L0–L6 level model, component composition (Orchestrator / Scheduler / Worker) |
| [Task Flow](docs/task-flow.md) | End-to-end data flow: Callable / TaskArgs / CallConfig handles, IWorker interface |
| [Orchestrator](docs/orchestrator.md) | DAG submission internals: submit flow, TensorMap, Scope, Ring, task state machine |
| [Scheduler](docs/scheduler.md) | DAG dispatch internals: wiring/ready/completion queues, dispatch loop |
| [Worker Manager](docs/worker-manager.md) | Worker pool, WorkerThread, THREAD/PROCESS modes, fork + mailbox mechanics |
| [Getting Started](docs/getting-started.md) | Setup, prerequisites, build process, configuration |
| [Developer Guide](docs/developer-guide.md) | Directory structure, role ownership, conventions |
| [Testing Guide](docs/testing.md) | CI pipeline, test types, writing new tests |

### Per-arch docs

| Document | a2a3 arch | a5 arch |
| -------- | --------- | ------- |
| Runtimes | [a2a3/docs/runtimes.md](src/a2a3/docs/runtimes.md) | [a5/docs/runtimes.md](src/a5/docs/runtimes.md) |
| Platform | [a2a3/docs/platform.md](src/a2a3/docs/platform.md) | [a5/docs/platform.md](src/a5/docs/platform.md) |

## License

This project is licensed under the **CANN Open Software License Agreement Version 2.0**. See the [LICENSE](LICENSE) file for the full license text.

## References

- [src/a2a3/platform/](src/a2a3/platform/) - Platform implementations
- [src/a2a3/runtime/](src/a2a3/runtime/) - Runtime implementations
- [examples/a2a3/](examples/a2a3/) - Examples organized by runtime
- [simpler_setup/](simpler_setup/) - SceneTestCase framework, runtime builder, kernel compiler
- [python/](python/) - Python bindings and user-facing runtime API
