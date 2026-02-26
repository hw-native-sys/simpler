# Platform Porting Guide

## When to Add a New Platform

Add a new platform backend when targeting a new hardware architecture or execution environment. The platform layer abstracts hardware-specific details behind shared interfaces.

## Architecture

```
src/platform/
├── include/           # Shared interface headers (platform-agnostic)
│   ├── host/
│   ├── aicpu/
│   ├── aicore/
│   └── common/
├── src/               # Shared source code
│   ├── host/
│   └── aicpu/
├── a2a3/              # Hardware backend (reference implementation)
│   ├── host/
│   ├── aicpu/
│   └── aicore/
└── a2a3sim/           # Simulation backend (reference implementation)
    ├── host/
    ├── aicpu/
    └── aicore/
```

## Steps to Add a New Platform

### 1. Create the platform directory

```
src/platform/<new_platform>/
├── host/              # Host runtime implementation
├── aicpu/             # AICPU scheduler implementation
└── aicore/            # AICore executor implementation
```

### 2. Implement the shared interfaces

Each component must implement the interfaces declared in `src/platform/include/`:

- **Host**: Device memory management, binary loading, kernel launching
- **AICPU**: Task scheduling, handshake protocol with AICore
- **AICore**: Task execution, kernel dispatch, completion signaling

Use `a2a3sim/` as a reference — it's the simplest implementation since it uses host threads.

### 3. Register the platform in the build system

Add the platform to `python/toolchain.py` so the build pipeline knows how to compile for it:

- Define compiler paths and flags
- Define linker settings for each component (host .so, aicpu .so, aicore .o)

### 4. Add platform detection to `python/runtime_builder.py`

The builder needs to know which compilers and flags to use for the new platform.

### 5. Test

```bash
# Run CI with the new platform
./ci.sh -p <new_platform>

# Run a single example
python examples/scripts/run_example.py \
    -k examples/host_build_graph/vector_example/kernels \
    -g examples/host_build_graph/vector_example/golden.py \
    -p <new_platform>
```

## Key Contracts

A platform backend must satisfy these contracts:

1. **Behavioral equivalence** — All platforms must produce the same computation results for the same input
2. **Handshake protocol** — AICPU-AICore coordination must follow the same protocol across platforms
3. **Memory layout** — Tensor memory layout must match the shared interface expectations
4. **API surface** — Implement all functions declared in `src/platform/include/`

## Common Pitfalls

- Forgetting to implement both `a2a3` and `a2a3sim` equivalents of a new API
- Memory alignment differences between platforms
- Endianness assumptions in handshake buffer serialization
- Thread safety in simulation (host threads share address space)
