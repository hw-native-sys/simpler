# PTO Runtime Test Framework

This directory contains scripts and tools for running PTO Runtime tests.

## Overview

The PTO Runtime test framework provides a simplified interface for testing runtime implementations. Users only need to provide:

1. **Kernel configuration** (`kernel_config.py`) - Defines kernels and orchestration function
2. **Golden script** (`golden.py`) - Defines input generation and expected output computation

The test framework automatically handles compilation, execution, and result validation.

## Quick Start

### Basic Usage

```bash
python examples/scripts/run_example.py \
  --kernels <kernels_directory> \
  --golden <golden_script_path> \
  --platform <platform_name>
```

### Examples

#### Running Hardware Platform Tests (Requires Ascend Device)

```bash
python examples/scripts/run_example.py \
  -k examples/host_build_graph/vector_example/kernels \
  -g examples/host_build_graph/vector_example/golden.py \
  -p a2a3
```

#### Running Simulation Platform Tests (No Hardware Required)

```bash
python examples/scripts/run_example.py \
  -k examples/host_build_graph/vector_example/kernels \
  -g examples/host_build_graph/vector_example/golden.py \
  -p a2a3sim
```

## Command Line Arguments

### `run_example.py` Parameters

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--kernels` | `-k` | Kernels directory path (contains kernel_config.py) | **Required** |
| `--golden` | `-g` | golden.py script path | **Required** |
| `--platform` | `-p` | Platform name: `a2a3` or `a2a3sim` | `a2a3` |
| `--device` | `-d` | Device ID | From env var or 0 |
| `--runtime` | `-r` | Runtime implementation name | `host_build_graph` |
| `--verbose` | `-v` | Enable verbose output (equivalent to `--log-level debug`) | False |
| `--silent` | | Enable silent mode (equivalent to `--log-level error`) | False |
| `--log-level` | | Set log level: `error`, `warn`, `info`, `debug` | `info` |

### Platform Description

- **`a2a3`**: Hardware platform, requires Ascend device and CANN toolkit
- **`a2a3sim`**: Simulation platform, uses thread simulation, only requires gcc/g++

## Logging Control

The test framework supports unified logging across Python and C++ Host code with four levels:

### Log Levels

- **`error`** (ERROR): Only show errors, suitable for CI/CD or production
- **`warn`** (WARNING): Show warnings and errors
- **`info`** (INFO, default): Show progress and status information
- **`debug`** (DEBUG): Show detailed debug information including:
  - Compiler commands
  - Compiler stdout/stderr output
  - Detailed tensor data
  - Intermediate step information

### Usage Examples

```bash
# Error level - only show errors
python examples/scripts/run_example.py -k ./kernels -g ./golden.py --silent
# or explicitly:
python examples/scripts/run_example.py -k ./kernels -g ./golden.py --log-level error

# Info level (default) - show progress
python examples/scripts/run_example.py -k ./kernels -g ./golden.py

# Debug level - show all debug info
python examples/scripts/run_example.py -k ./kernels -g ./golden.py --verbose
# or explicitly:
python examples/scripts/run_example.py -k ./kernels -g ./golden.py --log-level debug

# Warning level - show warnings and errors
python examples/scripts/run_example.py -k ./kernels -g ./golden.py --log-level warn
```

### Environment Variable

You can also control logging via environment variable (lower priority than CLI arguments):

```bash
export PTO_LOG_LEVEL=debug  # Options: error, warn, info, debug
python examples/scripts/run_example.py -k ./kernels -g ./golden.py
```

### Priority

Log level is determined by (highest to lowest priority):
1. CLI arguments (`--log-level`, `--verbose`, `--silent`)
2. Environment variable (`PTO_LOG_LEVEL`)
3. Default value (`info` / INFO level)

### Compiler Output Behavior

- **error/warn/info levels**: Compiler output hidden unless compilation fails
- **debug level**: All compiler output displayed via DEBUG level
- **Compilation failures**: Always show error messages regardless of log level

## File Structure Requirements

### 1. Kernels Directory Structure

The kernels directory must contain a `kernel_config.py` file:

```
kernels/
├── kernel_config.py          # Required: kernel configuration
├── orchestration/
│   └── example_orch.cpp      # Orchestration function implementation
└── aiv/                      # or aic/, depending on core type
    ├── kernel_add.cpp
    ├── kernel_mul.cpp
    └── ...
```

### 2. `kernel_config.py` Format

```python
from pathlib import Path

KERNELS_DIR = Path(__file__).parent

# Kernel list
KERNELS = [
    {
        "func_id": 0,                                      # Kernel ID
        "core_type": "aiv",                                # Core type: aiv or aic
        "source": str(KERNELS_DIR / "aiv/kernel_add.cpp") # Kernel source file path
    },
    # More kernels...
]

# Orchestration function configuration
ORCHESTRATION = {
    "source": str(KERNELS_DIR / "orchestration/example_orch.cpp"),
    "function_name": "BuildExampleGraph"  # Orchestration function name
}
```

### 3. `golden.py` Format

```python
import torch

# Output tensor names list (optional, or use 'out_' prefix convention)
__outputs__ = ["f"]

# Tensor order (required, must match orchestration function parameter order)
TENSOR_ORDER = ["a", "b", "f"]

# Comparison tolerances
RTOL = 1e-5
ATOL = 1e-5

def generate_inputs(params: dict) -> dict:
    """
    Generate input and output tensors.

    Args:
        params: Parameter dictionary (from ALL_CASES)

    Returns:
        Dictionary containing all tensors (inputs + outputs)
    """
    SIZE = 16384
    return {
        "a": torch.full((SIZE,), 2.0, dtype=torch.float32),
        "b": torch.full((SIZE,), 3.0, dtype=torch.float32),
        "f": torch.zeros(SIZE, dtype=torch.float32),  # Output tensor
    }

def compute_golden(tensors: dict, params: dict) -> None:
    """
    Compute expected output (in-place modification).

    Args:
        tensors: Dictionary containing all tensors
        params: Parameter dictionary
    """
    a = tensors["a"]
    b = tensors["b"]
    tensors["f"][:] = (a + b + 1) * (a + b + 2)

# Optional: Multiple test cases
ALL_CASES = {
    "Default": {},
    # "Large": {"size": 1024},  # Other test cases
}
DEFAULT_CASE = "Default"
```

### Golden Script Interface Description

#### Required Functions

1. **`generate_inputs(params: dict) -> dict`**
   - Generate input and output tensors
   - Returns: Dictionary with tensor names as keys and torch tensors as values

2. **`compute_golden(tensors: dict, params: dict) -> None`**
   - Compute expected output values
   - Modifies output tensors in `tensors` dictionary in-place

#### Required Configuration

- **`TENSOR_ORDER`**: List specifying the order of tensors passed to orchestration function

#### Optional Configuration

- **`__outputs__`**: Output tensor names list (or use `out_` prefix convention)
- **`ALL_CASES`**: Dict of named parameter sets for parameterized tests
- **`DEFAULT_CASE`**: Name of the default case to run
- **`RTOL`**: Relative tolerance (default `1e-5`)
- **`ATOL`**: Absolute tolerance (default `1e-5`)

## Output Tensor Identification

The test framework supports two methods for identifying output tensors:

### Method 1: Explicit Declaration (Recommended)

```python
__outputs__ = ["f", "result", ...]
```

### Method 2: Naming Convention

Use `out_` prefix to name output tensors:

```python
def generate_inputs(params: dict) -> dict:
    return {
        "a": torch.randn(1024),      # Input
        "b": torch.randn(1024),      # Input
        "out_f": torch.zeros(1024),  # Output (auto-detected)
    }
```

## Orchestration Function Interface

The orchestration function's parameter order must match `TENSOR_ORDER`:

```cpp
// Assume TENSOR_ORDER = ["a", "b", "f"]
int BuildExampleGraph(Runtime* runtime, uint64_t* args, int arg_count) {
    // args layout: [ptr_a, ptr_b, ptr_f, size_a, size_b, size_f, count]
    void* ptr_a = reinterpret_cast<void*>(args[0]);
    void* ptr_b = reinterpret_cast<void*>(args[1]);
    void* ptr_f = reinterpret_cast<void*>(args[2]);
    uint64_t size_a = args[3];
    uint64_t size_b = args[4];
    uint64_t size_f = args[5];
    uint64_t count = args[6];

    // Build task graph...
}
```

## Environment Variables

### Logging Configuration (All Platforms)

```bash
# Set log level (optional, CLI arguments take priority)
export PTO_LOG_LEVEL=debug  # Options: error, warn, info, debug

# Optional: Output C++ Host logs to file
export PTO_LOG_FILE=/tmp/pto_runtime.log
```

### a2a3 Platform (Hardware)

```bash
# Required
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0

# PTO_ISA_ROOT is auto-detected (auto-cloned to examples/scripts/_deps/pto-isa on first run)
# Override if needed:
# export PTO_ISA_ROOT=/path/to/pto-isa

# Optional
export PTO_DEVICE_ID=0
```

### a2a3sim Platform (Simulation)

No special platform-specific environment variables required.

## Complete Example

### Directory Structure

```
my_test/
├── kernels/
│   ├── kernel_config.py
│   ├── orchestration/
│   │   └── my_orch.cpp
│   └── aiv/
│       ├── kernel_add.cpp
│       └── kernel_mul.cpp
└── golden.py
```

### Running Tests

```bash
# Hardware platform
python examples/scripts/run_example.py -k my_test/kernels -g my_test/golden.py -p a2a3

# Simulation platform
python examples/scripts/run_example.py -k my_test/kernels -g my_test/golden.py -p a2a3sim

# Verbose output
python examples/scripts/run_example.py -k my_test/kernels -g my_test/golden.py -p a2a3sim -v
```

## Test Output

### Success Example

```
=== Building Runtime: host_build_graph (platform: a2a3sim) ===
...
=== Compiling and Registering Kernels ===
Compiling kernel: kernels/aiv/kernel_add.cpp (func_id=0)
...
=== Generating Input Tensors ===
Inputs: ['a', 'b']
Outputs: ['f']
...
=== Launching Runtime ===
...
=== Comparing Results ===
Comparing f: shape=(16384,), dtype=float32
  First 10 actual:   [42. 42. 42. 42. 42. 42. 42. 42. 42. 42.]
  First 10 expected: [42. 42. 42. 42. 42. 42. 42. 42. 42. 42.]
  f: PASS (16384/16384 elements matched)

============================================================
TEST PASSED
============================================================
```

### Failure Example

```
=== Comparing Results ===
Comparing f: shape=(16384,), dtype=float32
  First 10 actual:   [40. 40. 40. 40. 40. 40. 40. 40. 40. 40.]
  First 10 expected: [42. 42. 42. 42. 42. 42. 42. 42. 42. 42.]

TEST FAILED: Output 'f' does not match golden
```

## Reference Examples

- **Hardware Example**: [examples/host_build_graph/vector_example/](../host_build_graph/vector_example/)
- **Simulation Example**: [examples/host_build_graph/vector_example/](../host_build_graph/vector_example/)

## FAQ

### Q: How to debug test failures?

Use the `-v` flag to enable verbose output:

```bash
python examples/scripts/run_example.py -k ... -g ... -p ... -v
```

### Q: Why "binary_data cannot be empty" error?

This usually happens when:
- Using wrong platform (a2a3 vs a2a3sim)
- Kernel compilation failed silently

Solutions:
1. Verify correct `-p` parameter is used
2. Check if kernel source files exist
3. Use `-v` to view detailed compilation logs

### Q: How to add multiple test cases?

Define `ALL_CASES` and `DEFAULT_CASE` in `golden.py`:

```python
ALL_CASES = {
    "Small": {"size": 1024},
    "Medium": {"size": 2048},
    "Large": {"size": 4096},
}
DEFAULT_CASE = "Small"

def generate_inputs(params: dict) -> dict:
    size = params["size"]
    return {
        "a": torch.randn(size, dtype=torch.float32),
        "b": torch.randn(size, dtype=torch.float32),
        "out_f": torch.zeros(size, dtype=torch.float32),
    }
```

Then use `--all` to run all cases or `--case Medium` to run a specific one.

### Q: Are PyTorch tensors supported?

Yes. The test framework uses PyTorch tensors by default:

```python
import torch

def generate_inputs(params: dict) -> dict:
    return {
        "a": torch.randn(1024),
        "b": torch.randn(1024),
        "out_f": torch.zeros(1024),
    }
```

### Q: How to control log output verbosity?

Use the `--log-level` argument or `--verbose`/`--silent` flags:

```bash
# Show detailed debug information
python examples/scripts/run_example.py -k ... -g ... --verbose

# Only show errors
python examples/scripts/run_example.py -k ... -g ... --silent

# Or use explicit log level
python examples/scripts/run_example.py -k ... -g ... --log-level debug

# Show warnings and errors
python examples/scripts/run_example.py -k ... -g ... --log-level warn
```

### Q: How to hide compiler warnings?

Use info level (default). Compiler output is automatically hidden in error/warn/info levels unless compilation fails:

```bash
# Default behavior - hides compiler output
python examples/scripts/run_example.py -k ... -g ...
```

To see compiler output for debugging, use debug level:

```bash
python examples/scripts/run_example.py -k ... -g ... --verbose
# or
python examples/scripts/run_example.py -k ... -g ... --log-level debug
```

### Q: How to save C++ logs to a file?

Set the `PTO_LOG_FILE` environment variable:

```bash
export PTO_LOG_FILE=/tmp/pto_runtime.log
python examples/scripts/run_example.py -k ... -g ...

# View the logs
cat /tmp/pto_runtime.log
```

## Advanced Usage

### Custom Runtime Implementation

If you have a custom runtime implementation:

```bash
python examples/scripts/run_example.py \
  -k my_test/kernels \
  -g my_test/golden.py \
  -r my_custom_runtime \
  -p a2a3sim
```

Runtime implementation should be located at: `src/runtime/<runtime_name>/`

### Programmatic Usage

You can use `create_code_runner` directly in Python scripts. It creates a
`CodeRunner` configured from the `RUNTIME_CONFIG` in your `kernel_config.py`:

```python
from code_runner import create_code_runner

runner = create_code_runner(
    kernels_dir="my_test/kernels",
    golden_path="my_test/golden.py",
    platform="a2a3sim",
    device_id=0,
)

runner.run()  # Execute test
```

## Related Documentation

- [Main Project README](../../README.md)
- [Logging System Usage Guide](../../LOGGING_USAGE.md)
- [Python Bindings Documentation](../../python/README.md)
- [Examples Documentation](../README.md)

## Contributing

For issues or suggestions, please submit an Issue or Pull Request.
