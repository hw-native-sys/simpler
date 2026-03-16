# benchmark_rounds.sh

A benchmark wrapper that runs device test examples on Ascend hardware, parses per-round timing data from device logs, and reports latency statistics.

## Quick Start

```bash
# Run with defaults from benchmark_config.json
./tools/benchmark_rounds.sh

# Run with custom rounds and verbose output
./tools/benchmark_rounds.sh -n 20 -w 3 -v

# Use a different config file
./tools/benchmark_rounds.sh -c /path/to/my_config.json

# Generate plots and save logs
./tools/benchmark_rounds.sh --plot --log
```

## Usage

```
./tools/benchmark_rounds.sh [-c <config>] [-p <platform>] [-d <device>] [-n <rounds>] [-w <warmup>] [-v] [--plot] [--log]
```

## Command-Line Options

All CLI arguments override their corresponding values from the config file.

| Option | Long Form | Description | Config Key | Default |
|--------|-----------|-------------|------------|---------|
| `-c` | `--config` | Path to JSON config file | — | `tools/benchmark_config.json` |
| `-p` | `--platform` | Platform to run on | `platform` | `a2a3` |
| `-d` | `--device` | Device ID | `device_id` | `0` |
| `-n` | `--rounds` | Number of measured rounds per example | `rounds` | `10` |
| `-w` | `--warmup` | Number of warm-up rounds to discard | `warmup_rounds` | `2` |
| `-v` | `--verbose` | Print detailed `run_example.py` output | `verbose` | `false` |
| | `--plot` | Generate scatter plot PNG for each example | `plot` | `false` |
| | `--log` | Save statistics to `benchmark_logs/` | `log` | `true` |
| `-h` | `--help` | Show help message | — | — |

Any unrecognized arguments are passed through to `run_example.py` (e.g., `--case`).

## Configuration File

The script loads settings from a JSON config file (`benchmark_config.json` by default, located next to the script). Example:

```json
{
    "project_root": "..",
    "examples_subdir": "tests/device_tests/${platform}/tensormap_and_ringbuffer",
    "examples": [
        "alternating_matmul_add",
        "benchmark_bgemm",
        "paged_attention_unroll",
        "batch_paged_attention",
        "paged_attention"
    ],
    "device_id": 0,
    "rounds": 10,
    "warmup_rounds": 2,
    "platform": "a2a3",
    "verbose": false,
    "log": true,
    "plot": false
}
```

### Config Fields

| Field | Type | Description |
|-------|------|-------------|
| `project_root` | string | Relative path from the script directory to the project root. |
| `examples_subdir` | string | Subdirectory under project root containing examples. The `${platform}` placeholder is replaced with the active platform value. |
| `examples` | array | List of example directory names to benchmark. |
| `device_id` | int | Ascend device ID to run on. |
| `rounds` | int | Number of measured (non-warmup) rounds. |
| `warmup_rounds` | int | Number of initial rounds to discard before measuring. |
| `platform` | string | Target platform (e.g., `a2a3` for hardware, `a2a3sim` for simulation). |
| `verbose` | bool | Whether to show full `run_example.py` output. |
| `log` | bool | Whether to save per-example statistics to `benchmark_logs/`. |
| `plot` | bool | Whether to generate scatter plot PNGs (requires `matplotlib`). |

## Execution Logic

The script follows these steps:

### 1. Load Configuration

Configuration is loaded from the JSON file using `python3`. The config file path defaults to `benchmark_config.json` in the same directory as the script. If `-c` is provided on the command line, that config file is loaded instead. CLI arguments then override any config values.

### 2. Resolve Paths

- The `examples_subdir` path has its `${platform}` placeholder replaced with the actual platform value.
- The device log directory is resolved from `$ASCEND_WORK_PATH/log/debug/device-<id>` (falling back to `$HOME/ascend/log/debug/device-<id>`).

### 3. Iterate Over Examples

For each example listed in the config:

1. **Validate**: Check that both `golden.py` and `kernels/` exist in the example directory. Skip otherwise.
2. **Snapshot logs**: Record the list of existing `.log` files in the device log directory (used later to detect the newly created log).
3. **Run**: Execute `run_example.py` with the example's kernels directory and golden file. The total number of rounds passed is `rounds + warmup_rounds`. In non-verbose mode, stdout/stderr are suppressed.
4. **Find new log**: Wait up to 15 seconds for a new `.log` file to appear in the device log directory. Falls back to the newest log file if no new file is detected.
5. **Parse timing**: Extract `orch_start` and `end` timestamps from the device log and compute per-round elapsed time.

### 4. Timing Analysis

The `parse_timing` function processes device log lines containing `orch_start=` and `end=` markers:

- For each round, it computes elapsed time as `(max_end - min_start) / 50.0` microseconds (clock ticks to microseconds conversion).
- The first N rounds (controlled by `--warmup`) are labeled as warmup and excluded from statistics.
- For the remaining measured rounds, it computes:
  - **Arithmetic mean**: Average of all measured rounds.
  - **Median**: Middle value of sorted measurements.
  - **Trimmed mean**: Mean after dropping the single lowest and highest values (requires at least 3 samples).
  - **Range (Max-Min)**: Difference between highest and lowest measurements.
  - **Mean Absolute Deviation (MAD)**: Average absolute distance from the mean.
  - **Standard deviation**: Root-mean-square deviation from the mean.
  - **Fluctuation rate (CV)**: Coefficient of variation, i.e., `(stddev / mean) * 100%`.

### 5. Optional: Save Logs

When `--log` is enabled, per-example statistics are saved to `tools/benchmark_logs/<example>_<timestamp>.log`.

### 6. Optional: Generate Plots

When `--plot` is enabled, the script collects per-round data and generates a scatter plot PNG for each example using `matplotlib`. Each plot shows:

- Individual per-round elapsed times as scatter points.
- Horizontal reference lines for the arithmetic mean and trimmed mean.

Plots are saved to `tools/benchmark_logs/benchmark_<example>_<timestamp>.png`. If `matplotlib` is not installed, a warning is printed and plot generation is skipped.

### 7. Summary

After all examples complete, a summary line reports the number of passed and failed examples. The script exits with a non-zero status if any example failed.

## Output Example

```
================================================================
  alternating_matmul_add
================================================================
  Log: /path/to/device-0/xxx.log
  Round     Elapsed (us)
  -----     ------------
  0              123.4  (warmup)
  1              121.8  (warmup)
  2              118.2
  3              117.9
  ...

  Mean: 118.1 us  |  Median: 118.0 us  |  Trimmed-mean: 118.0 us  (10 rounds, 2 warmup)
  Range(Max-Min): 2.3 us  |  Avg variation (MAD): 0.8 us  |  Std deviation: 0.9 us  |  Fluctuation rate (CV): 0.76%
  result: 118.1 us

================================================================
  Benchmark complete: 5 passed, 0 failed (5 total)
================================================================
```

## Prerequisites

- **python3**: Required for config parsing, running examples, and optional plot generation.
- **Ascend device**: Required for hardware benchmarking (platform `a2a3`).
- **matplotlib** (optional): Required only when `--plot` is enabled.
- **PTO2_PROFILING**: Must be enabled in the device environment to produce the `orch_start`/`end` timing markers in device logs.
