# Performance Optimization Guide

## Profiling Workflow

1. Run with profiling enabled:
   ```bash
   python examples/scripts/run_example.py \
       -k <kernels> -g <golden.py> -p a2a3 --enable-profiling
   ```

2. Analyze with swimlane converter:
   ```bash
   python3 tools/swimlane_converter.py -k <kernels>/kernel_config.py
   ```

3. Open `outputs/merged_swimlane_*.json` in [Perfetto](https://ui.perfetto.dev/)

4. Generate dependency graph:
   ```bash
   python3 tools/perf_to_mermaid.py -k <kernels>/kernel_config.py
   ```

See `tools/README.md` for full tool documentation.

## Host-Side Timing

The test framework prints wall-clock timing for each phase at `info` log level:

```
[INFO] >>> runtime.initialize() took 1.234s
[INFO] >>> compute_golden() took 0.005s
[INFO] >>> Total init-to-launch: 1.239s (initialize=1.234s, golden=0.005s)
[INFO] >>> launch_runtime() took 0.142s
```

`launch_runtime()` measures the total on-device execution time (AICPU scheduling + AICore compute + synchronization).

## Key Metrics to Watch

- **Total function time** — Candidates for kernel optimization
- **AIC/AIV overlap** — Good parallelism means AIC and AIV tasks overlap
- **AICPU scheduling overhead** — Should be small relative to compute time
- **Min/Max spread** — Large spread indicates inconsistent performance (investigate)

## Optimization Strategies

### Task Graph Optimization

- **Maximize parallelism** — Structure dependencies so independent tasks can run on different cores simultaneously
- **Minimize fanin bottlenecks** — Avoid tasks that wait for many predecessors
- **Balance core utilization** — Distribute work evenly across AIC and AIV cores

### Kernel Optimization

- **Use appropriate core type** — Matrix operations on AIC (CUBE), element-wise on AIV (VECTOR)
- **Tile sizes** — Adjust tile sizes to match hardware vector width and L1 cache
- **Memory access patterns** — Sequential access patterns are faster than scattered reads
- **Minimize synchronization** — Reduce `pipe_barrier()` calls to necessary synchronization points

### Memory Optimization

- **Tensor layout** — Choose layouts that minimize data movement between operations
- **Buffer reuse** — Reuse output buffers as inputs for subsequent operations where possible
- **Alignment** — Ensure tensors are aligned to hardware-preferred boundaries

## Simulation vs Hardware Comparison

Run the same example on both platforms and compare:

```bash
# Simulation (functional correctness)
python examples/scripts/run_example.py -k <kernels> -g <golden.py> -p a2a3sim

# Hardware (real performance)
python examples/scripts/run_example.py -k <kernels> -g <golden.py> -p a2a3 --enable-profiling
```

Simulation validates correctness; hardware profiling reveals actual performance characteristics. Large discrepancies between simulation and hardware timing indicate platform-specific bottlenecks.
