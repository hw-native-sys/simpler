# Benchmark Scene Tests

This directory contains benchmark scene tests for the `tensormap_and_ringbuffer` runtime on the A2/A3 platform. These tests are designed to systematically characterize runtime performance across two dimensions: **dispatch overhead** and **graph topology**.

All tests use trivial kernels (noop or increment-by-one) to isolate runtime scheduling overhead from compute. Results are collected via `tools/benchmark_rounds.sh`.

## Scene 1: Dispatch & Scheduling Overhead

These tests isolate and quantify the runtime's "scheduling tax" — framework overhead independent of kernel computation.

### dispatch-independent (Task Scaling)

**Intent**: Measure how dispatch overhead grows with task count when tasks are fully independent (no inter-task data dependencies).

Each task writes `1.0` to its own cache-line-aligned slot (stride = 16 float32 = 64 bytes) in a shared output tensor, avoiding false sharing across non-coherent AICore L1 caches.

| Parameter | Values |
| --------- | ------ |
| num_tasks | 100, 500, 1000, 2000 |
| mode | AIC-only, AIV-only, AIC+AIV alternating |

**What to look for**: Linear growth in total dispatch time vs. task count. Super-linear growth indicates a scheduling bottleneck (e.g., O(N^2) dependency tracking).

### dispatch-serial (Dispatch Throughput)

**Intent**: Measure maximum scheduler throughput under serial task submission with accumulation dependencies.

All N tasks write to the same counter (AIC counter or AIV counter), forming a serial dependency chain. The final counter value equals N, validating correctness.

| Parameter | Values |
| --------- | ------ |
| num_tasks | 100, 500, 1000, 2000 |
| mode | AIC-only, AIV-only, AIC+AIV alternating |

**What to look for**: Per-task dispatch latency (total time / N). Compare with `dispatch-independent` to quantify the overhead of serial dependencies vs. independent dispatch.

## Scene 2: Graph Topology Patterns

These tests stress-test the scheduler with different DAG dependency structures. Each topology exercises a different aspect of dependency resolution.

### graph-chain_n (Linear Chain)

**Intent**: Measure serial dependency resolution overhead as chain length increases.

```text
seed(0.0) -> Task_0 -> Task_1 -> ... -> Task_{N-1} -> result(N.0)
```

Each task is an AIV increment kernel (`out = in + 1.0`). The result equals the chain length, validating every link executed.

| Parameter | Values |
| --------- | ------ |
| chain_len | 4, 8, 16, 32, 64 |

**What to look for**: End-to-end latency vs. chain length. Ideally linear; deviation reveals per-hop scheduling overhead.

### graph-fanout_n (Wide Fan-Out)

**Intent**: Test parallel dispatch capability — can the runtime simultaneously issue N independent tasks from a single source?

```text
seed -> [Source] -> intermediate -> [Consumer_0] -> result[0]
                                 -> [Consumer_1] -> result[1]
                                 -> ...
                                 -> [Consumer_{N-1}] -> result[N-1]
```

Consumer output slots are cache-line-aligned to avoid false sharing. Each consumer reads the same source output and writes `source + 1.0`.

| Parameter | Values |
| --------- | ------ |
| fanout_width | 2, 4, 8, 15 |

**What to look for**: Whether fan-out width impacts total latency. Ideal runtime dispatches all consumers in parallel, so latency should plateau rather than grow linearly.

### graph-fanin_n (Convergence Barrier)

**Intent**: Measure dependency convergence overhead — how efficiently the runtime tracks N predecessors for a single barrier task.

```text
seed -> [Producer_0] -> prod_out_0 -.
seed -> [Producer_1] -> prod_out_1 -+-> [Barrier] -> result(1.0)
...                                 |
seed -> [Producer_{N-1}] -> ...    -'
```

Each producer writes independently; the barrier depends on all N producer outputs.

| Parameter | Values |
| --------- | ------ |
| fanin_width | 2, 4, 8, 15 |

**What to look for**: Barrier wait overhead vs. fan-in width. Measures the cost of tracking and synchronizing N predecessor completions.

### graph-diamond (Fork-Join)

**Intent**: Test the most common real-world DAG pattern — fan-out followed by fan-in (fork-join).

```text
seed -> [Source A] -> a_out -> [Branch B_0] -> b_out_0 -.
                            -> [Branch B_1] -> b_out_1 -+-> [Merge D] -> result(1.0)
                            -> ...                      |
                            -> [Branch B_{W-1}] -> ... -'
```

Three branch modes exercise different core-type scheduling paths:

- **mode=0**: All AIV branches
- **mode=1**: All AIC branches
- **mode=2**: Mixed AIC+AIV (even=AIC, odd=AIV)

| Parameter | Values |
| --------- | ------ |
| width | 2, 4, 8, 15 |
| mode | AIV-only, AIC-only, Mixed AIC+AIV |

**What to look for**: Combined fan-out + fan-in overhead. Compare with isolated fanout/fanin tests to check for compounding effects. Mixed mode reveals cross-core-type scheduling costs.

## Also Updated: benchmark_bgemm

The existing `benchmark_bgemm` test was extended with structured parameter sweeps:

- **Tile size sweep** (16, 32, 64, 128) at fixed batch and grid_k
- **Batch/group sweep** (1, 4, 16, 64 groups) at fixed tile size
- **Grid-K sweep** (1, 2, 4) at fixed tile and batch

These complement the original 5 cases with systematic single-variable sweeps for identifying performance cliffs.

## Running

```bash
# Run all benchmark scene tests (100 rounds each, default)
./tools/benchmark_rounds.sh

# Customize
./tools/benchmark_rounds.sh -n 50 -d 0 -p a2a3 -r tensormap_and_ringbuffer -v
```
