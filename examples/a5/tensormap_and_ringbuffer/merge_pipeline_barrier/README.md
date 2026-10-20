# merge_pipeline_barrier — three stages, one task, a barrier instead of a scheduler

Everywhere else in this tree, ordering between pipeline stages is what the
runtime is *for*: submit three tasks, let TensorMap chain them. This example
does the opposite — it merges all three stages into **one** `block_num=8` AIV
task and orders them with a cross-core barrier inside the kernel.

```text
stage A: x + 1        stage B: × 2        stage C: + 1        →  out = 2x + 3
```

The computation is trivial on purpose. What is under test is the barrier.

## Why a barrier at all

The three stages have deliberately mismatched shapes, so the barrier has to
work when the cores are unevenly loaded:

| Stage | Work | Blocks doing it |
| ----- | ---- | --------------- |
| A | 8 tiles | block 0 only |
| B | 2 tiles each | blocks 0–3 |
| C | 1 tile each | all 8 |

**Every block arrives at every barrier, including the idle ones**, so each
barrier waits for all `block_num` arrivals. Blocks 4–7 do nothing until stage
C, and still have to show up twice before it.

## Two barrier implementations, chosen at compile time

`MERGE_BARRIER_COUNTER` in `kernels/aiv/merge_pipeline.cpp` selects between
them (default `1`):

| Value | Mechanism | Reads per barrier |
| ----- | --------- | ----------------- |
| `0` | **Per-slot.** Each block writes its own slot with `st_dev`, single-writer so no atomic needed; the reader polls all `n` slots with `ld_dev`. | **O(n)** non-cacheable |
| `1` | **Single counter.** Arrival is a scalar hardware `st_atomic` add, bracketed by `dcci` so the line is written out atomically; every block then polls that one counter with `ld_dev` until it reaches `epoch * n`. | **O(1)** |

The slot version needs no atomic and no cache maintenance; the counter version
trades one atomic-add on arrival for a single-address poll. Both use
`ld_dev`/`st_dev` — non-cacheable device accesses — rather than ordinary loads,
and the `epoch` argument advances monotonically so a slot or counter is never
mistaken for a previous barrier's.

Note where the `dcci` calls sit in the counter version: around the *arrival*
write, not in the poll loop. The reader's `ld_dev` bypasses cache, so the hot
path carries no cache maintenance at all.

## Measurement is built in

The kernel records each block's per-barrier gap with `get_sys_cnt` into a
`timing` output — `NBLK × 32` int32, one 32-int stride per block, first 6
entries used. `compare_outputs` is overridden to print those (ticks at 50 MHz,
so ÷50 gives µs) and then **drop `timing` from the golden comparison**, since
it has no expected value.

That override is the pattern to copy whenever a scene test needs a diagnostic
output alongside checked ones:

```python
def compare_outputs(self, test_args, golden_args, output_names, params):
    ...  # print the diagnostic
    super().compare_outputs(
        test_args, golden_args, [n for n in output_names if n != "timing"], params
    )
```

## Run

```bash
pytest examples/a5/tensormap_and_ringbuffer/merge_pipeline_barrier --platform a5 --device 0
```

**Onboard only, and it cannot be otherwise.** The barrier is a busy-wait, so
all `block_num` logical blocks must be co-resident — if a block is not running,
the ones waiting on it never proceed. The A5 runtime can launch 36 logical
blocks, while this task uses `block_num` 8 (set per task via
`launch_spec.set_block_num` in `merge_orch.cpp`), so all blocks can be resident.

Wrap in `task-submit` on a shared box.
