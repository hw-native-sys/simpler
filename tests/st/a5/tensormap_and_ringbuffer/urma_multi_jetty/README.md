# URMA multi-jetty acceptance

These two-rank A5 TMR cases create a PER_PEER domain with
`workspace_engines={"urma": UrmaConfig(jetties_per_aiv=J)}`. They compare
`J=1` and `J=2` while retaining the provider's default queue depths.

Each request is split into at most 256 MiB chunks. Chunk `i` uses jetty
`i % J` in the resident AIV's private queue group. Every new request starts
at jetty zero, so repeated small requests exercise only that queue.

The transfer task records the resident sender, actual queue rows, SQ heads,
submitted WQE counts and per-queue completion targets. A dependent observer
reads the original sender's SQ/CQ tails after the transfer retires. The test
checks actual submitted and completed counter deltas against the expected
chunk distribution; an event naming several queues is not sufficient.
PUT cases read the peer's receive region back with a dependent GET. Every
case compares the complete result and a downstream vector consumer's output.
Input values differ between 256 MiB chunks and between logical senders.

## Cases

| Class | Coverage |
| ----- | -------- |
| `TestUrmaMultiJettyBoundaries` | GET/PUT, one/two jetties, 256 MiB minus 32 KiB, exactly 256 MiB, plus 32 KiB, and 512 MiB plus 32 KiB |
| `TestUrmaMultiJettySmallRequests` | Multiple requests before task retirement, each restarting at jetty zero |
| `TestUrmaMultiJettySenders` | Two simultaneous SPMD AIV owners with disjoint two-jetty groups |
| `TestUrmaMultiJettyDepth` | Explicit SQ depth request of 1024; actual SQ/CQ capacities are recorded for both jetties |
| `TestUrmaMultiJettyQueueWrap` | Repeated tasks cross the actual SQ/CQ capacity of jetty zero while jetty one stays unused |
| `TestUrmaMultiJettyTiming` | Manual single-sender GET/PUT comparison, 21 waves per jetty count |

The queue-wrap case retains a single scheduler owner to keep physical queue
reuse bounded. It checks the actual depths returned by the provider and fails
if no queue wraps; 4097 waves of eight small requests cover the current default
32768-entry queues. Cross-task counters must continue from the last observation
of the same row. This case does not claim hardware wrap coverage for every jetty.
Deterministic C++ tests cover counter wrap, exhausted capacity, delayed queue
completion and partial failures without deliberately poisoning a shared device.

## Running

Run the architecture precheck before acquiring devices, then use one exclusive
`task-submit` allocation for the complete comparison:

```bash
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device auto --device-num 2 --timeout 1800 --max-time 1800 \
  --run 'PYTEST_ADDOPTS=-s .venv/bin/python -m pytest tests/st/a5/tensormap_and_ringbuffer/urma_multi_jetty \
    --platform a5 --device "$TASK_DEVICE" --pto-session-timeout 1200 -v -s'
```

`PYTEST_ADDOPTS=-s` also exposes queue records from the scheduled pytest children.
Select `TestUrmaMultiJettyTiming` with `--manual include` for timing. Run both
jetty counts in the same allocation, with the same workload, device order,
ISA pin and CANN installation. Record those identities alongside the logs.
The `equal_sq_capacity` cases request SQ depth 2048 for one jetty and 1024
for two jetties, holding the sender group's aggregate SQ capacity constant.
The provider's requested depth and actual hardware entry count can differ.
The paired cases check that the returned aggregate SQ capacities are equal;
CQ capacity is reported separately.

## Evidence and timing interpretation

Each case prints one `URMA_MULTI_JETTY` JSON record containing:

- First-wave per-sender queue rows and real submitted/completed increments.
- Actual SQ/CQ depths, workspace bytes, queue-context row count, and the
  observed sender group's SQ/CQ payload bytes (64 bytes per entry).
- Cross-task reuse observations and queue rows that crossed their capacities.
- Median elapsed time from request submission to the dependent observer's
  start, with the first wave omitted when additional waves exist.

The elapsed interval includes runtime completion observation and dispatch of
the observer. It is not exact NIC latency. `observed_bytes_per_us` uses that
same interval and counts repeated requests within the producer task. The
workspace byte count covers metadata and workspace-owned backing; the sender
SQ/CQ payload count covers only that sender group's send queues. Neither is
total domain/device memory usage: HCCL receive resources and other senders'
queues are additional. Capture device memory before/after domain creation
when comparing total resource cost. Multi-jetty does not imply multi-rail,
and no throughput gain is required for correctness.
