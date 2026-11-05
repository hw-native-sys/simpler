# URMA concurrent sender validation

These A5 `tensormap_and_ringbuffer` tests use one communication domain and the
public deferred URMA request adapter. Rank 0 sends into disjoint remote slots,
then reads them back after PUT completion. An add-one consumer depends on the
GET output. Peer-specific payloads distinguish target ranks even when the
remote offsets are identical.

The host provisions the default `PER_PEER` workspace: one jetty for each
resident AIV and remote peer, with the provider's default queue depth. A
completion's queue row must be `peer * rowsPerPeer + stable_sender_id`.
Different AIVs use different rows for the same peer, and one AIV uses different
rows for different peers. These cases do not exercise shared queues or multiple
jetties per sender.

Each case has its own scene-test class and Worker. Two-rank cases use a complete
two-rank communicator; `two_peers` uses a complete three-rank communicator.
Repeated task submissions reuse one live domain. The suite does not depend on
reallocating a released URMA domain or creating overlapping subcommunicators.

| Case | Coverage |
| ---- | -------- |
| `same_peer` | Two SPMD AIV blocks submit to different slots on one peer |
| `concurrent_tasks` | Two independent AIV tasks submit to one peer |
| `mix_siblings` | Both AIV lanes of one MIX block submit through distinct queue groups |
| `two_peers` | One AIV uses distinct per-peer queues to reach two peers with distinct data |
| `queue_wrap` | Independent tasks reuse queues; actual SQ and CQ progress must exceed their observed depths |
| `reuse_observation` | Independent submission groups can reuse queues before earlier tasks retire; reports entry counters |

## Evidence and limits

Device records report stable AIV identity, raw CCE block/subblock, runtime lane, queue row,
SQ/CQ depths, submitted counters, completed counters, and producer timestamps.
The tests check every registered URMA completion belongs to the executing AIV
and the request's peer, and carries the domain workspace cookie. The two-peer
case checks both queues advance and each remote address uses its peer MR base
plus the requested offset. It does not require MR virtual addresses in separate
processes to differ. MIX requires runtime lanes 0 and 1 to have distinct stable
senders and queues; raw CCE values are observations and may alias.
The first producer group uses a bounded rendezvous, and overlapping
hardware timestamp intervals establish concurrent execution.

Completion counts are checked against the adapter's chunk size:
`ceil(bytes / (256 MiB))` per request with the default single jetty. Every
transfer record must report success and no completion error, and the final
payload and downstream consumer must match the peer-specific input.

The wrap case verifies physical ring reuse, not 32-bit counter overflow. Its
single scheduler owner concentrates requests on a bounded set of queues; its
iteration count fails explicitly if no queue reaches its actual depth.
4097 waves of eight requests target queues of depth 32768; acceptance still
requires measured progress to exceed both actual depths on at least one queue.
Each wave closes its own scope so completed tasks and dependency entries can
be reclaimed during the run. Explicit dependencies between waves preserve PUT
ordering and the final GET; the consumer depends on the GET output through
automatic tensor dependencies.
CPU regressions cover arithmetic rollover and injected errors separately.
`reuse_observation` reports submitted-minus-completed counters at task entry;
zero is valid and does not constitute evidence of an outstanding request.
The suite does not claim that a delayed completion implies DMA is still active.

The 32-word device record keeps peer 1's SQ/CQ progress in words 10–13, 16–17
and 19. Words 14 and 22 hold peer 2's submitted counts after and before the
requests; 18 holds the runtime lane; 23 is `rowsPerPeer`; 24–25 are the actual
completion rows for peers 1–2; 26–27 are their remote addresses; 28 is the remote
offset; 29–30 are their MR bases; and 31 is the completion error code.

## Run

Build the A5 package with the repository's pinned ISA in a project-local venv.
Use the onboard architecture precheck and the device queue:

```bash
python -m simpler_setup.tools.scene_test_compile \
  tests/st/a5/tensormap_and_ringbuffer/urma_concurrent_senders \
  --platform a5 --require-pto-isa --compile-workers 4
mkdir -p outputs/urma_concurrent/ascend
export ASCEND_PROCESS_LOG_PATH="$PWD/outputs/urma_concurrent/ascend"
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device auto --device-num 3 --timeout 1800 --max-time 1800 \
  --run ".venv/bin/python -m pytest tests/st/a5/tensormap_and_ringbuffer/urma_concurrent_senders --platform a5 --device \$TASK_DEVICE --require-pto-isa --pto-session-timeout 1200 -v"
```

Select an individual case with `--case same_peer` or another name from the table.
Preserve the exact simpler/ISA/CANN revisions, runtime artifacts, device IDs,
resource-child results, and device logs with the acceptance report. A skipped or
unexecuted case is not hardware acceptance.
