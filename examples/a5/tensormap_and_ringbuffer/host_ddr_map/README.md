# L3 shared Host tables with local A5 MAP

A same-machine L3 Worker owns complete POSIX Host tables. Each A5 TMR child
process imports the same backing and registers a local Device mapping. Kernels
use synchronous MTE; MAP creates no communication domain, URMA workspace or peer
connection. The examples reuse the direct-L2 gather and consumer kernels.

Outer `TaskArgs` declare MAP backings before the callback. Every child is ready
before dispatch. Child arguments may select subviews, while registrations stay
cached across fully drained rounds. CPU edits between rounds are read by both
devices without table replication or broadcast. Release waits for accepted uses
and coordinated child cleanup before unlinking parent storage.

Run on at least two reserved A5 devices, with the repository installed into a
project-local venv. Run the onboard architecture precheck first, then use the
`task-submit` queue:

```bash
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device auto --device-num 2 --timeout 1200 --max-time 1200 \
  --run 'python examples/a5/tensormap_and_ringbuffer/host_ddr_map/run.py --devices "$TASK_DEVICE"'
```

- `run.py`: two concurrent backings, BF16 gathers with offsets, boundary/repeated/
  invalid IDs, zero/oversized request counts, native Host/H2D IDs, a dependent
  Device consumer and CPU updates across rounds.
- `run_mx.py`: Host MXFP8 payload with an offset, local MTE gather, Device E8M0
  scales and the existing Device matrix consumer.
- Add `--staged` to hold two accepted L3 runs; both reuse prepared mappings.
- `--prepare` compiles without initializing hardware. `--output-prefix` enables
  dependency and chip-swimlane diagnostics for each round.

The runtime supports one or more local children; these examples require two for
qualification. L3 HBG, A3, remote/cross-node MAP, borrowed contexts and capture are
unsupported. Declare every mapped backing in each outer submission. First-time
preparation requires earlier calls to drain; reuse permits overlapping accepted
calls. CPU updates and overlapping Device writers require application ordering.

See [Host DDR design](../../../../docs/design/a5-host-ddr.md) for failure retention
and cleanup retry. Historical URMA measurements are not MAP qualification.
