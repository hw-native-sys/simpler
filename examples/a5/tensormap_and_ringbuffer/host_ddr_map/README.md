# L3 shared Host tables with local A5 MAP

A same-machine L3 Worker owns complete POSIX Host tables. Each A5 TMR child
process imports the same backing and registers a local Device mapping. Kernels
use synchronous MTE; MAP creates no communication domain, URMA workspace or peer
connection. The examples reuse the direct-L2 gather and consumer kernels.

Child `TaskArgs` select MAP, and the target L2 prepares its local mapping before
native execution, at the parameter-preparation boundary used by H2D. Outer args
need no MAP declaration. Child arguments may select subviews, while registrations
stay cached across rounds. CPU edits between rounds are read by both
devices without table replication or broadcast. Release waits for accepted uses
and coordinated child cleanup before unlinking parent storage.

Run on at least two reserved A5 devices, with the repository installed into a
project-local venv. Run the onboard architecture precheck and example in the
same `task-submit` allocation:

```bash
task-submit --device auto --device-num 2 --timeout 1200 --max-time 1200 \
  --run 'bash .claude/skills/onboard-arch-precheck/check.sh a5 &&
  .venv/bin/python examples/a5/tensormap_and_ringbuffer/host_ddr_map/run.py --devices "$TASK_DEVICE"'
```

- `run.py`: two concurrent backings, BF16 gathers with offsets, boundary/repeated/
  invalid IDs, zero/oversized request counts, native Host/H2D IDs, a dependent
  Device consumer and CPU updates across rounds.
- `run_mx.py`: Host MXFP8 payload with an offset, local MTE gather, Device E8M0
  scales and the existing Device matrix consumer.
- Add `--staged` to hold two accepted L3 runs; BF16 uses different backings in
  the two calls, including cold mappings in the first round. Each BF16 invocation
  has distinct table values, outputs, status and golden checks. A SubWorker socket
  gate keeps the first L3 run incomplete until the second graph submission returns.
  Both calls are submitted before waiting, then checked independently. This proves
  overlapping L3 admission/graph construction, not overlapping child MAP preparation
  or simultaneous execution of Device kernels. BF16
  forwards outer HOST/NONE views; MXFP8 uses empty outer args. Both select MAP only
  in child args.
- `--prepare` compiles without initializing hardware. `--output-prefix` enables
  dependency and chip-swimlane diagnostics for each round.

The runtime supports one or more local children; these examples require two for
qualification. L3 HBG, A3, remote/cross-node MAP, borrowed contexts and capture are
unsupported. First-time mapping and cache reuse both permit earlier accepted
calls to remain active. A child preparation failure is reported like H2D failure;
other dispatched tasks may already have executed. CPU updates and overlapping Device writers require application ordering.

See [Host DDR design](../../../../docs/design/a5-host-ddr.md) for failure retention
and cleanup retry. Historical URMA measurements are not MAP qualification.
