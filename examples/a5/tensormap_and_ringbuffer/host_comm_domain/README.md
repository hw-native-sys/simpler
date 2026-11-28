# A5 Host-source communication domain

Each L2 owns a separate Host table. L3 initializes its named `table` slice through
`Buffer.shm.buf` with the offset from `domain.buffer_range(name)` inside the
domain's `initializer` callback. The gather kernel subtracts the local window base from the table
tensor address to recover the 256-byte offset. It adds that offset to the peer
MR base and gathers into local HBM. A dependent consumer adds one to every gathered
element after the asynchronous producer completes.

The example keeps both a Host-source and a Device-source URMA domain alive. It
uses the same L3 orchestration on the first and every subsequent invocation:
`get_or_allocate_domain` creates and initializes the Host table on the first
invocation, then reuses its data without calling the initializer again. There is
no separate domain-preparation run. The example checks the initializer count,
reuses both domains' windows across completed runs, updates the Host table from L3,
releases the Host domain, and verifies that the Device domain still transfers.
Each rank submits one gather task per domain per run. With `--aiv-blocks 4`,
four AIV blocks start as one resident cohort. Each block handles two distinct
output rows, using URMA reads from the same peer. Two blocks fetch
the repeated row index into separate destinations. A dependent consumer checks
all eight URMA results, so early task completion or missing block output fails
verification. Each resident AIV block keeps its URMA sender binding.

Pytest checks the Host–Device connection of both selected devices. It skips
the four Host-source URMA cases when either device connects to the Host over
PCIe. On that topology, cross-rank URMA reads of Host memory have reported
an ACK timeout after successful memory registration. The cases run on both UB
and HCCS. All four passed on UB hardware. HCCS has only been checked with a
simulated link query.

Compile the orchestration and both AIV kernels before reserving devices.
The standard scene precompiler discovers the L3 class and populates the same
cache that pytest reads on the target machine.

```bash
.venv/bin/python -m simpler_setup.tools.scene_test_compile \
  examples/a5/tensormap_and_ringbuffer/host_comm_domain \
  --platform a5 --exclude-level 4 --require-pto-isa --manual include
```

Run all four combinations of serial or staged submissions with one or four AIV
blocks on the allocated devices:

```bash
task-submit --device auto --device-num 2 --run \
  'bash .claude/skills/onboard-arch-precheck/check.sh a5 &&
   .venv/bin/python -m pytest examples/a5/tensormap_and_ringbuffer/host_comm_domain \
     --platform a5 --device "$TASK_DEVICE" --manual include -v'
```

Staged execution submits two runs with independent output slices and checks both
after completion. The example clears outputs between rounds and before the
final Device-only run to detect missing writes. To select one configuration,
add a pytest filter such as `-k "multi_aiv and staged"`.

The source and lifecycle contract is described in [A5 Host-source domains](../../../../docs/a5-host-comm-domain.md).
