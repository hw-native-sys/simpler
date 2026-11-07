# A5 Host-source communication domain

Each L2 owns a separate Host table. L3 initializes its named `table` slice through
`Buffer.shm.buf` with the offset from `domain.buffer_range(name)` inside the
domain's `initializer` callback; kernels read local rows with MTE and peer rows with URMA. A
consumer adds one only after the asynchronous producer completes. Local MTE
writes its gathered rows into a separate Host `local_echo` slice; L3 checks those
bytes directly after the run, demonstrating Device writes to Host DDR. The table
starts after a 256-byte slice, so peer addressing includes the window offset.

The example keeps both a Host-source and a Device-source URMA domain alive. It
uses the same L3 orchestration on the first and every subsequent invocation:
`get_or_allocate_domain` creates and initializes the Host table on the first
invocation, then reuses its data without calling the initializer again. There is
no separate domain-preparation run. The example checks the initializer count,
reuses both domains' windows across completed runs, updates the Host table from L3,
releases the Host domain, and verifies that the Device domain still transfers.
Each rank submits one gather task per domain per run. With `--aiv-blocks 4`,
four AIV blocks start as one resident cohort. Each block handles two distinct
output rows, using local MTE and URMA reads from the same peer. Two blocks fetch
the repeated row index into separate destinations. A dependent consumer checks
all eight URMA results, so early task completion or missing block output fails
verification. This uses the stable resident URMA sender binding from #2886.

The four hardware cases pass on Ascend950DT_9581. On the A5 PR CI machine,
Host-source URMA reports an ACK timeout despite successful Host mapping and
segment registration. PCIe Host-memory reachability remains under investigation
in [#2922](https://github.com/hw-native-sys/simpler/issues/2922#issuecomment-6681771862).
These cases are manual and skipped in GitHub Actions, including runs that select
manual tests. They remain available for explicit local DT validation.

Use an initialized project-local environment on a validated A5 DT machine and reserve two
devices through the device queue:

```bash
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device auto --device-num 2 --run \
  '.venv/bin/python examples/a5/tensormap_and_ringbuffer/host_comm_domain/run.py --devices "$TASK_DEVICE"'
```

Add `--staged` to exercise the staged child execution path, and `--aiv-blocks 4`
for multiple AIV senders. Pytest collects all four combinations of serial/staged
execution and one/four AIV blocks:

```bash
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device auto --device-num 2 --run \
  '.venv/bin/python -m pytest examples/a5/tensormap_and_ringbuffer/host_comm_domain --platform a5 --manual include --device "$TASK_DEVICE" -v'
```

`run.py --prepare` compiles the kernels without initializing devices. The source
and lifecycle contract is described in [A5 Host-source domains](../../../../docs/a5-host-comm-domain.md).
