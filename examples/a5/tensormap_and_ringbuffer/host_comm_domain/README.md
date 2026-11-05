# A5 Host-source communication domain

Each L2 owns a separate Host table. L3 initializes its named `table` slice through
`Buffer.shm.buf` with the offset from `domain.buffer_range(name)`; kernels read local rows with MTE and peer rows with URMA. A
consumer adds one only after the asynchronous producer completes. Local MTE
writes its gathered rows into a separate Host `local_echo` slice; L3 checks those
bytes directly after the run, demonstrating Device writes to Host DDR. The table
starts after a 256-byte slice, so peer addressing includes the window offset.

The example keeps both a Host-source and a Device-source URMA domain alive. It
reuses their windows across completed runs, updates the Host table from L3,
releases the Host domain, and verifies that the Device domain still transfers.
Each rank submits one gather task per domain per run. With `--aiv-blocks 4`,
four AIV blocks start as one resident cohort. Each block handles two distinct
output rows, using local MTE and URMA reads from the same peer. Two blocks fetch
the repeated row index into separate destinations. A dependent consumer checks
all eight URMA results, so early task completion or missing block output fails
verification. This uses the stable resident URMA sender binding from #2886.

Use an initialized project-local environment on an A5 machine and reserve two
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
  '.venv/bin/python -m pytest examples/a5/tensormap_and_ringbuffer/host_comm_domain --platform a5 --device "$TASK_DEVICE" -v'
```

`run.py --prepare` compiles the kernels without initializing devices. The source
and lifecycle contract is described in [A5 Host-source domains](../../../../docs/a5-host-comm-domain.md).
