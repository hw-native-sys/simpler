# A5 L2 Host DDR gather

`run.py` reads selected BF16 rows from this process's POSIX_SHM table into Device
GM and submits a dependent BF16-to-FP32-plus-one consumer. This is synchronous
mapped MTE: the gather AIV returns only after its transfers finish. It is not an
asynchronous URMA example. Both TMR and HBG use the same mapping implementation. HBG also reads a native
Host control Tensor from the same Buffer in the same call, independently of the
Device-mapped table view.
No URMA workspace or communication domain is required.

```python
table = l2.create_buffer(nbytes)
# Fill table.host_view() before submitting; release exported views before close.
native = table.tensor(shape, DataType.BFLOAT16)
args.add_tensor(native, transfer=TensorTransfer.MAP)  # INPUT
pending = l2.submit(entry, args)  # entry declares the table parameter IN
pending.wait()
l2.release_buffer(table)
```

The Buffer and its views stay HOST. Submission prepares a cached mapping per Buffer and uses the
ACL-returned device address, never the CPU pointer. The gather example uses `l2.malloc` for its outputs.
No table-sized Device allocation or automatic whole-table H2D occurs. A kernel
that ignores the table performs no automatic read. The original Tensor retains ordinary CPU access and default H2D behavior. HOST/NONE must be explicit.

The example keeps fixed Device payload allocations while growing Host backing
from about 2 MiB to 8 MiB, creates the second table after a completed run, and
reuses the first table after CPU updates. It checks duplicate/first/last IDs,
nonzero view offsets, zero requests, invalid IDs and capacity overflow. Invalid
IDs set status 2 and leave those output rows untouched. Excess request count sets
status 1 and transfers nothing. Successful/zero requests set status 0. Its row
copy kernel accepts BF16 and MXFP8 contiguous rows of 32–4096 bytes in multiples
of 32; these are example constraints, not Buffer limits.

Both `Buffer.close()` and `release_buffer()` refuse accepted uses until `wait()`
finishes cleanup. Direct `table.shm.close()`/`unlink()` also refuse a registered
backing, so release must pass through the Buffer owner. A timeout or execution
error can leave the use unproven: backing and registration then remain retained, including across garbage collection.
Wait for Device writers before reading their results on CPU, and do not modify
Host backing while any run can access it. `Worker.close()` drains
unawaited successful runs before unmapping; it cannot claim a failed device has
stopped merely because a timeout expired. Contexts with unproven accesses remain
unavailable and retained until process termination; automatic recovery is not
provided. A failed unregister retains the backing for a later cleanup retry.

Build and run from the repository with its local venv and sourced A5 CANN:

```bash
.venv/bin/python -m pip install \
  --no-build-isolation -e . --config-settings=build.targets=build_package_a5
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device 4 --timeout 600 --max-time 600 --run \
  "$PWD/.venv/bin/python -m pytest tests/ut/py/test_host_ddr_onboard.py --platform a5 --device 4 -q -s"
```

`run_write.py` maps two disjoint Float32 views of one Host Buffer. One is declared
OUT (`OUTPUT_EXISTING` in TaskArgs); the other is declared INOUT. It verifies
Device writes through `Buffer.host_view()` after `wait()`, without D2H staging,
and checks that the surrounding bytes stay unchanged. MAP has no access selector; ordinary Buffer grants and callable directions apply. An IN declaration
is a kernel/dependency contract, not hardware protection against pointer writes.

Each runtime is exercised in its own process. `run_mx.py` reads Host E4M3FN
payloads, then invokes the existing `mx_fp_gemm` TMATMUL_MX kernel with Device
E8M0 scales and Device B. No quantization or model operator is added.

This example covers this process's Host table on one A5. Cross-rank asynchronous reads and
L3-to-L2 resource protocols belong to later PRs and are not required to use this
example. See the [same-device mapping design](../../../../docs/design/a5-host-ddr.md).
