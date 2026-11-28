# A3 IN_DIE RDMA through the direct SDK

This example compiles the shared PUT/GET templates, registers ordinary
`ChipCallable` objects, and uses one retained DEVICE domain for four runs.
The operation, offsets, byte count and QP index change between requests. Sources
are staged on both ranks before submission; readback starts only after `run()`
has joined the AIV FIN and the real CQ completion on both devices.

Requirements: Linux, CANN with A3 support, HCCP/verbs headers, `libra` and two A3
devices with configured RoCE IPs. `a2a3sim` cannot validate the IN_DIE data path.
Build the runtime and the kernels with the same `pto_isa.pin` and the existing
`SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=ON` setting. The Host domain object is compiled
with CANN `ccec --cce-host-only`; it links into the normal native Host runtime.
Use `PTO_HCCP_INCLUDE_DIR` for the SDK's HCCP network header directory when those
headers are distributed separately. Standard CMake include prefixes may locate
the verbs and HCCL development headers.

```bash
source /path/to/ascend-toolkit/set_env.sh
export SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=ON
python -m pip install --no-build-isolation --config-settings=build.targets=build_package_a2a3 .
python examples/a2a3/tensormap_and_ringbuffer/rdma_async_completion_demo/test_rdma_async_completion_demo.py \
    --devices 0 1 --roce-ips 192.0.2.1 192.0.2.2
```

Replace the documentation addresses with the **device RoCE** addresses, in
domain member order. Omit `--roce-ips` only when PTO can discover one unambiguous
device address. Host Ethernet addresses are not substitutes. `RdmaConfig`
contains domain resources; each submission still passes six fresh uint64 values.
Offsets and byte counts must be multiples of 64; transfers must be nonempty
and no larger than `INT32_MAX` bytes under this backend's initial contract.
The private example compiler helper creates no communicator, MR or QP and is
not an additional public SDK factory.

On hosts with Docker bridges or multiple interfaces, set the existing HCCL
`HCCL_SOCKET_IFNAME` variable to an active Host interface for communicator
bootstrap. This selects the control network; payloads still use the device RoCE
addresses. The declared `window` buffer covers all bytes staged by the example.
The Python request is captured by the callback; `Worker.run(args=...)` remains
reserved for ordinary `TaskArgs`.

One AIV invocation emits one transfer. AICPU observes CQEs without advancing any
hardware tail or doorbell. Reuse the same QP sequentially, or let PTO serialize
its AIV queue access; remote source stability and permission to overwrite a
destination remain application responsibilities. This example stages both ranks
and joins each phase through the existing Worker run fence.

On 2026-11-27, the four-run example passed on two Ascend910_9362 devices with
CANN 9.3.0-beta.1 and driver 25.5.1. Additional acceptance runs passed PUT/GET
with 256 B, 4 KiB and 64 KiB payloads on QP 0/1, checking complete window contents
and retained domain identity. This is a same-Host smoke validation; CQ wrap,
concurrent producers, fault injection and cross-Host operation remain unverified.
