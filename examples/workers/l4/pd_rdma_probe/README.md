# L4 PD RDMA probe

This probe uses the CANN Host C API on two Atlas A3 hosts. Each process
allocates and initializes 12 KiB of device memory before creating a HIXL CS
handle. The client writes three ranges at nonzero offsets. The receiver
checks the entire allocation, including untouched bytes.

Build with the installed CANN SDK:

```bash
source "$CANN_ROOT/set_env.sh"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread \
  -I"$ASCEND_HOME_PATH/include" examples/workers/l4/pd_rdma_probe/main.cpp \
  -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" \
  -lcann_hixl -lhcomm -lascendcl -lc_sec -o pd_rdma_probe
```

Use one idle device on each host and the Decode management address for
the listener. Set the process log directory to capture RoCE and engine
selection. The SDK performs the network transfer between device regions.
The explicit ACL copies initialize and verify the test patterns.

```bash
./pd_rdma_probe decode "$D_DEVICE" "$D_PHYSICAL" "$D_ROCE_IP" \
  "$P_PHYSICAL" "$P_ROCE_IP" "$D_MANAGEMENT_IP" "$PORT"
./pd_rdma_probe prefill "$P_DEVICE" "$P_PHYSICAL" "$P_ROCE_IP" \
  "$D_PHYSICAL" "$D_ROCE_IP" "$D_MANAGEMENT_IP" "$PORT"
```

Start the client after the receiver reports ready. Send `VERIFY` to the
receiver's stdin after the client exits successfully with confirmed retirement.
If the client failed before submission and retired, send `NO_SUBMISSION`.
An uncertain result does not authorize explicit deregistration or device free.

The tested A3 CANN 9.1.0 runtime reports RoCE QPs and an AICPU TS transport.

## Serving integration

Build the bridge from the Simpler checkout against the same CANN SDK.

```bash
source "$CANN_ROOT/set_env.sh"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread -fPIC -shared \
  -I"$ASCEND_HOME_PATH/include" examples/workers/l4/pd_rdma_probe/bridge.cpp \
  -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" \
  -lcann_hixl -lhcomm -lascendcl -lc_sec -o libpd_rdma_bridge.so
export PYPTO_HIXL_BRIDGE="$PWD/libpd_rdma_bridge.so"
```

Set `PYPTO_HIXL_DEVICE_ENDPOINTS` on each host to a JSON list of
`["RoCE IPv4", physical_device_id]` pairs in local rank order. For a single
device, you can set `PYPTO_HIXL_DEVICE_IP` and `PYPTO_HIXL_PHYSICAL_DEVICE`
instead. Set `PYPTO_HIXL_PORT` on Decode to the listener base port. Decode
rank `r` listens on `base port + r` at the advertised management address.

`Worker.bind_transfer_domain(transport="rdma", ...)` binds a Prefill member,
a Decode member, and descriptors of their existing retained regions. Use
`simpler.transfer_domain.TransferCopy` to describe ranks, region names, offsets,
and byte lengths. The caller's callbacks route operations to the process that
owns each region:

- `ensure_connected` awaits Decode listener preparation for the requested pairs.
  The connect callback groups pending pairs by Prefill rank and passes each
  owner its Decode ranks and the bound Decode member identity.
- The `put_sync_host` submit callback groups copies by Prefill owner and Decode
  rank, awaits every native write, and then returns the submitted transfer ID.
  An individual owner write returns `None` on success.
- The `close` release callback retires the source links and waits for peer
  retirement. Each owner must confirm native unregister before unpinning and
  releasing the `Worker` retain. A release callback that reports a known
  failure must be safe to retry. Starting retirement blocks new writes.

HIXL is the allocation-owner backend. It registers caller-owned device
addresses in their ACL contexts, verifies the exported regions, and
synchronously batch-writes to the linked device. Each connected Decode rank
must expose the same set of region names as its Prefill owner for HIXL. An
uncertain failure keeps the domain and its retained memory protected until
retirement is confirmed.

## VMM import probe

Use the same two A3 hosts and CANN runtime with the bridge built above. On
each host, install Simpler in its project virtual environment and build the
onboard `tensormap_and_ringbuffer` runtime. The Python probe starts an
allocation owner in a separate process. It registers the local VMM mapping
in the transfer process's ACL context.

Run Decode first and wait for `DECODE_READY`:

```bash
python examples/workers/l4/pd_rdma_probe/vmm_probe.py --role decode \
  --device "$D_DEVICE" --physical "$D_PHYSICAL" --roce-ip "$D_ROCE_IP" \
  --destination-host "$D_MANAGEMENT_IP" --port "$PORT"
```

Run Prefill in a second terminal:

```bash
python examples/workers/l4/pd_rdma_probe/vmm_probe.py --role prefill \
  --device "$P_DEVICE" --physical "$P_PHYSICAL" --roce-ip "$P_ROCE_IP" \
  --peer-physical "$D_PHYSICAL" --peer-roce-ip "$D_ROCE_IP" \
  --destination-host "$D_MANAGEMENT_IP" --port "$PORT"
```

After Prefill exits successfully, send `VERIFY` to Decode's stdin. Decode
unregisters the memory and closes its mapping. The allocation owner then
reads back all 64 KiB, checks both writes and the untouched bytes, and releases
the export lease. `VMM_VERIFIED` and zero exits on both hosts confirm success.
Send `NO_SUBMISSION` only when you have confirmed that Prefill never issued a
write and its link has retired.
