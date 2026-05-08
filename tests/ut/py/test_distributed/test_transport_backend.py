import ctypes
import os

import pytest

from simpler.distributed.proto import dispatch_pb2
from simpler.distributed.transport_backend import (
    EndpointSpec,
    HcommDataPlaneClient,
    HcommRuntime,
    RxeDataPlaneClient,
    RxeTensorTransport,
    _CommAbiHeader,
    _CommAddr,
    _EndpointDesc,
    _EndpointLoc,
    _HcommChannelDesc,
    _HcommMem,
    _RxeServerDesc,
    _RXE_DESC_MAGIC,
    _decode_rxe_desc,
    _encode_rxe_desc,
)


REAL_HCOMM = pytest.mark.skipif(
    os.getenv("SIMPLER_HCOMM_REAL_TEST") != "1",
    reason="set SIMPLER_HCOMM_REAL_TEST=1 to run against a real HCOMM library/environment",
)


def test_hcomm_ctypes_struct_layout_matches_public_headers():
    assert ctypes.sizeof(_CommAddr) == 40
    assert ctypes.sizeof(_EndpointLoc) == 64
    assert ctypes.sizeof(_EndpointDesc) == 160
    assert ctypes.sizeof(_HcommMem) == 24
    assert ctypes.sizeof(_CommAbiHeader) == 16
    assert ctypes.sizeof(_HcommChannelDesc) == 344
    assert _HcommChannelDesc.header.offset == 0
    assert _HcommChannelDesc.remoteEndpoint.offset == 16
    assert _HcommChannelDesc.notifyNum.offset == 176
    assert _HcommChannelDesc.memHandles.offset == 184
    assert _HcommChannelDesc.memHandleNum.offset == 192
    assert _HcommChannelDesc.socket.offset == 200
    assert _HcommChannelDesc.role.offset == 208
    assert _HcommChannelDesc.port.offset == 212
    assert _HcommChannelDesc.attr.offset == 216


def test_rxe_desc_roundtrip():
    desc = _RxeServerDesc()
    desc.ip = b"192.168.0.243"
    desc.port = 12345
    desc.rkey = 678
    desc.addr = 0xABCDEF
    desc.size = 4096

    payload = _encode_rxe_desc(desc, "rxe0", 1)
    assert payload.startswith(_RXE_DESC_MAGIC)
    decoded = _decode_rxe_desc(payload)
    assert decoded.ip == "192.168.0.243"
    assert decoded.port == 12345
    assert decoded.rkey == 678
    assert decoded.addr == 0xABCDEF
    assert decoded.size == 4096
    assert decoded.device == "rxe0"
    assert decoded.gid_index == 1


def test_rxe_legacy_json_desc_is_still_accepted():
    payload = (
        b'{"addr":11259375,"device":"rxe0","gid_index":1,"ip":"192.168.0.243",'
        b'"port":12345,"rkey":678,"size":4096,"transport":"rxe","version":1}'
    )
    decoded = _decode_rxe_desc(payload)
    assert decoded.ip == "192.168.0.243"
    assert decoded.port == 12345
    assert decoded.rkey == 678
    assert decoded.addr == 0xABCDEF
    assert decoded.size == 4096
    assert decoded.device == "rxe0"
    assert decoded.gid_index == 1


def test_rxe_client_rejects_empty_transport_desc():
    client = RxeDataPlaneClient()
    handle = dispatch_pb2.TensorHandle(transport="rxe", nbytes=1)
    with pytest.raises(Exception, match="transport_desc"):
        client.write_handle(handle, 1, 1)


@REAL_HCOMM
def test_real_hcomm_runtime_loads_required_symbols():
    runtime = HcommRuntime.from_env(required=True)
    assert runtime.available
    for symbol in (
        "HcommEndpointCreate",
        "HcommEndpointDestroy",
        "HcommMemReg",
        "HcommMemUnreg",
        "HcommMemExport",
        "HcommMemImport",
        "HcommMemUnimport",
        "HcommChannelCreate",
        "HcommChannelDestroy",
        "HcommWriteWithNotifyNbi",
        "HcommChannelFence",
    ):
        assert hasattr(runtime._lib, symbol)


@REAL_HCOMM
def test_real_hcomm_endpoint_mem_reg_export_smoke():
    endpoint = EndpointSpec.from_env()
    if endpoint is None and os.getenv("SIMPLER_HCOMM_ENDPOINT_HANDLE"):
        endpoint_handle = int(os.environ["SIMPLER_HCOMM_ENDPOINT_HANDLE"], 0)
        owns_endpoint = False
    elif endpoint is not None:
        runtime = HcommRuntime.from_env(required=True)
        endpoint_handle = runtime.endpoint_create(endpoint)
        owns_endpoint = True
    else:
        pytest.skip("set SIMPLER_HCOMM_ENDPOINT_IP or SIMPLER_HCOMM_ENDPOINT_HANDLE")

    runtime = HcommRuntime.from_env(required=True)
    data = bytearray(b"simpler-hcomm-smoke")
    addr = ctypes.addressof(ctypes.c_char.from_buffer(data))
    mem_handle = 0
    try:
        mem_handle = runtime.mem_reg(endpoint_handle, addr, len(data), tag="simpler-real-smoke")
        desc = runtime.mem_export(endpoint_handle, mem_handle)
        assert desc
        assert len(desc) >= ctypes.sizeof(_EndpointDesc)
    finally:
        if mem_handle:
            runtime.mem_unreg(endpoint_handle, mem_handle)
        if owns_endpoint:
            runtime.endpoint_destroy(endpoint_handle)


@REAL_HCOMM
def test_real_hcomm_client_precreated_channel_write_smoke():
    required = [
        "SIMPLER_HCOMM_ENDPOINT_HANDLE",
        "SIMPLER_HCOMM_CHANNEL_HANDLE",
        "SIMPLER_HCOMM_REMOTE_ADDR",
        "SIMPLER_HCOMM_REMOTE_NBYTES",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip(f"missing real channel smoke env: {', '.join(missing)}")

    runtime = HcommRuntime.from_env(required=True)
    client = HcommDataPlaneClient(runtime=runtime)
    remote_addr = int(os.environ["SIMPLER_HCOMM_REMOTE_ADDR"], 0)
    remote_nbytes = int(os.environ["SIMPLER_HCOMM_REMOTE_NBYTES"], 0)
    payload = bytes(range(min(remote_nbytes, 64)))
    local = ctypes.create_string_buffer(payload)
    handle = dispatch_pb2.TensorHandle(remote_addr=remote_addr, nbytes=remote_nbytes)

    client.write_handle(handle, ctypes.addressof(local), len(payload))
    client.fence()
    client.close()
