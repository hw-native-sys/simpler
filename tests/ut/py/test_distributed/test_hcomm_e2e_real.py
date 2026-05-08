import ctypes
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback

import pytest

from simpler.distributed.transport_backend import EndpointSpec, HcommRuntime


REAL_HCOMM_E2E = pytest.mark.skipif(
    os.getenv("SIMPLER_HCOMM_E2E_REAL_TEST") != "1",
    reason="set SIMPLER_HCOMM_E2E_REAL_TEST=1 to run the real HCOMM channel smoke test",
)


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server(q, lib_path: str, ip: str, port: int, payload_len: int) -> None:  # noqa: ANN001
    runtime = None
    endpoint = 0
    mem_handle = 0
    listening = False
    try:
        runtime = HcommRuntime(lib_path=lib_path, required=True)
        endpoint = runtime.endpoint_create(EndpointSpec(ip=ip))
        target = bytearray(payload_len)
        mem_handle = runtime.mem_reg(
            endpoint,
            ctypes.addressof(ctypes.c_char.from_buffer(target)),
            len(target),
            tag="server-target",
        )
        desc = runtime.mem_export(endpoint, mem_handle)
        runtime.endpoint_start_listen(endpoint, port)
        listening = True
        q.put(("ready", desc))
        remote_endpoint = EndpointSpec(ip=ip).to_ctypes()
        channel = runtime.channel_create(
            endpoint,
            remote_endpoint=remote_endpoint,
            socket_handle=0,
            local_mem_handles=[mem_handle],
            notify_num=0,
            engine=0,
            role=1,
            port=port,
        )
        deadline = time.time() + 30
        while bytes(target) == b"\x00" * payload_len and time.time() < deadline:
            time.sleep(0.05)
        runtime.channel_destroy([channel])
        q.put(("result", bytes(target)))
    except Exception:
        q.put(("error", "server", traceback.format_exc()))
    finally:
        if runtime is not None and listening:
            runtime.endpoint_stop_listen(endpoint, port)
        if runtime is not None and mem_handle:
            runtime.mem_unreg(endpoint, mem_handle)
        if runtime is not None and endpoint:
            runtime.endpoint_destroy(endpoint)


def _client(q, lib_path: str, ip: str, port: int, payload: bytes, desc: bytes) -> None:  # noqa: ANN001
    runtime = None
    endpoint = 0
    staging_handle = 0
    try:
        runtime = HcommRuntime(lib_path=lib_path, required=True)
        endpoint = runtime.endpoint_create(EndpointSpec(ip=ip))
        staging = bytearray(payload)
        staging_handle = runtime.mem_reg(
            endpoint,
            ctypes.addressof(ctypes.c_char.from_buffer(staging)),
            len(staging),
            tag="client-staging",
        )
        remote = runtime.mem_import(endpoint, desc)
        remote_endpoint = EndpointSpec(ip=ip).to_ctypes()
        channel = runtime.channel_create(
            endpoint,
            remote_endpoint=remote_endpoint,
            socket_handle=0,
            local_mem_handles=[staging_handle],
            notify_num=0,
            engine=0,
            role=0,
            port=port,
        )
        runtime.write_with_notify(
            channel,
            remote.remote_addr,
            ctypes.addressof(ctypes.c_char.from_buffer(staging)),
            len(staging),
            0,
        )
        runtime.channel_fence(channel)
        runtime.channel_destroy([channel])
        runtime.mem_unimport(endpoint, desc)
    except Exception:
        q.put(("error", "client", traceback.format_exc()))
    finally:
        if runtime is not None and staging_handle:
            runtime.mem_unreg(endpoint, staging_handle)
        if runtime is not None and endpoint:
            runtime.endpoint_destroy(endpoint)


@REAL_HCOMM_E2E
def test_real_hcomm_cpu_roce_channel_write_smoke():
    lib_path = os.getenv("SIMPLER_HCOMM_LIB")
    if not lib_path:
        pytest.skip("set SIMPLER_HCOMM_LIB")
    ip = os.getenv("SIMPLER_HCOMM_ENDPOINT_IP") or os.getenv("SIMPLER_RXE_SERVER_IP") or "192.168.0.243"
    port = int(os.getenv("SIMPLER_HCOMM_CHANNEL_PORT") or _unused_tcp_port())
    ready_timeout = int(os.getenv("SIMPLER_HCOMM_E2E_READY_TIMEOUT", "180"), 0)
    join_timeout = int(os.getenv("SIMPLER_HCOMM_E2E_JOIN_TIMEOUT", "180"), 0)
    payload = b"simpler-hcomm-e2e-smoke"
    q = mp.Queue()
    server = mp.Process(target=_server, args=(q, lib_path, ip, port, len(payload)))
    server.start()
    try:
        kind, *items = q.get(timeout=ready_timeout)
    except queue.Empty:
        server.terminate()
        server.join(timeout=5)
        pytest.fail(f"server did not publish HCOMM endpoint descriptor within {ready_timeout}s")
    if kind == "error":
        server.join(timeout=5)
        pytest.fail(f"{items[0]} failed before ready:\n{items[1]}")
    assert kind == "ready"
    desc = items[0]
    client = mp.Process(target=_client, args=(q, lib_path, ip, port, payload, desc))
    client.start()
    client.join(timeout=join_timeout)
    server.join(timeout=join_timeout)
    if client.is_alive():
        client.terminate()
        client.join(timeout=5)
    if server.is_alive():
        server.terminate()
        server.join(timeout=5)
    result = None
    errors = []
    while True:
        try:
            kind, *items = q.get_nowait()
        except queue.Empty:
            break
        if kind == "result":
            result = items[0]
        elif kind == "error":
            errors.append(tuple(items))
    _xfail_if_stock_hcomm_host_roce_unsupported(errors)
    assert client.exitcode == 0, errors
    assert server.exitcode == 0, errors
    if errors:
        role, tb = errors[0]
        pytest.fail(f"{role} failed:\n{tb}")
    if result is None:
        pytest.fail("server did not publish HCOMM smoke result")
    assert result == payload


def _xfail_if_stock_hcomm_host_roce_unsupported(errors):
    if os.getenv("SIMPLER_HCOMM_E2E_REQUIRE_CHANNEL") == "1":
        return
    for role, tb in errors:
        if "HcommChannelCreate failed with HcclResult=5" in tb:
            pytest.xfail(
                f"stock HCOMM Host CPU RoCE channel is unsupported in this environment ({role}: HCCL_E_NOT_SUPPORT)"
            )
