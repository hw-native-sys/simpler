import ctypes
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from simpler.distributed.l3_daemon import L3Daemon
from simpler.distributed.transport_backend import EndpointSpec, HcommRuntime, RxeRuntime, _EndpointDesc
from simpler.task_interface import CallConfig, ContinuousTensor, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker


REAL_E2E = pytest.mark.skipif(
    os.getenv("SIMPLER_REAL_E2E_TEST") != "1",
    reason="set SIMPLER_REAL_E2E_TEST=1 to run the real distributed data-plane smoke",
)


def _start_daemon():
    daemon = L3Daemon(0, lambda: Worker(level=3, num_sub_workers=1))
    port = daemon.start()
    return daemon, f"127.0.0.1:{port}"


def _start_daemon_with_transport(tensor_transport: str):
    daemon = L3Daemon(0, lambda: Worker(level=3, num_sub_workers=1), tensor_transport=tensor_transport)
    port = daemon.start()
    return daemon, f"127.0.0.1:{port}"


@REAL_E2E
def test_real_l4_l3_tensorpool_handle_e2e(tmp_path):
    result = tmp_path / "remote_tensor_sum.txt"
    payload = bytes(range(256)) * 32
    in_buf = ctypes.create_string_buffer(payload)
    out_payload = bytes((255 - (i % 256) for i in range(len(payload))))
    out_buf = ctypes.create_string_buffer(b"\x00" * len(out_payload))
    daemon, endpoint = _start_daemon()

    try:
        w4 = Worker(level=4, num_sub_workers=0)

        def l3_orch(orch, args, config):
            in_tensor = args.tensor(0)
            out_tensor = args.tensor(1)
            data = ctypes.string_at(int(in_tensor.data), int(in_tensor.nbytes()))
            result.write_text(f"{len(data)}:{sum(data)}")
            ctypes.memmove(int(out_tensor.data), out_payload, len(out_payload))

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint)
        w4.init()

        def l4_orch(orch, args, config):
            sub_args = TaskArgs()
            sub_args.add_tensor(
                ContinuousTensor.make(ctypes.addressof(in_buf), (len(payload),), DataType.UINT8),
                TensorArgType.INPUT,
            )
            sub_args.add_tensor(
                ContinuousTensor.make(ctypes.addressof(out_buf), (len(out_payload),), DataType.UINT8),
                TensorArgType.OUTPUT_EXISTING,
            )
            orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert result.read_text() == f"{len(payload)}:{sum(payload)}"
        assert bytes(out_buf.raw) == out_payload + b"\x00"
    finally:
        daemon.stop()


@REAL_E2E
def test_real_rxe_ibverbs_smoke(tmp_path):
    binary = os.getenv("SIMPLER_RXE_PINGPONG") or shutil.which("ibv_rc_pingpong")
    if not binary:
        pytest.skip("ibv_rc_pingpong is not available")

    device = os.getenv("SIMPLER_RXE_DEVICE") or _first_existing_rxe_device()
    if not device:
        pytest.skip("no rxe* device found under /sys/class/infiniband")

    gid_index = os.getenv("SIMPLER_RXE_GID_INDEX")
    server_ip = os.getenv("SIMPLER_RXE_SERVER_IP")
    if not gid_index or not server_ip:
        inferred = _find_ipv4_gid(device)
        if inferred is None:
            pytest.skip(f"no IPv4-mapped GID found for {device}; set SIMPLER_RXE_GID_INDEX and SIMPLER_RXE_SERVER_IP")
        inferred_gid_index, inferred_ip = inferred
        gid_index = gid_index or inferred_gid_index
        server_ip = server_ip or inferred_ip

    server_log = tmp_path / "rxe_rc_server.log"
    client_log = tmp_path / "rxe_rc_client.log"
    server_cmd = [binary, "-d", device, "-i", "1", "-g", gid_index]
    client_cmd = [binary, "-d", device, "-i", "1", "-g", gid_index, server_ip]

    with server_log.open("wb") as server_out:
        server = subprocess.Popen(server_cmd, stdout=server_out, stderr=subprocess.STDOUT)
    try:
        time.sleep(1.0)
        with client_log.open("wb") as client_out:
            client = subprocess.run(client_cmd, stdout=client_out, stderr=subprocess.STDOUT, timeout=15, check=False)
        try:
            server_rc = server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server_rc = server.wait(timeout=5)
    finally:
        if server.poll() is None:
            server.kill()
            server.wait(timeout=5)

    server_text = server_log.read_text(encoding="utf-8", errors="replace")
    client_text = client_log.read_text(encoding="utf-8", errors="replace")
    assert client.returncode == 0 and server_rc == 0, (
        f"RXE RC pingpong failed for device={device}, gid_index={gid_index}, server_ip={server_ip}\n"
        f"server rc={server_rc}\n{server_text}\nclient rc={client.returncode}\n{client_text}"
    )
    assert "bytes in" in server_text
    assert "bytes in" in client_text


@REAL_E2E
def test_real_l4_l3_rxe_tensor_transport_e2e(tmp_path, monkeypatch):
    runtime = RxeRuntime.from_env(required=False)
    if not runtime.available or not runtime.device or not runtime.server_ip:
        pytest.skip(runtime.unavailable_reason() or "RXE runtime is not configured")

    monkeypatch.setenv("SIMPLER_TENSOR_TRANSPORT", "rxe")
    result = tmp_path / "remote_rxe_tensor_sum.txt"
    payload = bytes((i * 7) % 251 for i in range(12 * 1024))
    in_buf = ctypes.create_string_buffer(payload, len(payload))
    out_payload = bytes((i * 11) % 253 for i in range(len(payload)))
    out_buf = ctypes.create_string_buffer(b"\x00" * len(out_payload), len(out_payload))
    daemon, endpoint = _start_daemon_with_transport("rxe")

    try:
        w4 = Worker(level=4, num_sub_workers=0)

        def l3_orch(orch, args, config):
            in_tensor = args.tensor(0)
            out_tensor = args.tensor(1)
            data = ctypes.string_at(int(in_tensor.data), int(in_tensor.nbytes()))
            result.write_text(f"{len(data)}:{sum(data)}")
            ctypes.memmove(int(out_tensor.data), out_payload, len(out_payload))

        l3_cid = w4.register(l3_orch)
        w4.add_remote_worker(endpoint, tensor_transport="rxe")
        w4.init()

        def l4_orch(orch, args, config):
            sub_args = TaskArgs()
            sub_args.add_tensor(
                ContinuousTensor.make(ctypes.addressof(in_buf), (len(payload),), DataType.UINT8),
                TensorArgType.INPUT,
            )
            sub_args.add_tensor(
                ContinuousTensor.make(ctypes.addressof(out_buf), (len(out_payload),), DataType.UINT8),
                TensorArgType.OUTPUT_EXISTING,
            )
            orch.submit_next_level(l3_cid, sub_args, CallConfig())

        w4.run(l4_orch)
        w4.close()
        assert result.read_text() == f"{len(payload)}:{sum(payload)}"
        assert bytes(out_buf.raw) == out_payload
    finally:
        daemon.stop()


@REAL_E2E
def test_real_hcomm_endpoint_mem_export_smoke():
    lib_path = os.getenv("SIMPLER_HCOMM_LIB")
    endpoint = EndpointSpec.from_env()
    if not lib_path or endpoint is None:
        pytest.skip("set SIMPLER_HCOMM_LIB and SIMPLER_HCOMM_ENDPOINT_IP")

    runtime = HcommRuntime.from_env(required=True)
    endpoint_handle = runtime.endpoint_create(endpoint)
    data = bytearray(b"simpler-real-e2e-hcomm")
    mem_handle = 0
    try:
        mem_handle = runtime.mem_reg(
            endpoint_handle,
            ctypes.addressof(ctypes.c_char.from_buffer(data)),
            len(data),
            tag="simpler-real-e2e",
        )
        desc = runtime.mem_export(endpoint_handle, mem_handle)
        assert desc
        assert len(desc) >= ctypes.sizeof(_EndpointDesc)
    finally:
        if mem_handle:
            runtime.mem_unreg(endpoint_handle, mem_handle)
        runtime.endpoint_destroy(endpoint_handle)


def _first_existing_rxe_device() -> str | None:
    infiniband = Path("/sys/class/infiniband")
    if not infiniband.exists():
        return None
    for path in sorted(infiniband.iterdir()):
        if path.name.startswith("rxe"):
            return path.name
    return None


def _ipv4_from_gid(gid: str) -> str | None:
    parts = gid.strip().split(":")
    if len(parts) != 8 or parts[5].lower() != "ffff":
        return None
    try:
        hi = int(parts[6], 16)
        lo = int(parts[7], 16)
    except ValueError:
        return None
    return ".".join(str(octet) for octet in (hi >> 8, hi & 0xFF, lo >> 8, lo & 0xFF))


def _find_ipv4_gid(device: str) -> tuple[str, str] | None:
    gid_dir = Path("/sys/class/infiniband") / device / "ports" / "1" / "gids"
    if not gid_dir.exists():
        return None
    for path in sorted(gid_dir.iterdir(), key=lambda item: int(item.name) if item.name.isdigit() else item.name):
        ip = _ipv4_from_gid(path.read_text(encoding="ascii").strip())
        if ip:
            return path.name, ip
    return None
