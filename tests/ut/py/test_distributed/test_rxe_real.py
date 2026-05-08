import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


REAL_RXE = pytest.mark.skipif(
    os.getenv("SIMPLER_RXE_REAL_TEST") != "1",
    reason="set SIMPLER_RXE_REAL_TEST=1 to run the local RXE/ibverbs smoke test",
)


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


@REAL_RXE
def test_real_rxe_rc_pingpong_smoke(tmp_path):
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
