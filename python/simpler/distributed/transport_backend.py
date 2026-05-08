"""Optional tensor data-plane transport backends.

The default distributed data plane still uses gRPC chunk streaming.  This
module adds a narrow backend boundary for transports that can expose registered
memory to the peer, and a first HCOMM C-API facade that can be enabled on
systems where HCOMM endpoint/channel resources are already available.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import ipaddress
import json
import os
import shlex
import struct
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class TransportBackendError(RuntimeError):
    pass


class TransportUnavailable(TransportBackendError):
    pass


@dataclass(frozen=True)
class RegisteredRegion:
    remote_addr: int
    rkey: int = 0
    transport: str = "grpc"
    transport_desc: bytes = b""


@dataclass(frozen=True)
class ImportedMemory:
    remote_addr: int
    nbytes: int
    mem_type: int = 1


@dataclass
class _HcommStagingBuffer:
    data: Optional[bytearray]
    addr: int
    nbytes: int
    mem_handle: int
    owned: bool


@dataclass
class _RxeServerRegion:
    region: RegisteredRegion
    handle: int


class TensorTransportBackend:
    """Storage-side transport hook used by ``TensorPool``."""

    name = "grpc"

    @property
    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def register_region(self, data: bytearray, *, tag: str) -> RegisteredRegion:
        return RegisteredRegion(remote_addr=_buffer_addr(data), transport=self.name)

    def unregister_region(self, region: RegisteredRegion) -> None:
        del region


class GrpcTensorTransport(TensorTransportBackend):
    """Current Python byte-pool transport."""

    name = "grpc"


class RxeTensorTransport(TensorTransportBackend):
    """Registers TensorPool buffers for direct RXE/ibverbs RDMA writes."""

    name = "rxe"

    def __init__(self, runtime: Optional["RxeRuntime"] = None) -> None:
        self.runtime = runtime or RxeRuntime.from_env(required=False)
        self._regions: dict[int, _RxeServerRegion] = {}

    @classmethod
    def from_env(cls) -> "RxeTensorTransport":
        return cls()

    @property
    def available(self) -> bool:
        return self.runtime.available and self.runtime.device is not None and self.runtime.server_ip is not None

    def unavailable_reason(self) -> str:
        if not self.runtime.available:
            return self.runtime.unavailable_reason()
        if self.runtime.device is None:
            return "no RXE device found; set SIMPLER_RXE_DEVICE"
        if self.runtime.server_ip is None:
            return "no IPv4 GID found for RXE; set SIMPLER_RXE_SERVER_IP and SIMPLER_RXE_GID_INDEX"
        return ""

    def register_region(self, data: bytearray, *, tag: str) -> RegisteredRegion:
        del tag
        if not data:
            return RegisteredRegion(remote_addr=0, rkey=0, transport=self.name)
        if not self.available:
            raise TransportUnavailable(self.unavailable_reason())
        addr = _buffer_addr(data)
        desc, server_handle = self.runtime.server_start(addr, len(data))
        payload = _encode_rxe_desc(desc, self.runtime.device or "", self.runtime.gid_index)
        region = RegisteredRegion(
            remote_addr=int(desc.addr),
            rkey=int(desc.rkey),
            transport=self.name,
            transport_desc=payload,
        )
        self._regions[addr] = _RxeServerRegion(region=region, handle=server_handle)
        return region

    def unregister_region(self, region: RegisteredRegion) -> None:
        item = self._regions.pop(int(region.remote_addr), None)
        if item is None:
            return
        self.runtime.server_stop(item.handle)

    def refresh_region(self, region: RegisteredRegion, data: bytearray, *, tag: str) -> RegisteredRegion:
        self.unregister_region(region)
        return self.register_region(data, tag=tag)

    def close(self) -> None:
        for item in list(self._regions.values()):
            self.unregister_region(item.region)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class HcommTensorTransport(TensorTransportBackend):
    """Registers TensorPool byte buffers with HCOMM when configured.

    This backend is intentionally conservative: HCOMM needs an endpoint handle
    created from a rank graph/endpoint description.  For this first integration
    layer we accept an already-created endpoint handle via environment or
    constructor and expose HCOMM memory descriptors in TensorHandle metadata.
    """

    name = "hcomm"

    def __init__(self, runtime: Optional["HcommRuntime"] = None, endpoint_handle: int = 0) -> None:
        self.runtime = runtime or HcommRuntime.from_env(required=False)
        self.endpoint_handle = _parse_handle(endpoint_handle or os.getenv("SIMPLER_HCOMM_ENDPOINT_HANDLE", "0"))
        self._owns_endpoint = False
        if self.endpoint_handle == 0 and self.runtime.available and hasattr(self.runtime, "endpoint_create"):
            endpoint = EndpointSpec.from_env()
            if endpoint is not None:
                self.endpoint_handle = self.runtime.endpoint_create(endpoint)
                self._owns_endpoint = True
        self._regions: dict[int, tuple[RegisteredRegion, int, int]] = {}

    @classmethod
    def from_env(cls) -> "HcommTensorTransport":
        return cls()

    @property
    def available(self) -> bool:
        return self.runtime.available and self.endpoint_handle != 0

    def unavailable_reason(self) -> str:
        if not self.runtime.available:
            return self.runtime.unavailable_reason()
        if self.endpoint_handle == 0:
            return "SIMPLER_HCOMM_ENDPOINT_HANDLE is not set"
        return ""

    def register_region(self, data: bytearray, *, tag: str) -> RegisteredRegion:
        if not data:
            return RegisteredRegion(remote_addr=0, rkey=0, transport=self.name)
        if not self.available:
            raise TransportUnavailable(self.unavailable_reason())
        addr = _buffer_addr(data)
        size = len(data)
        mem_handle = self.runtime.mem_reg(self.endpoint_handle, addr, size, tag=tag)
        try:
            desc = self.runtime.mem_export(self.endpoint_handle, mem_handle)
        except Exception:
            self.runtime.mem_unreg(self.endpoint_handle, mem_handle)
            raise
        region = RegisteredRegion(remote_addr=addr, rkey=0, transport=self.name, transport_desc=desc)
        self._regions[addr] = (region, self.endpoint_handle, mem_handle)
        return region

    def unregister_region(self, region: RegisteredRegion) -> None:
        item = self._regions.pop(int(region.remote_addr), None)
        if item is None:
            return
        _, endpoint_handle, mem_handle = item
        self.runtime.mem_unreg(endpoint_handle, mem_handle)

    def close(self) -> None:
        for region, _, _ in list(self._regions.values()):
            self.unregister_region(region)
        if self._owns_endpoint and self.endpoint_handle:
            self.runtime.endpoint_destroy(self.endpoint_handle)
            self.endpoint_handle = 0
            self._owns_endpoint = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class HcommDataPlaneClient:
    """Thin caller for HCOMM channel primitives on the L4 side."""

    def __init__(
        self,
        runtime: Optional["HcommRuntime"] = None,
        *,
        endpoint_handle: int = 0,
        channel_handle: int = 0,
        socket_handle: int = 0,
        local_mem_handle: int = 0,
        remote_notify_idx: int = 0,
    ) -> None:
        self.runtime = runtime or HcommRuntime.from_env(required=False)
        self.endpoint_handle = _parse_handle(endpoint_handle or os.getenv("SIMPLER_HCOMM_ENDPOINT_HANDLE", "0"))
        self.channel_handle = _parse_handle(channel_handle or os.getenv("SIMPLER_HCOMM_CHANNEL_HANDLE", "0"))
        self.socket_handle = _parse_handle(socket_handle or os.getenv("SIMPLER_HCOMM_SOCKET_HANDLE", "0"))
        self.local_mem_handle = _parse_handle(local_mem_handle or os.getenv("SIMPLER_HCOMM_LOCAL_MEM_HANDLE", "0"))
        self.remote_notify_idx = int(os.getenv("SIMPLER_HCOMM_REMOTE_NOTIFY_IDX", str(remote_notify_idx)), 0)
        self.notify_num = int(os.getenv("SIMPLER_HCOMM_NOTIFY_NUM", "1"), 0)
        self.engine = int(os.getenv("SIMPLER_HCOMM_ENGINE", "0"), 0)  # COMM_ENGINE_CPU
        self.channel_role = _hcomm_socket_role(os.getenv("SIMPLER_HCOMM_CHANNEL_ROLE", "client"))
        self.channel_port = int(os.getenv("SIMPLER_HCOMM_CHANNEL_PORT", "60001"), 0)
        self._owns_channel = False
        self._owns_endpoint = False
        if self.endpoint_handle == 0 and self.runtime.available and hasattr(self.runtime, "endpoint_create"):
            endpoint = EndpointSpec.from_env()
            if endpoint is not None:
                self.endpoint_handle = self.runtime.endpoint_create(endpoint)
                self._owns_endpoint = True
        self._imports: dict[bytes, ImportedMemory] = {}
        self._staging: Optional[_HcommStagingBuffer] = None

    @classmethod
    def from_env(cls) -> "HcommDataPlaneClient":
        return cls()

    @property
    def available(self) -> bool:
        return self.runtime.available and (self.channel_handle != 0 or self._can_create_channel)

    @property
    def _can_create_channel(self) -> bool:
        return self.endpoint_handle != 0

    def unavailable_reason(self) -> str:
        if not self.runtime.available:
            return self.runtime.unavailable_reason()
        if self.channel_handle == 0 and not self._can_create_channel:
            return "SIMPLER_HCOMM_CHANNEL_HANDLE is not set, and automatic channel creation requires an endpoint handle"
        return ""

    def resolve_remote_memory(self, handle) -> ImportedMemory:  # noqa: ANN001
        desc = bytes(getattr(handle, "transport_desc", b""))
        if desc and self.endpoint_handle:
            cached = self._imports.get(desc)
            if cached is not None:
                return cached
            imported = self.runtime.mem_import(self.endpoint_handle, desc)
            self._imports[desc] = imported
            return imported
        return ImportedMemory(remote_addr=int(handle.remote_addr), nbytes=int(handle.nbytes))

    def write_handle(self, handle, local_addr: int, nbytes: int) -> None:  # noqa: ANN001
        remote_mem = self.resolve_remote_memory(handle)
        if int(nbytes) > int(remote_mem.nbytes):
            raise TransportBackendError(
                f"HCOMM remote memory too small: write={int(nbytes)}, remote={int(remote_mem.nbytes)}"
            )
        staging = self._stage_local(local_addr, int(nbytes))
        self.ensure_channel(handle)
        self.write_with_notify(remote_mem.remote_addr, staging.addr, nbytes)

    def ensure_channel(self, handle=None) -> int:  # noqa: ANN001
        if self.channel_handle:
            return self.channel_handle
        if not self._can_create_channel:
            raise TransportUnavailable(self.unavailable_reason())
        self._ensure_staging(1)
        desc = bytes(getattr(handle, "transport_desc", b"")) if handle is not None else b""
        if not desc:
            raise TransportUnavailable("HCOMM channel creation requires TensorHandle.transport_desc")
        remote_endpoint = _endpoint_from_transport_desc(desc)
        self.channel_handle = self.runtime.channel_create(
            self.endpoint_handle,
            remote_endpoint=remote_endpoint,
            socket_handle=self.socket_handle,
            local_mem_handles=[self._local_mem_handle()],
            notify_num=self.notify_num,
            engine=self.engine,
            role=self.channel_role,
            port=self.channel_port,
        )
        self._owns_channel = True
        return self.channel_handle

    def remote_mems(self) -> list[ImportedMemory]:
        if not self.channel_handle:
            raise TransportUnavailable(self.unavailable_reason())
        return self.runtime.channel_get_remote_mem(self.channel_handle)

    def write_with_notify(self, remote_addr: int, local_addr: int, nbytes: int) -> None:
        if not self.available:
            raise TransportUnavailable(self.unavailable_reason())
        self.runtime.write_with_notify(
            self.channel_handle,
            int(remote_addr),
            int(local_addr),
            int(nbytes),
            self.remote_notify_idx,
        )

    def fence(self) -> None:
        if not self.available:
            raise TransportUnavailable(self.unavailable_reason())
        self.runtime.channel_fence(self.channel_handle)

    def close(self) -> None:
        for desc in list(self._imports):
            if self.endpoint_handle:
                try:
                    self.runtime.mem_unimport(self.endpoint_handle, desc)
                except Exception:
                    pass
            self._imports.pop(desc, None)
        if self._owns_channel and self.channel_handle:
            try:
                self.runtime.channel_destroy([self.channel_handle])
            except Exception:
                pass
            self.channel_handle = 0
            self._owns_channel = False
        if self._staging is not None and self._staging.owned:
            try:
                self.runtime.mem_unreg(self.endpoint_handle, self._staging.mem_handle)
            except Exception:
                pass
        self._staging = None
        if self._owns_endpoint and self.endpoint_handle:
            try:
                self.runtime.endpoint_destroy(self.endpoint_handle)
            except Exception:
                pass
            self.endpoint_handle = 0
            self._owns_endpoint = False

    def _stage_local(self, local_addr: int, nbytes: int) -> "_HcommStagingBuffer":
        if self.local_mem_handle:
            self._staging = _HcommStagingBuffer(
                data=None,
                addr=int(local_addr),
                nbytes=int(nbytes),
                mem_handle=self.local_mem_handle,
                owned=False,
            )
            return self._staging
        staging = self._ensure_staging(nbytes)
        if nbytes:
            ctypes.memmove(staging.addr, int(local_addr), int(nbytes))
        return staging

    def _ensure_staging(self, nbytes: int) -> "_HcommStagingBuffer":
        if self._staging is not None and self._staging.nbytes >= int(nbytes):
            return self._staging
        if self._staging is not None and self._staging.owned:
            self.runtime.mem_unreg(self.endpoint_handle, self._staging.mem_handle)
            self._staging = None
        if self.local_mem_handle:
            if nbytes <= 0:
                raise TransportUnavailable("external SIMPLER_HCOMM_LOCAL_MEM_HANDLE requires nonzero staged writes")
            # External channel users are responsible for ensuring this handle covers
            # the local source address.  Automatic channel creation uses owned staging.
            self._staging = _HcommStagingBuffer(
                data=None,
                addr=0,
                nbytes=int(nbytes),
                mem_handle=self.local_mem_handle,
                owned=False,
            )
            return self._staging
        if not self.endpoint_handle:
            raise TransportUnavailable("HCOMM staging requires an endpoint handle")
        capacity = max(1, int(nbytes))
        data = bytearray(capacity)
        addr = _buffer_addr(data)
        mem_handle = self.runtime.mem_reg(self.endpoint_handle, addr, capacity, tag="simpler-l4-staging")
        self._staging = _HcommStagingBuffer(data=data, addr=addr, nbytes=capacity, mem_handle=mem_handle, owned=True)
        return self._staging

    def _local_mem_handle(self) -> int:
        if self.local_mem_handle:
            return self.local_mem_handle
        if self._staging is None:
            raise TransportUnavailable("HCOMM channel creation requires a staged local buffer")
        return self._staging.mem_handle


class RxeDataPlaneClient:
    """L4-side RXE/ibverbs writer for TensorHandle metadata."""

    def __init__(self, runtime: Optional["RxeRuntime"] = None) -> None:
        self.runtime = runtime or RxeRuntime.from_env(required=False)

    @classmethod
    def from_env(cls) -> "RxeDataPlaneClient":
        return cls()

    @property
    def available(self) -> bool:
        return self.runtime.available and self.runtime.device is not None

    def unavailable_reason(self) -> str:
        if not self.runtime.available:
            return self.runtime.unavailable_reason()
        if self.runtime.device is None:
            return "no RXE device found; set SIMPLER_RXE_DEVICE"
        return ""

    def write_handle(self, handle, local_addr: int, nbytes: int) -> None:  # noqa: ANN001
        if not self.available:
            raise TransportUnavailable(self.unavailable_reason())
        desc = _decode_rxe_desc(bytes(getattr(handle, "transport_desc", b"")))
        if int(nbytes) > int(desc.size):
            raise TransportBackendError(f"RXE remote memory too small: write={int(nbytes)}, remote={int(desc.size)}")
        self.runtime.write(desc.ip, desc.port, int(local_addr), int(nbytes), gid_index=desc.gid_index)

    def fence(self) -> None:
        return

    def close(self) -> None:
        return


class RxeRuntime:
    """Runtime loader for the Simpler-owned RXE/ibverbs helper."""

    def __init__(self, lib_path: Optional[str] = None, *, required: bool = False) -> None:
        self.device = os.getenv("SIMPLER_RXE_DEVICE") or _first_existing_rxe_device()
        self.gid_index: int = int(os.getenv("SIMPLER_RXE_GID_INDEX", "0"), 0)
        self.server_ip = os.getenv("SIMPLER_RXE_SERVER_IP")
        if self.device and (not self.server_ip or "SIMPLER_RXE_GID_INDEX" not in os.environ):
            inferred = _find_rxe_ipv4_gid(self.device)
            if inferred is not None:
                inferred_gid_index, inferred_ip = inferred
                if "SIMPLER_RXE_GID_INDEX" not in os.environ:
                    self.gid_index = inferred_gid_index
                self.server_ip = self.server_ip or inferred_ip
        self._lib_path = lib_path or os.getenv("SIMPLER_RXE_HELPER_LIB")
        self._lib = None
        self._load_error = ""
        try:
            path = Path(self._lib_path).expanduser().resolve() if self._lib_path else _build_rxe_verbs_helper()
            self._preload_dependencies()
            self._lib = ctypes.CDLL(str(path), mode=getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0))
            self._lib_path = str(path)
            self._bind_symbols()
        except (OSError, TransportBackendError, TransportUnavailable) as e:
            self._load_error = str(e)
            self._lib = None
        if required and self._lib is None:
            raise TransportUnavailable(self.unavailable_reason())

    @classmethod
    def from_env(cls, *, required: bool = False) -> "RxeRuntime":
        return cls(required=required)

    @property
    def available(self) -> bool:
        return self._lib is not None

    def unavailable_reason(self) -> str:
        if self._lib is not None:
            return ""
        return self._load_error or "RXE helper is unavailable"

    def server_start(self, addr: int, size: int) -> tuple["_RxeServerDesc", int]:
        self._require()
        if not self.device or not self.server_ip:
            raise TransportUnavailable(self.unavailable_reason() or "RXE device/server IP is not configured")
        desc = _RxeServerDesc()
        handle = ctypes.c_void_p()
        ret = self._lib.simpler_rxe_server_start(
            self.device.encode(),
            ctypes.c_int(int(self.gid_index)),
            self.server_ip.encode(),
            ctypes.c_void_p(int(addr)),
            ctypes.c_uint64(int(size)),
            ctypes.byref(desc),
            ctypes.byref(handle),
        )
        _check_rxe(ret, "simpler_rxe_server_start")
        return desc, int(handle.value or 0)

    def server_stop(self, handle: int) -> None:
        if self._lib is None or not handle:
            return
        self._lib.simpler_rxe_server_stop(ctypes.c_void_p(int(handle)))

    def write(self, ip: str, port: int, local_addr: int, size: int, *, gid_index: Optional[int] = None) -> None:
        self._require()
        if not self.device:
            raise TransportUnavailable("no RXE device found; set SIMPLER_RXE_DEVICE")
        ret = self._lib.simpler_rxe_write(
            self.device.encode(),
            ctypes.c_int(int(self.gid_index if gid_index is None else gid_index)),
            str(ip).encode(),
            ctypes.c_uint16(int(port)),
            ctypes.c_void_p(int(local_addr)),
            ctypes.c_uint64(int(size)),
        )
        _check_rxe(ret, "simpler_rxe_write")

    def _require(self) -> None:
        if self._lib is None:
            raise TransportUnavailable(self.unavailable_reason())

    def _bind_symbols(self) -> None:
        assert self._lib is not None
        self._lib.simpler_rxe_server_start.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(_RxeServerDesc),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.simpler_rxe_server_start.restype = ctypes.c_int
        self._lib.simpler_rxe_server_stop.argtypes = [ctypes.c_void_p]
        self._lib.simpler_rxe_server_stop.restype = None
        self._lib.simpler_rxe_write.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint16,
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        self._lib.simpler_rxe_write.restype = ctypes.c_int

    def _preload_dependencies(self) -> None:
        lib_dir = _rxe_lib_dir()
        if lib_dir is not None:
            _prepend_env_path("LD_LIBRARY_PATH", lib_dir)
            driver_dir = lib_dir / "libibverbs"
            if driver_dir.is_dir():
                _prepend_env_path("LD_LIBRARY_PATH", driver_dir)
            lib = lib_dir / "libibverbs.so.1"
            if lib.exists():
                ctypes.CDLL(str(lib), mode=getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0))


class HcommRuntime:
    """Runtime loader for the HCOMM experimental C API."""

    def __init__(self, lib_path: Optional[str] = None, *, required: bool = False) -> None:
        self._lib_path = lib_path or os.getenv("SIMPLER_HCOMM_LIB") or _find_hcomm_library()
        self._lib = None
        self._acl = None
        self._load_error = ""
        self._acl_load_error = ""
        if self._lib_path:
            try:
                self._ensure_acl_runtime()
                _preload_hcomm_dependencies(self._lib_path)
                _preload_hcomm_abi_shim(self._lib_path)
                self._lib = ctypes.CDLL(self._lib_path, mode=_hcomm_dlopen_mode())
                self._bind_symbols()
            except (AttributeError, OSError) as e:
                self._load_error = str(e)
                self._lib = None
        elif required:
            self._load_error = "HCOMM shared library not found; set SIMPLER_HCOMM_LIB"
        if required and self._lib is None:
            raise TransportUnavailable(self.unavailable_reason())

    @classmethod
    def from_env(cls, *, required: bool = False) -> "HcommRuntime":
        return cls(required=required)

    @property
    def available(self) -> bool:
        return self._lib is not None

    def unavailable_reason(self) -> str:
        if self._lib is not None:
            return ""
        return self._load_error or "HCOMM shared library not found; set SIMPLER_HCOMM_LIB"

    def endpoint_create(self, endpoint: "EndpointSpec") -> int:
        self._require()
        handle = ctypes.c_void_p()
        ret = self._lib.HcommEndpointCreate(ctypes.byref(endpoint.to_ctypes()), ctypes.byref(handle))
        _check_hcomm(ret, "HcommEndpointCreate")
        return int(handle.value or 0)

    def endpoint_destroy(self, endpoint_handle: int) -> None:
        self._require()
        ret = self._lib.HcommEndpointDestroy(ctypes.c_void_p(int(endpoint_handle)))
        _check_hcomm(ret, "HcommEndpointDestroy")

    def endpoint_start_listen(self, endpoint_handle: int, port: int) -> None:
        self._require()
        ret = self._lib.HcommEndpointStartListen(ctypes.c_void_p(int(endpoint_handle)), ctypes.c_uint32(int(port)), None)
        _check_hcomm(ret, "HcommEndpointStartListen")

    def endpoint_stop_listen(self, endpoint_handle: int, port: int) -> None:
        self._require()
        ret = self._lib.HcommEndpointStopListen(ctypes.c_void_p(int(endpoint_handle)), ctypes.c_uint32(int(port)))
        _check_hcomm(ret, "HcommEndpointStopListen")

    def mem_reg(self, endpoint_handle: int, addr: int, size: int, *, tag: str, mem_type: str = "host") -> int:
        self._require()
        mem = _HcommMem(
            type=_mem_type_value(mem_type),
            addr=ctypes.c_void_p(int(addr)),
            size=int(size),
        )
        handle = ctypes.c_void_p()
        ret = self._lib.HcommMemReg(
            ctypes.c_void_p(int(endpoint_handle)),
            tag.encode(),
            ctypes.byref(mem),
            ctypes.byref(handle),
        )
        _check_hcomm(ret, "HcommMemReg")
        return int(handle.value or 0)

    def mem_unreg(self, endpoint_handle: int, mem_handle: int) -> None:
        self._require()
        ret = self._lib.HcommMemUnreg(ctypes.c_void_p(int(endpoint_handle)), ctypes.c_void_p(int(mem_handle)))
        _check_hcomm(ret, "HcommMemUnreg")

    def mem_export(self, endpoint_handle: int, mem_handle: int) -> bytes:
        self._require()
        desc = ctypes.c_void_p()
        desc_len = ctypes.c_uint32()
        ret = self._lib.HcommMemExport(
            ctypes.c_void_p(int(endpoint_handle)),
            ctypes.c_void_p(int(mem_handle)),
            ctypes.byref(desc),
            ctypes.byref(desc_len),
        )
        _check_hcomm(ret, "HcommMemExport")
        if not desc.value or desc_len.value == 0:
            return b""
        return ctypes.string_at(desc.value, desc_len.value)

    def mem_import(self, endpoint_handle: int, desc: bytes) -> ImportedMemory:
        self._require()
        payload = bytes(desc)
        out = _HcommMem()
        ret = self._lib.HcommMemImport(
            ctypes.c_void_p(int(endpoint_handle)),
            ctypes.c_char_p(payload),
            ctypes.c_uint32(len(payload)),
            ctypes.byref(out),
        )
        _check_hcomm(ret, "HcommMemImport")
        return ImportedMemory(remote_addr=int(out.addr or 0), nbytes=int(out.size), mem_type=int(out.type))

    def mem_unimport(self, endpoint_handle: int, desc: bytes) -> None:
        self._require()
        payload = bytes(desc)
        ret = self._lib.HcommMemUnimport(
            ctypes.c_void_p(int(endpoint_handle)),
            ctypes.c_char_p(payload),
            ctypes.c_uint32(len(payload)),
        )
        _check_hcomm(ret, "HcommMemUnimport")

    def channel_create(
        self,
        endpoint_handle: int,
        *,
        remote_endpoint: "_EndpointDesc",
        socket_handle: int,
        local_mem_handles: list[int],
        notify_num: int = 1,
        engine: int = 0,
        role: int = 0,
        port: int = 60001,
        exchange_all_mems: bool = False,
    ) -> int:
        self._require()
        if not local_mem_handles and not exchange_all_mems:
            raise TransportBackendError("HCOMM channel creation requires at least one local mem handle")
        mem_array = None
        if local_mem_handles:
            mem_array_type = ctypes.c_void_p * len(local_mem_handles)
            mem_array = mem_array_type(*(ctypes.c_void_p(int(handle)) for handle in local_mem_handles))
        desc = _HcommChannelDesc()
        _init_hcomm_channel_desc(desc)
        desc.remoteEndpoint = remote_endpoint
        desc.notifyNum = int(notify_num)
        desc.exchangeAllMems = bool(exchange_all_mems)
        desc.memHandles = ctypes.cast(mem_array, ctypes.POINTER(ctypes.c_void_p)) if mem_array is not None else None
        desc.memHandleNum = len(local_mem_handles)
        desc.socket = ctypes.c_void_p(int(socket_handle))
        desc.role = int(role)
        desc.port = int(port)
        channels = (ctypes.c_uint64 * 1)()
        ret = self._lib.HcommChannelCreate(
            ctypes.c_void_p(int(endpoint_handle)),
            ctypes.c_int(int(engine)),
            ctypes.byref(desc),
            ctypes.c_uint32(1),
            channels,
        )
        _check_hcomm(ret, "HcommChannelCreate")
        return int(channels[0])

    def channel_destroy(self, channels: list[int]) -> None:
        self._require()
        if not channels:
            return
        arr_type = ctypes.c_uint64 * len(channels)
        arr = arr_type(*(int(channel) for channel in channels))
        ret = self._lib.HcommChannelDestroy(arr, ctypes.c_uint32(len(channels)))
        _check_hcomm(ret, "HcommChannelDestroy")

    def channel_get_remote_mem(self, channel_handle: int) -> list[ImportedMemory]:
        self._require()
        raise TransportUnavailable(
            "remote memory query is not available in the public HCOMM C API; "
            "use TensorHandle.transport_desc with HcommMemImport instead"
        )

    def write_with_notify(
        self,
        channel_handle: int,
        remote_addr: int,
        local_addr: int,
        nbytes: int,
        remote_notify_idx: int,
    ) -> None:
        self._require()
        ret = self._lib.HcommWriteWithNotifyNbi(
            ctypes.c_uint64(int(channel_handle)),
            ctypes.c_void_p(int(remote_addr)),
            ctypes.c_void_p(int(local_addr)),
            ctypes.c_uint64(int(nbytes)),
            ctypes.c_uint32(int(remote_notify_idx)),
        )
        _check_hcomm(ret, "HcommWriteWithNotifyNbi")

    def channel_fence(self, channel_handle: int) -> None:
        self._require()
        ret = self._lib.HcommChannelFence(ctypes.c_uint64(int(channel_handle)))
        _check_hcomm(ret, "HcommChannelFence")

    def _require(self) -> None:
        if self._lib is None:
            raise TransportUnavailable(self.unavailable_reason())

    def _ensure_acl_runtime(self) -> None:
        if os.getenv("SIMPLER_HCOMM_ACL_AUTO_INIT", "1").lower() in {"0", "false", "no", "off"}:
            return
        acl = self._load_acl_runtime()
        if acl is None:
            raise TransportUnavailable(self._acl_load_error or "libascendcl.so not found for HCOMM endpoint creation")

        device = ctypes.c_int(-1)
        if acl.aclrtGetDevice(ctypes.byref(device)) == 0:
            return

        ret = acl.aclInit(None)
        if ret not in (0, 100002):  # ACL_ERROR_REPEAT_INITIALIZE
            raise TransportBackendError(f"aclInit failed before HCOMM endpoint creation with aclError={ret}")

        device_id = int(os.getenv("SIMPLER_HCOMM_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")), 0)
        ret = acl.aclrtSetDevice(ctypes.c_int(device_id))
        if ret != 0:
            raise TransportBackendError(
                f"aclrtSetDevice failed before HCOMM endpoint creation with aclError={ret}, device_id={device_id}"
            )

    def _load_acl_runtime(self):  # noqa: ANN202
        if self._acl is not None:
            return self._acl
        search_dirs = _hcomm_dependency_dirs(self._lib_path or "")
        acl_path = _find_dependency_library("libascendcl.so", search_dirs)
        if acl_path is None:
            self._acl_load_error = "libascendcl.so not found; set ASCEND_HOME_PATH or SIMPLER_HCOMM_DEP_LIB_DIRS"
            return None
        _ensure_cann_runtime_env(acl_path)
        try:
            mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
            try:
                acl = ctypes.CDLL(str(acl_path), mode=mode)
            except OSError:
                _preload_needed(acl_path, search_dirs, set(), mode)
                acl = ctypes.CDLL(str(acl_path), mode=mode)
            acl.aclInit.argtypes = [ctypes.c_char_p]
            acl.aclInit.restype = ctypes.c_int
            acl.aclrtSetDevice.argtypes = [ctypes.c_int]
            acl.aclrtSetDevice.restype = ctypes.c_int
            acl.aclrtGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
            acl.aclrtGetDevice.restype = ctypes.c_int
            self._acl = acl
            return self._acl
        except OSError as e:
            self._acl_load_error = str(e)
            return None

    def _bind_symbols(self) -> None:
        assert self._lib is not None
        self._lib.HcommEndpointCreate.argtypes = [ctypes.POINTER(_EndpointDesc), ctypes.POINTER(ctypes.c_void_p)]
        self._lib.HcommEndpointCreate.restype = ctypes.c_int
        self._lib.HcommEndpointDestroy.argtypes = [ctypes.c_void_p]
        self._lib.HcommEndpointDestroy.restype = ctypes.c_int
        self._lib.HcommEndpointStartListen.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        self._lib.HcommEndpointStartListen.restype = ctypes.c_int
        self._lib.HcommEndpointStopListen.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._lib.HcommEndpointStopListen.restype = ctypes.c_int
        self._lib.HcommMemReg.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_HcommMem),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.HcommMemReg.restype = ctypes.c_int
        self._lib.HcommMemUnreg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._lib.HcommMemUnreg.restype = ctypes.c_int
        self._lib.HcommMemExport.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self._lib.HcommMemExport.restype = ctypes.c_int
        self._lib.HcommMemImport.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_HcommMem),
        ]
        self._lib.HcommMemImport.restype = ctypes.c_int
        self._lib.HcommMemUnimport.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        self._lib.HcommMemUnimport.restype = ctypes.c_int
        self._lib.HcommChannelCreate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_HcommChannelDesc),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.HcommChannelCreate.restype = ctypes.c_int
        self._lib.HcommChannelDestroy.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        self._lib.HcommChannelDestroy.restype = ctypes.c_int
        self._lib.HcommWriteWithNotifyNbi.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
        ]
        self._lib.HcommWriteWithNotifyNbi.restype = ctypes.c_int
        self._lib.HcommChannelFence.argtypes = [ctypes.c_uint64]
        self._lib.HcommChannelFence.restype = ctypes.c_int


@dataclass(frozen=True)
class EndpointSpec:
    ip: str
    loc_type: str = "host"
    host_id: int = 0
    dev_phy_id: int = 0
    super_dev_id: int = 0
    server_idx: int = 0
    super_pod_idx: int = 0

    @classmethod
    def from_env(cls) -> Optional["EndpointSpec"]:
        ip = os.getenv("SIMPLER_HCOMM_ENDPOINT_IP")
        if not ip:
            return None
        return cls(
            ip=ip,
            loc_type=os.getenv("SIMPLER_HCOMM_ENDPOINT_LOC_TYPE", "host"),
            host_id=int(os.getenv("SIMPLER_HCOMM_ENDPOINT_HOST_ID", "0"), 0),
            dev_phy_id=int(os.getenv("SIMPLER_HCOMM_ENDPOINT_DEV_PHY_ID", "0"), 0),
            super_dev_id=int(os.getenv("SIMPLER_HCOMM_ENDPOINT_SUPER_DEV_ID", "0"), 0),
            server_idx=int(os.getenv("SIMPLER_HCOMM_ENDPOINT_SERVER_IDX", "0"), 0),
            super_pod_idx=int(os.getenv("SIMPLER_HCOMM_ENDPOINT_SUPER_POD_IDX", "0"), 0),
        )

    def to_ctypes(self) -> "_EndpointDesc":
        endpoint = _EndpointDesc()
        ctypes.memset(ctypes.byref(endpoint), 0xFF, ctypes.sizeof(endpoint))
        endpoint.protocol = 1  # COMM_PROTOCOL_ROCE
        ip = ipaddress.ip_address(self.ip)
        if ip.version == 4:
            endpoint.commAddr.type = 0  # COMM_ADDR_TYPE_IP_V4
            endpoint.commAddr.raws[:4] = ip.packed
        else:
            endpoint.commAddr.type = 1  # COMM_ADDR_TYPE_IP_V6
            endpoint.commAddr.raws[:16] = ip.packed
        if self.loc_type == "device":
            endpoint.loc.locType = 0  # ENDPOINT_LOC_TYPE_DEVICE
            endpoint.loc.words[0] = int(self.dev_phy_id)
            endpoint.loc.words[1] = int(self.super_dev_id)
            endpoint.loc.words[2] = int(self.server_idx)
            endpoint.loc.words[3] = int(self.super_pod_idx)
        else:
            endpoint.loc.locType = 1  # ENDPOINT_LOC_TYPE_HOST
            endpoint.loc.words[0] = int(self.host_id)
        return endpoint


class _CommAddr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("raws", ctypes.c_uint8 * 36),
    ]


class _EndpointLoc(ctypes.Structure):
    _fields_ = [
        ("locType", ctypes.c_int),
        ("words", ctypes.c_uint32 * 15),
    ]


class _EndpointDesc(ctypes.Structure):
    _fields_ = [
        ("protocol", ctypes.c_int),
        ("commAddr", _CommAddr),
        ("loc", _EndpointLoc),
        ("raws", ctypes.c_uint8 * 52),
    ]


class _HcommMem(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("addr", ctypes.c_void_p),
        ("size", ctypes.c_uint64),
    ]


class _CommAbiHeader(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("magicWord", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _HcommRoceAttr(ctypes.Structure):
    _fields_ = [
        ("queueNum", ctypes.c_uint32),
        ("retryCnt", ctypes.c_uint32),
        ("retryInterval", ctypes.c_uint32),
        ("tc", ctypes.c_uint8),
        ("sl", ctypes.c_uint8),
    ]


class _HcommChannelAttr(ctypes.Union):
    _fields_ = [
        ("raws", ctypes.c_uint8 * 128),
        ("roceAttr", _HcommRoceAttr),
    ]


class _HcommChannelDesc(ctypes.Structure):
    _fields_ = [
        ("header", _CommAbiHeader),
        ("remoteEndpoint", _EndpointDesc),
        ("notifyNum", ctypes.c_uint32),
        ("exchangeAllMems", ctypes.c_bool),
        ("memHandles", ctypes.POINTER(ctypes.c_void_p)),
        ("memHandleNum", ctypes.c_uint32),
        ("socket", ctypes.c_void_p),
        ("role", ctypes.c_int),
        ("port", ctypes.c_uint16),
        ("attr", _HcommChannelAttr),
    ]


class _RxeServerDesc(ctypes.Structure):
    _fields_ = [
        ("ip", ctypes.c_char * 64),
        ("port", ctypes.c_uint16),
        ("rkey", ctypes.c_uint32),
        ("addr", ctypes.c_uint64),
        ("size", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class _DecodedRxeDesc:
    ip: str
    port: int
    rkey: int
    addr: int
    size: int
    device: str
    gid_index: int


_RXE_DESC_MAGIC = b"SRXE"
_RXE_DESC_VERSION = 2
_RXE_DESC_STRUCT = struct.Struct("<4sHHHHIQQ64s64s")


def build_tensor_transport(name: str) -> TensorTransportBackend:
    selected = (name or "grpc").lower()
    if selected == "grpc":
        return GrpcTensorTransport()
    if selected == "rxe":
        backend = RxeTensorTransport.from_env()
        if not backend.available:
            raise TransportUnavailable(backend.unavailable_reason())
        return backend
    if selected == "hcomm":
        backend = HcommTensorTransport.from_env()
        if not backend.available:
            raise TransportUnavailable(backend.unavailable_reason())
        return backend
    if selected == "auto":
        if os.getenv("SIMPLER_RXE_AUTO", "").lower() in {"1", "true", "yes", "on"}:
            rxe_backend = RxeTensorTransport.from_env()
            if rxe_backend.available:
                return rxe_backend
        backend = HcommTensorTransport.from_env()
        return backend if backend.available else GrpcTensorTransport()
    raise ValueError(f"unknown tensor transport backend {name!r}")


def _check_hcomm(ret: int, op: str) -> None:
    if int(ret) != 0:
        raise TransportBackendError(f"{op} failed with HcclResult={ret}")


def _check_rxe(ret: int, op: str) -> None:
    if int(ret) != 0:
        raise TransportBackendError(f"{op} failed with errno-style rc={int(ret)}")


def _encode_rxe_desc(desc: "_RxeServerDesc", device: str, gid_index: int) -> bytes:
    ip = bytes(desc.ip).split(b"\0", 1)[0]
    device_bytes = str(device).encode("ascii")
    if len(ip) >= 64:
        raise TransportBackendError(f"RXE descriptor IP is too long: {ip!r}")
    if len(device_bytes) >= 64:
        raise TransportBackendError(f"RXE descriptor device name is too long: {device!r}")
    return _RXE_DESC_STRUCT.pack(
        _RXE_DESC_MAGIC,
        _RXE_DESC_VERSION,
        _RXE_DESC_STRUCT.size,
        int(desc.port),
        int(gid_index),
        int(desc.rkey),
        int(desc.addr),
        int(desc.size),
        ip.ljust(64, b"\0"),
        device_bytes.ljust(64, b"\0"),
    )


def _decode_rxe_desc(desc: bytes) -> "_DecodedRxeDesc":
    if not desc:
        raise TransportBackendError("RXE TensorHandle.transport_desc is empty")
    if desc.startswith(_RXE_DESC_MAGIC):
        if len(desc) < _RXE_DESC_STRUCT.size:
            raise TransportBackendError(
                f"RXE binary transport_desc is too short: {len(desc)} < {_RXE_DESC_STRUCT.size}"
            )
        magic, version, header_size, port, gid_index, rkey, addr, size, ip_raw, device_raw = _RXE_DESC_STRUCT.unpack(
            desc[: _RXE_DESC_STRUCT.size]
        )
        if magic != _RXE_DESC_MAGIC or version != _RXE_DESC_VERSION or header_size != _RXE_DESC_STRUCT.size:
            raise TransportBackendError(
                f"unsupported RXE transport_desc header: magic={magic!r}, version={version}, size={header_size}"
            )
        return _DecodedRxeDesc(
            ip=ip_raw.split(b"\0", 1)[0].decode("ascii"),
            port=int(port),
            rkey=int(rkey),
            addr=int(addr),
            size=int(size),
            device=device_raw.split(b"\0", 1)[0].decode("ascii"),
            gid_index=int(gid_index),
        )
    try:
        payload = json.loads(desc.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise TransportBackendError(f"invalid RXE transport_desc: {e}") from e
    if payload.get("transport") != "rxe":
        raise TransportBackendError(f"RXE transport_desc has unexpected transport {payload.get('transport')!r}")
    return _DecodedRxeDesc(
        ip=str(payload["ip"]),
        port=int(payload["port"]),
        rkey=int(payload.get("rkey", 0)),
        addr=int(payload.get("addr", 0)),
        size=int(payload["size"]),
        device=str(payload.get("device") or os.getenv("SIMPLER_RXE_DEVICE") or "rxe0"),
        gid_index=int(payload.get("gid_index", os.getenv("SIMPLER_RXE_GID_INDEX", "0"))),
    )


def _build_rxe_verbs_helper() -> Path:
    src = Path(__file__).with_name("rxe_verbs_helper.c")
    if not src.exists():
        raise TransportUnavailable(f"RXE verbs helper source not found: {src}")

    build_dir = _repo_root() / ".cache" / "rxe_verbs_helper"
    build_dir.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = build_dir / f"libsimpler_rxe_verbs_helper{suffix}"
    stamp = build_dir / "rxe_verbs_helper.stamp"
    include_dir = _rxe_include_dir()
    lib_dir = _rxe_lib_dir()
    signature = (
        f"{src.resolve()}:{src.stat().st_mtime_ns}:{src.stat().st_size}\n"
        f"include={include_dir}\nlib={lib_dir}\n"
    )
    if out.exists() and stamp.exists() and stamp.read_text() == signature:
        return out
    if include_dir is None or lib_dir is None:
        raise TransportUnavailable("RXE helper needs rdma-core headers/libs; set SIMPLER_RXE_INCLUDE_DIR and SIMPLER_RXE_LIB_DIR")

    compiler = os.getenv("CC") or "cc"
    cmd = [
        compiler,
        "-shared",
        "-fPIC",
        "-O2",
        f"-I{include_dir}",
        str(src),
        f"-L{lib_dir}",
        f"-Wl,-rpath,{lib_dir}",
        "-libverbs",
        "-lpthread",
        "-o",
        str(out),
    ]
    try:
        subprocess.check_call(cmd)
    except (OSError, subprocess.CalledProcessError) as e:
        raise TransportUnavailable(f"failed to build RXE helper with {' '.join(cmd)}: {e}") from e
    stamp.write_text(signature)
    return out


def _rxe_include_dir() -> Optional[Path]:
    for value in (
        os.getenv("SIMPLER_RXE_INCLUDE_DIR"),
        "/home/ntlab/rdma-build/rdma-core-50.0/build/include",
        "/home/ntlab/local/include",
        "/usr/include",
    ):
        if not value:
            continue
        path = Path(value).expanduser()
        if (path / "infiniband" / "verbs.h").exists():
            return path.resolve()
    return None


def _rxe_lib_dir() -> Optional[Path]:
    for value in (
        os.getenv("SIMPLER_RXE_LIB_DIR"),
        "/home/ntlab/rdma-build/rdma-core-50.0/build/lib",
        "/home/ntlab/local/lib64",
        "/home/ntlab/local/lib",
        "/usr/lib64",
        "/usr/lib",
    ):
        if not value:
            continue
        path = Path(value).expanduser()
        if (path / "libibverbs.so").exists() or (path / "libibverbs.so.1").exists():
            return path.resolve()
    return None


def _first_existing_rxe_device() -> Optional[str]:
    infiniband = Path("/sys/class/infiniband")
    if not infiniband.exists():
        return None
    for path in sorted(infiniband.iterdir()):
        if path.name.startswith("rxe"):
            return path.name
    return None


def _find_rxe_ipv4_gid(device: str) -> Optional[tuple[int, str]]:
    gid_dir = Path("/sys/class/infiniband") / device / "ports" / "1" / "gids"
    if not gid_dir.exists():
        return None
    for path in sorted(gid_dir.iterdir(), key=lambda item: int(item.name) if item.name.isdigit() else item.name):
        try:
            text = path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        ip = _ipv4_from_gid(text)
        if ip:
            return int(path.name), ip
    return None


def _ipv4_from_gid(gid: str) -> Optional[str]:
    parts = gid.strip().split(":")
    if len(parts) != 8 or parts[5].lower() != "ffff":
        return None
    try:
        hi = int(parts[6], 16)
        lo = int(parts[7], 16)
    except ValueError:
        return None
    return ".".join(str(octet) for octet in (hi >> 8, hi & 0xFF, lo >> 8, lo & 0xFF))


def _find_hcomm_library() -> Optional[str]:
    for name in ("hcomm", "hccl", "ascendcl"):
        found = ctypes.util.find_library(name)
        if found:
            return found
    return None


def _hcomm_dlopen_mode() -> int:
    mode = getattr(os, "RTLD_LOCAL", 0)
    # Local HCOMM builds can contain unresolved C++ symbols in paths unrelated
    # to the public C data-plane API used here.  Lazy binding keeps those paths
    # from blocking endpoint/memory smoke tests.
    mode |= getattr(os, "RTLD_LAZY", 0)
    return mode


def _preload_hcomm_dependencies(lib_path: str) -> None:
    """Load build-tree sidecar libs before libhcomm.so is dlopened.

    The HCOMM build emits `libhcomm.so`, `libhccl_alg.so`, `libhccl_plf.so`,
    and `libhccl_v2.so` into sibling directories without an rpath between
    them.  Preloading by absolute path keeps the Python smoke tests usable
    against a fresh local build without asking callers to hand-maintain a long
    `LD_LIBRARY_PATH`.
    """

    mode = getattr(os, "RTLD_GLOBAL", 0)
    mode |= getattr(os, "RTLD_NOW", 0)
    search_dirs = _hcomm_dependency_dirs(lib_path)
    root = Path(lib_path).expanduser().resolve()
    loaded: set[Path] = set()
    _preload_needed(root, search_dirs, loaded, mode)


def _preload_hcomm_abi_shim(lib_path: str) -> None:
    """Preload Simpler-owned ABI compatibility symbols for local HCOMM builds."""

    if os.getenv("SIMPLER_HCOMM_ABI_SHIM", "1").lower() in {"0", "false", "no", "off"}:
        return
    missing = _hcomm_missing_abi_symbols(lib_path)
    if not missing:
        return
    shim = _build_hcomm_abi_shim()
    mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    ctypes.CDLL(str(shim), mode=mode)


_HCOMM_ABI_SHIM_SYMBOLS = {
    "_ZN4hccl16HcclCommunicator15GenIbvAiRMAInfoI13HcclAiRMAInfoEE10HcclResultjRKSt10shared_ptrINS_9TransportEERKSsPT_",
}


def _hcomm_missing_abi_symbols(lib_path: str) -> set[str]:
    try:
        output = subprocess.check_output(
            ["nm", "-D", "--undefined-only", str(Path(lib_path).expanduser())],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    missing: set[str] = set()
    for line in output.splitlines():
        for symbol in _HCOMM_ABI_SHIM_SYMBOLS:
            if symbol in line:
                missing.add(symbol)
    return missing


def _build_hcomm_abi_shim() -> Path:
    src = Path(__file__).with_name("hcomm_abi_shim.cc")
    if not src.exists():
        raise TransportUnavailable(f"HCOMM ABI shim source not found: {src}")

    build_dir = _repo_root() / ".cache" / "hcomm_abi_shim"
    build_dir.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = build_dir / f"libsimple_hcomm_abi_shim{suffix}"
    stamp = build_dir / "hcomm_abi_shim.stamp"
    signature = f"{src.resolve()}:{src.stat().st_mtime_ns}:{src.stat().st_size}\n"
    if out.exists() and stamp.exists() and stamp.read_text() == signature:
        return out

    compiler = os.getenv("CXX") or "c++"
    hcomm_root = _hcomm_source_root()
    cmd = [
        compiler,
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c++14",
        "-DLOG_CPP",
        "-DOPEN_BUILD_PROJECT",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        *[f"-I{path}" for path in _hcomm_abi_shim_include_dirs(hcomm_root)],
        str(src),
        "-o",
        str(out),
    ]
    try:
        subprocess.check_call(cmd)
    except (OSError, subprocess.CalledProcessError) as e:
        raise TransportUnavailable(f"failed to build HCOMM ABI shim with {' '.join(cmd)}: {e}") from e
    stamp.write_text(signature)
    return out


def _hcomm_source_root() -> Optional[Path]:
    explicit = os.getenv("SIMPLER_HCOMM_SRC_DIR")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if root.exists():
            return root
    root = _repo_root().parent / "3rd" / "hcomm"
    if root.exists():
        return root.resolve()
    return None


def _hcomm_abi_shim_include_dirs(hcomm_root: Optional[Path]) -> list[Path]:
    if hcomm_root is None:
        raise TransportUnavailable("HCOMM ABI shim needs HCOMM source headers; set SIMPLER_HCOMM_SRC_DIR")
    flags_make = hcomm_root / "build" / "src" / "framework" / "CMakeFiles" / "hcomm.dir" / "flags.make"
    if flags_make.exists():
        includes = _parse_make_include_dirs(flags_make)
        if includes:
            return includes
    src = hcomm_root / "src"
    return [
        hcomm_root / "include",
        hcomm_root / "include" / "hccl",
        hcomm_root / "pkg_inc",
        hcomm_root / "pkg_inc" / "hccl",
        src / "framework" / "communicator" / "impl",
        src / "framework" / "common" / "src",
        src / "framework" / "common" / "src" / "h2d_dto",
        src / "framework" / "inc",
        src / "pub_inc",
        src / "pub_inc" / "inner",
        src / "pub_inc" / "new",
        src / "algorithm" / "pub_inc",
        src / "algorithm" / "base" / "inc",
    ]


def _parse_make_include_dirs(flags_make: Path) -> list[Path]:
    lines = flags_make.read_text().splitlines()
    value = ""
    collecting = False
    for line in lines:
        if line.startswith("CXX_INCLUDES = "):
            value = line.split("=", 1)[1].strip()
            collecting = line.endswith("\\")
        elif collecting:
            value += " " + line.strip()
            collecting = line.endswith("\\")
        elif value:
            break
    includes: list[Path] = []
    seen: set[Path] = set()
    for token in shlex.split(value.replace("\\\n", " ")):
        if not token.startswith("-I"):
            continue
        path = Path(token[2:]).expanduser()
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        includes.append(resolved)
    return includes


def _repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return path.parents[3]


_SYSTEM_LIB_PREFIXES = ("libc.so", "libdl.so", "libm.so", "libpthread.so", "librt.so", "ld-linux-", "libgcc_s.so")


def _preload_needed(path: Path, search_dirs: list[Path], loaded: set[Path], mode: int) -> None:
    for lib_name in _elf_needed(path):
        if _is_system_library(lib_name):
            continue
        dep_path = _find_dependency_library(lib_name, search_dirs)
        if dep_path is None or dep_path in loaded:
            continue
        _preload_needed(dep_path, search_dirs, loaded, mode)
        ctypes.CDLL(str(dep_path), mode=mode)
        loaded.add(dep_path)


def _elf_needed(path: Path) -> list[str]:
    try:
        output = subprocess.check_output(["readelf", "-d", str(path)], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    needed: list[str] = []
    for line in output.splitlines():
        if "(NEEDED)" not in line:
            continue
        start = line.find("[")
        end = line.find("]", start + 1)
        if start >= 0 and end > start:
            needed.append(line[start + 1 : end])
    return needed


def _is_system_library(lib_name: str) -> bool:
    return lib_name.startswith(_SYSTEM_LIB_PREFIXES) or lib_name in {"libstdc++.so.6"}


def _find_dependency_library(lib_name: str, search_dirs: list[Path]) -> Optional[Path]:
    for directory in _ordered_dependency_dirs(lib_name, search_dirs):
        path = directory / lib_name
        if path.exists():
            return path.resolve()
    found = ctypes.util.find_library(lib_name.removeprefix("lib").removesuffix(".so"))
    if found:
        path = Path(found)
        if path.exists():
            return path.resolve()
    return None


_HCOMM_BUILD_LIBS = {
    "libhcomm.so",
    "libhccl_alg.so",
    "libhccl_plf.so",
    "libhccl_v2.so",
    "libhccl_legacy.so",
    "libccl_dpu.so",
    "libra.so",
    "libra_hdc.so",
    "libra_peer.so",
    "librs.so",
    "libtls_adp.so",
    "libtopoaddrinfo.so",
}


def _ordered_dependency_dirs(lib_name: str, search_dirs: list[Path]) -> list[Path]:
    if lib_name not in _HCOMM_BUILD_LIBS:
        return search_dirs
    build_dirs = [directory for directory in search_dirs if "/3rd/hcomm/build/" in str(directory)]
    other_dirs = [directory for directory in search_dirs if directory not in build_dirs]
    return build_dirs + other_dirs


def _hcomm_dependency_dirs(hcomm_lib_path: str) -> list[Path]:
    explicit = os.getenv("SIMPLER_HCOMM_DEP_LIB_DIRS", "")
    dirs = [Path(item).expanduser() for item in explicit.split(os.pathsep) if item]

    for env_name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "CANN_HOME", "ASCEND_HOME"):
        root = os.getenv(env_name)
        if root:
            dirs.extend(_cann_dependency_dirs(Path(root).expanduser()))
    for root in (Path("/home/ntlab/zcy/cann/cann-9.0.0"), Path("/usr/local/Ascend/ascend-toolkit/latest")):
        dirs.extend(_cann_dependency_dirs(root))

    lib = Path(hcomm_lib_path).expanduser()
    if lib.parent:
        dirs.extend([lib.parent, lib.parent / "legacy"])
        if lib.parent.name == "framework":
            src = lib.parent.parent
            build = src.parent
            dirs.extend(
                [
                    src / "algorithm",
                    src / "platform",
                    src / "platform" / "hccp" / "rdma_agent" / "hdc",
                    src / "platform" / "hccp" / "rdma_agent" / "peer",
                    src / "framework" / "legacy",
                    build / "stub",
                    build / "_CPack_Packages" / "makeself_staging" / "aarch64-linux" / "lib64",
                ]
            )

    result: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        resolved = directory.resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def _cann_dependency_dirs(root: Path) -> list[Path]:
    return [
        root / "aarch64-linux" / "lib64",
        root / "aarch64-linux" / "lib64" / "device" / "lib64",
        root / "aarch64-linux" / "devlib",
        root / "aarch64-linux" / "devlib" / "device",
        root / "aarch64-linux" / "devlib" / "linux" / "aarch64",
        root / "lib64",
        root / "lib64" / "device" / "lib64",
        root / "devlib",
    ]


def _ensure_cann_runtime_env(acl_path: Path) -> None:
    root = _cann_root_from_acl_path(acl_path)
    if root is None:
        return
    os.environ.setdefault("ASCEND_HOME_PATH", str(root))
    os.environ.setdefault("ASCEND_TOOLKIT_HOME", str(root))
    os.environ.setdefault("ASCEND_OPP_PATH", str(root / "opp"))
    os.environ.setdefault("ASCEND_AICPU_PATH", str(root))
    os.environ.setdefault("TOOLCHAIN_HOME", str(root / "toolkit"))

    arch = os.uname().machine
    for directory in (
        root / "lib64",
        root / "lib64" / "plugin" / "opskernel",
        root / "lib64" / "plugin" / "nnengine",
        root / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe" / "op_tiling" / "lib" / "linux" / arch,
        Path("/usr/local/Ascend/driver/lib64"),
        Path("/usr/local/Ascend/driver/lib64/common"),
        Path("/usr/local/Ascend/driver/lib64/driver"),
        root / "devlib",
    ):
        _prepend_env_path("LD_LIBRARY_PATH", directory)


def _cann_root_from_acl_path(acl_path: Path) -> Optional[Path]:
    resolved = acl_path.expanduser().resolve()
    for parent in resolved.parents:
        if (parent / "opp").is_dir() and (parent / "lib64").is_dir():
            return parent
        if parent.name.endswith("-linux") and (parent.parent / "opp").is_dir():
            return parent.parent
    return None


def _prepend_env_path(name: str, directory: Path) -> None:
    if not directory.is_dir():
        return
    value = str(directory)
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    if value in parts:
        return
    os.environ[name] = os.pathsep.join([value, *parts])


def _mem_type_value(mem_type: str) -> int:
    return 0 if mem_type == "device" else 1


def _parse_handle(value) -> int:  # noqa: ANN001
    if isinstance(value, int):
        return value
    return int(str(value), 0)


def _hcomm_socket_role(value) -> int:  # noqa: ANN001
    if isinstance(value, int):
        return value
    role = str(value).strip().lower()
    if role in {"client", "c", "0"}:
        return 0
    if role in {"server", "s", "1"}:
        return 1
    if role in {"reserved", "auto", "-1"}:
        return -1
    raise ValueError(f"unknown HCOMM socket role {value!r}")


def _init_hcomm_channel_desc(desc: "_HcommChannelDesc") -> None:
    ctypes.memset(ctypes.byref(desc), 0xFF, ctypes.sizeof(desc))
    desc.header.version = 1
    desc.header.magicWord = 0x0FCF0F0F
    desc.header.size = ctypes.sizeof(_HcommChannelDesc)
    desc.header.reserved = 0
    desc.notifyNum = 0
    desc.exchangeAllMems = False
    desc.memHandles = None
    desc.memHandleNum = 0
    desc.socket = None
    desc.role = -1
    desc.port = 0
    endpoint = EndpointSpec(ip="0.0.0.0").to_ctypes()
    endpoint.protocol = -1
    endpoint.commAddr.type = -1
    endpoint.loc.locType = -1
    desc.remoteEndpoint = endpoint


def _endpoint_from_transport_desc(desc: bytes) -> "_EndpointDesc":
    size = ctypes.sizeof(_EndpointDesc)
    if len(desc) < size:
        raise TransportBackendError(
            f"HCOMM transport_desc is too short to contain EndpointDesc: {len(desc)} < {size}"
        )
    return _EndpointDesc.from_buffer_copy(desc[-size:])


def _buffer_addr(data: bytearray) -> int:
    if not data:
        return 0
    return ctypes.addressof(ctypes.c_char.from_buffer(data))
