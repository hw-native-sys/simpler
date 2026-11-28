# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-side HIXL CS operations. Call from the thread bound to the allocation's ACL context."""

import ctypes as ct
import json
import os
from dataclasses import dataclass, field
from ipaddress import IPv4Address

from simpler.transfer_domain import TransferCopy, TransferDomainError, TransferRegion

_HANDLE = ct.c_void_p
_TEXT = ct.c_char_p
_I32 = ct.c_int32
_U64 = ct.c_uint64
_HANDLE_OUT = ct.POINTER(_HANDLE)
_U64_OUT = ct.POINTER(_U64)
_SIGNATURES = {
    "server_create": (_TEXT, _I32, _TEXT, _I32, _HANDLE_OUT),
    "client_create": (_TEXT, _I32, _TEXT, _I32, _TEXT, _I32, _HANDLE_OUT),
    "server_register": (_HANDLE, _TEXT, _U64, _U64, _HANDLE_OUT),
    "client_register": (_HANDLE, _TEXT, _U64, _U64, _HANDLE_OUT),
    "listen": (_HANDLE,),
    "connect": (_HANDLE,),
    "remote_region": (_HANDLE, _TEXT, _U64_OUT, _U64_OUT),
    "write": (_HANDLE, _U64_OUT, _U64_OUT, _U64_OUT, ct.c_uint32),
    "server_unregister": (_HANDLE, _HANDLE),
    "client_unregister": (_HANDLE, _HANDLE),
    "server_destroy": (_HANDLE,),
    "client_destroy": (_HANDLE,),
}


class HixlError(RuntimeError):
    """A native operation failed. Its caller must keep any possibly active buffers alive."""


@dataclass(frozen=True)
class DeviceEndpoint:
    ip: str
    physical_device: int

    def __post_init__(self):
        if IPv4Address(self.ip).is_unspecified or not 0 <= self.physical_device <= 0x7FFF_FFFF:
            raise ValueError("invalid HIXL device endpoint")


class HixlDeviceLink:
    """A client or listener that owns registrations, while the caller owns the memory."""

    @staticmethod
    def library(path=None):
        path = path or os.environ.get("PYPTO_HIXL_BRIDGE")
        if not path:
            raise ValueError("PYPTO_HIXL_BRIDGE must point to the HIXL CS bridge shared library")
        lib = ct.CDLL(path)
        for operation, signature in _SIGNATURES.items():
            fn = getattr(lib, f"hixl_pd_{operation}")
            fn.argtypes = signature
            fn.restype = _I32
        return lib

    def __init__(self, local, host, port, *, peer=None, library=None):
        if IPv4Address(host).is_unspecified or not 0 < port <= 65535:
            raise ValueError("invalid HIXL server address")
        self.library_ref = self.library(library)
        self.handle = _HANDLE()
        self.registrations = {}
        self.side = "server" if peer is None else "client"
        args = (local.ip.encode("ascii"), local.physical_device)
        if peer is not None:
            args += (peer.ip.encode("ascii"), peer.physical_device)
        self._call(f"{self.side}_create", *args, host.encode("ascii"), port, ct.byref(self.handle))

    def _call(self, operation, *args):
        result = getattr(self.library_ref, f"hixl_pd_{operation}")(*args)
        if result:
            raise HixlError(f"HIXL CS {operation} failed with status {result}")

    def register(self, tag, address, extent):
        if tag in self.registrations:
            raise ValueError("duplicate HIXL region tag")
        mr = _HANDLE()
        self._call(f"{self.side}_register", self.handle, tag.encode("ascii"), address, extent, ct.byref(mr))
        self.registrations[tag] = mr

    def listen(self):
        self._call("listen", self.handle)

    def connect(self):
        self._call("connect", self.handle)

    def remote_region(self, tag):
        address, extent = _U64(), _U64()
        self._call("remote_region", self.handle, tag.encode("ascii"), ct.byref(address), ct.byref(extent))
        return address.value, extent.value

    def batch_put(self, segments):
        """Return only when the native synchronous write succeeds for the whole batch."""
        if segments:
            source, destination, lengths = zip(*segments)
            array = _U64 * len(segments)
            self._call("write", self.handle, array(*source), array(*destination), array(*lengths), len(segments))

    def close(self):
        for tag, mr in list(self.registrations.items()):
            self._call(f"{self.side}_unregister", self.handle, mr)
            del self.registrations[tag]
        if self.handle:
            self._call(f"{self.side}_destroy", self.handle)
            self.handle = _HANDLE()


def _valid_range(address, extent):
    return type(address) is int and type(extent) is int and address > 0 and extent > 0 and address + extent < 1 << 64


@dataclass(frozen=True)
class _RegisteredRegion:
    address: int
    extent: int


@dataclass
class _DecodePeer:
    regions: dict
    endpoints: dict
    clients: dict = field(default_factory=dict)


class HixlTransferOwner:
    @staticmethod
    def endpoints():
        raw = os.environ.get("PYPTO_HIXL_DEVICE_ENDPOINTS")
        if raw is None:
            return (DeviceEndpoint(os.environ["PYPTO_HIXL_DEVICE_IP"], int(os.environ["PYPTO_HIXL_PHYSICAL_DEVICE"])),)
        values = json.loads(raw)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, list)
                or len(value) != 2
                or not isinstance(value[0], str)
                or type(value[1]) is not int
                for value in values
            )
        ):
            raise ValueError("HIXL endpoints must be a nonempty list of [RoCE IPv4, physical device] pairs")
        devices = tuple(DeviceEndpoint(ip, physical) for ip, physical in values)
        if len({d.ip for d in devices}) != len(devices) or len({d.physical_device for d in devices}) != len(devices):
            raise ValueError("HIXL device IPs and physical IDs must be unique")
        return devices

    @classmethod
    def check_available(cls):
        HixlDeviceLink.library()
        cls.endpoints()

    def __init__(self, hostname, role, rank=0):
        if role not in ("prefill", "decode"):
            raise ValueError("HIXL requires a Prefill or Decode allocation owner")
        self.role, self.rank = role, rank
        self.device = self.endpoints()[rank]
        self.local, self.peers = {}, {}
        self.link = None
        self.listening = False
        self.poisoned = self.closed = False
        if role == "decode":
            self.port = int(os.environ["PYPTO_HIXL_PORT"]) + rank
            self.host = str(IPv4Address(hostname))
            if not 0 < self.port <= 65535:
                raise ValueError("invalid HIXL listener port")

    def _native(self, operation, *args, **kwargs):
        if self.poisoned or self.closed:
            raise TransferDomainError("HIXL owner is gated or closed", uncertain=self.poisoned)
        try:
            return operation(*args, **kwargs)
        except HixlError as exc:
            self.poisoned = True
            raise TransferDomainError(str(exc), uncertain=True) from exc

    def register(self, boot, rank, name, address, extent):
        if (
            self.closed
            or self.listening
            or self.peers
            or rank != self.rank
            or name in self.local
            or not name
            or not name.isascii()
            or "\0" in name
            or not boot
            or not _valid_range(address, extent)
        ):
            raise ValueError("invalid, duplicate or late HIXL region registration")
        handle = dict(address=address)
        if self.role == "decode":
            handle.update(
                device_ip=self.device.ip, physical_device=self.device.physical_device, host=self.host, port=self.port
            )
        region = TransferRegion(boot, rank, name, extent, handle)
        self.local[name] = _RegisteredRegion(address, extent)
        return region

    def prepare(self):
        if self.role != "decode" or not self.local:
            raise ValueError("HIXL listener requires registered Decode regions")
        if not self.listening:
            self.link = self._native(HixlDeviceLink, self.device, self.host, self.port)
            for name, region in self.local.items():
                self._native(self.link.register, name, region.address, region.extent)
            self._native(self.link.listen)
            self.listening = True

    def install(self, regions, *, peer):
        if self.role != "prefill" or self.closed or peer in self.peers:
            raise ValueError("HIXL expects a new Decode registry")
        grouped = {}
        if len(peer) != 3 or not all(isinstance(value, str) and value for value in peer):
            raise ValueError("invalid Decode peer identity")
        boot = peer[1]
        for region in regions:
            grouped.setdefault(region.rank, []).append(region)
        if not grouped:
            raise ValueError("HIXL expects Decode regions")
        keys = ("device_ip", "physical_device", "host", "port")
        endpoints = {}
        registered = {}
        for rank, rank_regions in grouped.items():
            if len(rank_regions) != len(self.local) or {r.name for r in rank_regions} != self.local.keys():
                raise ValueError("HIXL expects matching Decode regions per rank")
            endpoint = tuple(rank_regions[0].handle.get(k) for k in keys)
            device_ip, physical_device, host, port = endpoint
            if type(physical_device) is not int:
                raise ValueError("invalid HIXL destination physical device")
            endpoints[rank] = DeviceEndpoint(device_ip, physical_device), str(IPv4Address(host)), port
            if type(port) is not int or not 0 < port <= 65535:
                raise ValueError("invalid HIXL destination port")
            for region in rank_regions:
                address = region.handle.get("address")
                if (
                    region.boot != boot
                    or tuple(region.handle.get(k) for k in keys) != endpoint
                    or not _valid_range(address, region.extent)
                ):
                    raise ValueError("invalid HIXL destination region")
                registered[rank, region.name] = _RegisteredRegion(address, region.extent)
        self.peers[peer] = _DecodePeer(registered, endpoints)

    def connect(self, peer, ranks):
        peer = self.peers.get(peer)
        if self.role != "prefill" or self.closed or peer is None or not set(ranks) <= peer.endpoints.keys():
            raise ValueError("HIXL expects installed Decode destinations")
        for rank in ranks:
            if rank in peer.clients:
                continue
            peer_device, host, port = peer.endpoints[rank]
            client = self._native(HixlDeviceLink, self.device, host, port, peer=peer_device)
            peer.clients[rank] = client
            for name, region in self.local.items():
                self._native(client.register, name, region.address, region.extent)
            self._native(client.connect)
            for name in self.local:
                region = peer.regions[rank, name]
                exported = self._native(client.remote_region, name)
                if exported != (region.address, region.extent):
                    self.poisoned = True
                    raise TransferDomainError("HIXL export differs from the authorized Decode registry", uncertain=True)

    def put_sync_host(self, peer, copies):
        if not copies:
            return
        rank = copies[0].destination_rank
        peer = self.peers.get(peer)
        if peer is None or rank not in peer.clients:
            raise ValueError("HIXL destination has no installed link")
        segments = []
        for copy in copies:
            if not isinstance(copy, TransferCopy):
                raise TypeError("expected a generic transfer copy")
            src = self.local[copy.source_region]
            dst = peer.regions[rank, copy.destination_region]
            if (
                copy.source_rank != self.rank
                or copy.destination_rank != rank
                or min(copy.source_offset, copy.destination_offset) < 0
                or copy.length <= 0
                or copy.source_offset + copy.length > src.extent
                or copy.destination_offset + copy.length > dst.extent
            ):
                raise ValueError("HIXL copy outside its registered region")
            segments.append(
                (
                    src.address + copy.source_offset,
                    dst.address + copy.destination_offset,
                    copy.length,
                )
            )
        self._native(peer.clients[rank].batch_put, segments)

    def close_peer(self, peer):
        connection = self.peers.get(peer)
        if connection is not None:
            for client in connection.clients.values():
                self._native(client.close)
            del self.peers[peer]

    def close(self):
        if self.closed:
            return
        if self.link is not None:
            self._native(self.link.close)
        for identity in tuple(self.peers):
            self.close_peer(identity)
        self.closed = True
        self.local.clear()
        self.peers.clear()
