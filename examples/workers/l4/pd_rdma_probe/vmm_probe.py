# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify RDMA between device VMM mappings imported by independent transfer processes."""

import argparse
import ctypes as ct
import multiprocessing
import os
import sys

from simpler.buffer_export import BufferMapping
from simpler.l4_pd_rdma import DeviceEndpoint, HixlDeviceLink
from simpler.worker import Worker

_EXTENT = 64 * 1024
_COPIES = ((256, 512, 64), (1024, 8192, 4096))
_TAG = "pd-probe-vmm-hbm"
_GUARD = 0xCA


def _source_pattern():
    return bytes((index * 73 + (index >> 5) * 29) % 251 for index in range(_EXTENT))


def _allocation_owner(channel, device_id, role):
    with channel:
        worker = Worker(level=2, platform="a2a3", runtime="tensormap_and_ringbuffer", device_id=device_id)
        worker.init()
        buffer = worker.malloc(_EXTENT, exportable=True)
        source = _source_pattern()
        guards = bytes([_GUARD]) * _EXTENT
        worker.copy_to(buffer, bytearray(source if role == "prefill" else guards))
        exported = worker.export_buffer(buffer)
        print(f"VMM_EXPORTED owner_pid={os.getpid()} bytes={_EXTENT}", flush=True)
        channel.send(exported)
        # The consumer sends this only after unregister and unmap both succeed.
        command = channel.recv()
        if command not in ("verify", "release"):
            raise ValueError("VMM owner did not receive a retirement confirmation")
        actual = expected = None
        if command == "verify":
            expected = bytearray(guards)
            for source_offset, destination_offset, length in _COPIES:
                expected[destination_offset : destination_offset + length] = source[
                    source_offset : source_offset + length
                ]
            actual = bytearray(_EXTENT)
            worker.copy_from(actual, buffer)
        worker.release_export(exported)
        worker.free(buffer)
        worker.close()
        print("VMM_OWNER_RELEASED", flush=True)
        if actual != expected:
            raise RuntimeError("the allocation owner read back incorrect data or modified guard bytes")
        if command == "verify":
            print(
                f"VMM_VERIFIED owner_read_bytes={_EXTENT} written_bytes={sum(copy[2] for copy in _COPIES)}", flush=True
            )


def _acl_check(acl, operation, *args):
    status = getattr(acl, operation)(*args)
    if status:
        raise RuntimeError(f"{operation} failed with status {status}")


def _transfer(args, exported):
    acl = ct.CDLL("libascendcl.so")
    acl.aclInit.argtypes = [ct.c_char_p]
    acl.aclrtSetDevice.argtypes = [ct.c_int32]
    _acl_check(acl, "aclInit", None)
    _acl_check(acl, "aclrtSetDevice", args.device)
    mapping = BufferMapping(exported)
    local = DeviceEndpoint(args.roce_ip, args.physical)
    peer = DeviceEndpoint(args.peer_roce_ip, args.peer_physical) if args.role == "prefill" else None
    print(f"VMM_IMPORTED transfer_pid={os.getpid()} bytes={mapping.descriptor.nbytes}", flush=True)
    link = HixlDeviceLink(local, args.destination_host, args.port, peer=peer)
    link.register(_TAG, mapping.base, _EXTENT)
    print(f"VMM_REGISTERED role={args.role}", flush=True)
    command = "release"
    if peer is None:
        link.listen()
        print("DECODE_READY transport=device_roce", flush=True)
        signal = sys.stdin.readline().strip()
        if signal not in ("VERIFY", "NO_SUBMISSION"):
            raise RuntimeError("Prefill retirement was not confirmed")
        if signal == "VERIFY":
            command = "verify"
    else:
        link.connect()
        remote_address, remote_extent = link.remote_region(_TAG)
        if remote_extent != _EXTENT:
            raise ValueError("Decode exported a different VMM extent")
        link.batch_put([(mapping.base + src, remote_address + dst, size) for src, dst, size in _COPIES])
        print(f"VMM_WRITE_COMPLETE bytes={sum(copy[2] for copy in _COPIES)}", flush=True)
    link.close()
    print("VMM_UNREGISTERED", flush=True)
    mapping.close()
    print("VMM_UNMAPPED", flush=True)
    _acl_check(acl, "aclFinalize")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("prefill", "decode"))
    parser.add_argument("--device", required=True, type=int, help="ACL logical device ID")
    parser.add_argument("--physical", required=True, type=int, help="local physical device ID")
    parser.add_argument("--roce-ip", required=True)
    parser.add_argument("--destination-host", required=True, help="Decode management IPv4 address")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--peer-physical", type=int)
    parser.add_argument("--peer-roce-ip")
    args = parser.parse_args()
    if args.role == "prefill" and (args.peer_physical is None or args.peer_roce_ip is None):
        parser.error("Prefill requires --peer-physical and --peer-roce-ip")

    context = multiprocessing.get_context("spawn")
    controller, child = context.Pipe()
    owner = context.Process(target=_allocation_owner, args=(child, args.device, args.role))
    with controller, child:
        owner.start()
        child.close()
        exported = controller.recv()
        controller.send(_transfer(args, exported))
    owner.join()
    if owner.exitcode:
        raise RuntimeError(f"VMM allocation owner exited with status {owner.exitcode}")


if __name__ == "__main__":
    main()
