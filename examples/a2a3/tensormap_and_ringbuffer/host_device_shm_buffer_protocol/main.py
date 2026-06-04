#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Multi-epoch shared-buffer protocol over one HostDeviceMappedRegion.

The mapped region persists across epochs, but each epoch submits one L2 device
task. This is not persistent-kernel streaming.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    MappedRegionInfo,
    TaskArgs,
)
from simpler.worker import MappedRegion, Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.runtime_builder import RuntimeBuilder

HERE = Path(__file__).resolve().parent
KERNEL_DIR = HERE / "kernels"
RUNTIME = "tensormap_and_ringbuffer"

SHM_MAGIC = 0x48534442
SHM_VERSION = 1
DEFAULT_EPOCHS = 16
DEFAULT_PAYLOAD_BYTES = 64 * 1024
HEADER = struct.Struct("<IIIIQQQQQQ")
HEADER_BYTES = 64
SUPPORTED_PLATFORMS = {"a2a3sim", "a2a3", "a5sim"}
TIMEOUT_US = 1_000_000
assert HEADER.size == HEADER_BYTES


@dataclass
class ShmBuffer:
    worker: Worker
    region: MappedRegion
    info: MappedRegionInfo
    payload_bytes: int
    worker_id: int = 0
    closed: bool = False

    @property
    def device_data_ptr(self) -> int:
        return int(self.info.device_data_ptr)

    @property
    def device_signal_ptr(self) -> int:
        return int(self.info.device_signal_ptr)

    def close(self) -> None:
        if self.closed:
            return
        self.worker.close_mapped_region(self.region, worker_id=self.worker_id)
        self.closed = True

    def __enter__(self) -> ShmBuffer:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


@dataclass(frozen=True)
class ShmResponse:
    magic: int
    version: int
    status: int
    seq: int
    input_bytes: int
    output_bytes: int
    checksum: int
    output: bytes


def input_offset(_payload_bytes: int) -> int:
    return HEADER_BYTES


def output_offset(payload_bytes: int) -> int:
    return HEADER_BYTES + payload_bytes


def region_bytes(payload_bytes: int) -> int:
    return HEADER_BYTES + payload_bytes * 2


def make_payload(seq: int, payload_bytes: int) -> bytes:
    return bytes(((seq * 29 + i * 3 + (i >> 3)) & 0xFF) for i in range(payload_bytes))


def transform_payload(seq: int, payload: bytes) -> bytes:
    return bytes((b ^ ((seq + i * 13) & 0xFF)) for i, b in enumerate(payload))


def checksum64(payload: bytes) -> int:
    acc = 0x9E3779B185EBCA87
    for i, value in enumerate(payload):
        acc ^= (value + ((i + 1) * 0x100000001B3)) & 0xFFFFFFFFFFFFFFFF
        acc = ((acc << 7) | (acc >> 57)) & 0xFFFFFFFFFFFFFFFF
        acc = (acc * 0xD6E8FEB86659FD93) & 0xFFFFFFFFFFFFFFFF
    return acc


def pack_header(seq: int, input_bytes: int, output_bytes: int, status: int, checksum: int = 0) -> bytes:
    return HEADER.pack(
        SHM_MAGIC,
        SHM_VERSION,
        status,
        0,
        seq,
        input_bytes,
        output_bytes,
        checksum,
        0,
        0,
    )


def unpack_header(raw: bytes) -> tuple[int, int, int, int, int, int, int, int]:
    magic, version, status, _reserved0, seq, input_bytes, output_bytes, checksum, _r0, _r1 = HEADER.unpack(raw)
    return magic, version, status, seq, input_bytes, output_bytes, checksum, _reserved0


def open_shm_buffer(worker: Worker, payload_bytes: int, worker_id: int = 0) -> ShmBuffer:
    region = worker.open_mapped_region(region_bytes(payload_bytes), signal_count=2, worker_id=worker_id)
    try:
        info = worker.mapped_region_info(region, worker_id=worker_id)
    except Exception:
        worker.close_mapped_region(region, worker_id=worker_id)
        raise
    return ShmBuffer(worker=worker, region=region, info=info, payload_bytes=payload_bytes, worker_id=worker_id)


def shm_buffer_send_cpu(buffer: ShmBuffer, seq: int, payload: bytes) -> None:
    if len(payload) > buffer.payload_bytes:
        raise ValueError(f"payload exceeds shm buffer capacity: {len(payload)} > {buffer.payload_bytes}")
    buffer.worker.mapped_region_datacopy_h2region(
        buffer.region,
        0,
        pack_header(seq, len(payload), buffer.payload_bytes, status=0),
        worker_id=buffer.worker_id,
    )
    buffer.worker.mapped_region_datacopy_h2region(
        buffer.region, input_offset(buffer.payload_bytes), payload, worker_id=buffer.worker_id
    )
    buffer.worker.mapped_region_notify(buffer.region, 0, seq, worker_id=buffer.worker_id)


def shm_buffer_recv_cpu(buffer: ShmBuffer, seq: int, timeout_us: int = TIMEOUT_US) -> ShmResponse:
    buffer.worker.mapped_region_wait(buffer.region, 1, seq, timeout_us, worker_id=buffer.worker_id)
    raw_header = buffer.worker.mapped_region_datacopy_region2h(
        buffer.region, 0, HEADER_BYTES, worker_id=buffer.worker_id
    )
    output = buffer.worker.mapped_region_datacopy_region2h(
        buffer.region, output_offset(buffer.payload_bytes), buffer.payload_bytes, worker_id=buffer.worker_id
    )
    magic, version, status, got_seq, input_bytes, got_output_bytes, got_checksum, _reserved0 = unpack_header(raw_header)
    return ShmResponse(
        magic=magic,
        version=version,
        status=status,
        seq=got_seq,
        input_bytes=input_bytes,
        output_bytes=got_output_bytes,
        checksum=got_checksum,
        output=output,
    )


def build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(RUNTIME)

    kernel_bytes = kc.compile_incore(
        source_path=str(KERNEL_DIR / "aiv" / "host_device_shm_buffer_protocol.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=RUNTIME,
        source_path=str(KERNEL_DIR / "orchestration" / "host_device_shm_buffer_protocol_orch.cpp"),
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        func_name="host_device_shm_buffer_protocol_orch",
        config_name="host_device_shm_buffer_protocol_config",
        binary=orch_bytes,
        children=[(0, CoreCallable.build(signature=[], binary=kernel_bytes))],
    )


def run(
    platform: str,
    device_id: int,
    *,
    build: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    payload_bytes: int = DEFAULT_PAYLOAD_BYTES,
) -> int:
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"unsupported platform: {platform}")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if payload_bytes <= 0:
        raise ValueError("payload_bytes must be positive")

    os.environ["PTO_ISA_ROOT"] = ensure_pto_isa_root(clone_protocol="https")
    RuntimeBuilder(platform=platform).get_binaries(RUNTIME, build=build)
    chip_callable = build_chip_callable(platform)

    worker = Worker(
        level=3,
        platform=platform,
        runtime=RUNTIME,
        device_ids=[device_id],
        num_sub_workers=0,
        build=build,
    )
    chip_cid = worker.register(chip_callable)
    try:
        worker.init()
        with open_shm_buffer(worker, payload_bytes) as buffer:
            cfg = CallConfig()
            cfg.block_dim = 1
            cfg.aicpu_thread_num = 2

            for epoch in range(epochs):
                seq = epoch + 1
                payload = make_payload(seq, payload_bytes)
                shm_buffer_send_cpu(buffer, seq, payload)

                def orch_fn(orch, _args, run_cfg):
                    args = TaskArgs()
                    args.add_scalar(buffer.device_data_ptr)
                    args.add_scalar(buffer.device_signal_ptr)
                    args.add_scalar(seq)
                    args.add_scalar(payload_bytes)
                    orch.submit_next_level(chip_cid, args, run_cfg, worker=0)

                worker.run(orch_fn, args=None, config=cfg)
                response = shm_buffer_recv_cpu(buffer, seq)

                expected = transform_payload(seq, payload)
                assert response.magic == SHM_MAGIC
                assert response.version == SHM_VERSION
                assert response.status == 0
                assert response.seq == seq
                assert response.input_bytes == payload_bytes
                assert response.output_bytes == payload_bytes
                assert response.checksum == checksum64(expected)
                assert response.output == expected
        return 0
    finally:
        worker.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", required=True, choices=sorted(SUPPORTED_PLATFORMS))
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--build", action="store_true", help="Rebuild runtime from source.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--payload-bytes", type=int, default=DEFAULT_PAYLOAD_BYTES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rc = run(args.platform, args.device, build=args.build, epochs=args.epochs, payload_bytes=args.payload_bytes)
    print(
        "[host_device_shm_buffer_protocol] "
        f"platform={args.platform} device={args.device} epochs={args.epochs} "
        f"payload_bytes={args.payload_bytes} PASSED"
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
