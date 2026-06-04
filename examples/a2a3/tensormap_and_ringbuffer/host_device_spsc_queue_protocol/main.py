#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Multi-epoch SPSC queue protocol over one HostDeviceMappedRegion.

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

HDCH_MAGIC = 0x48444348
HDCH_VERSION = 1
HDCH_MAX_INLINE_BYTES = 256
LANE_DEPTH = 16
LANE_MASK = LANE_DEPTH - 1
CPU_TO_L2 = 0
L2_TO_CPU = 1
DEFAULT_EPOCHS = 32
DEFAULT_MESSAGES_PER_EPOCH = 8
SUPPORTED_PLATFORMS = {"a2a3sim", "a2a3", "a5sim"}
TIMEOUT_US = 1_000_000

# 64-byte header. The implementation docs originally listed an 80-byte struct;
# the example ABI follows the documented offset table, so lane 0 starts at 64.
CHANNEL_HEADER = struct.Struct("<IIIIIIIIQQQQ")
LANE_HEADER = struct.Struct("<IIIIQQQQQQ")
DESC_HEADER = struct.Struct("<IIQQII")
CHANNEL_HEADER_SIZE = 64
LANE_HEADER_SIZE = 64
DESC_SIZE = 320
CHANNEL_BYTES = CHANNEL_HEADER_SIZE + 2 * (LANE_HEADER_SIZE + LANE_DEPTH * DESC_SIZE)
assert CHANNEL_HEADER.size == CHANNEL_HEADER_SIZE
assert LANE_HEADER.size == LANE_HEADER_SIZE
assert DESC_HEADER.size == 32
assert CHANNEL_BYTES == 10432


@dataclass(frozen=True)
class Message:
    payload: bytes
    route: int
    correlation_id: int
    seq: int
    flags: int = 0


@dataclass
class Channel:
    worker: Worker
    region: MappedRegion
    info: MappedRegionInfo
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

    def __enter__(self) -> Channel:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


def lane_offset(direction: int) -> int:
    if direction == CPU_TO_L2:
        return CHANNEL_HEADER_SIZE
    if direction == L2_TO_CPU:
        return CHANNEL_HEADER_SIZE + LANE_HEADER_SIZE + LANE_DEPTH * DESC_SIZE
    raise ValueError(f"invalid lane direction: {direction}")


def lane_header_offset(direction: int) -> int:
    return lane_offset(direction)


def desc_base_offset(direction: int) -> int:
    return lane_offset(direction) + LANE_HEADER_SIZE


def desc_offset(direction: int, index: int) -> int:
    return desc_base_offset(direction) + (index & LANE_MASK) * DESC_SIZE


def _init_lane(image: bytearray, direction: int) -> None:
    LANE_HEADER.pack_into(
        image,
        lane_header_offset(direction),
        0,
        0,
        LANE_DEPTH,
        LANE_MASK,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def make_initial_channel_image() -> bytes:
    image = bytearray(CHANNEL_BYTES)
    CHANNEL_HEADER.pack_into(
        image,
        0,
        HDCH_MAGIC,
        HDCH_VERSION,
        0,
        1,
        1,
        LANE_DEPTH,
        HDCH_MAX_INLINE_BYTES,
        0,
        CHANNEL_BYTES,
        0,
        0,
        0,
    )
    _init_lane(image, CPU_TO_L2)
    _init_lane(image, L2_TO_CPU)
    return bytes(image)


def _pack_u32(value: int) -> bytes:
    return struct.pack("<I", value & 0xFFFFFFFF)


def _unpack_u32(data: bytes) -> int:
    return struct.unpack("<I", data)[0]


def pack_desc(desc: Message) -> bytes:
    if len(desc.payload) > HDCH_MAX_INLINE_BYTES:
        raise ValueError(f"payload exceeds {HDCH_MAX_INLINE_BYTES} bytes")
    out = bytearray(DESC_SIZE)
    DESC_HEADER.pack_into(
        out,
        0,
        desc.flags,
        len(desc.payload),
        desc.seq,
        desc.correlation_id,
        desc.route,
        0,
    )
    out[64 : 64 + len(desc.payload)] = desc.payload
    return bytes(out)


def unpack_desc(raw: bytes) -> Message:
    if len(raw) != DESC_SIZE:
        raise ValueError(f"descriptor must be {DESC_SIZE} bytes, got {len(raw)}")
    flags, payload_bytes, seq, correlation_id, route, _reserved0 = DESC_HEADER.unpack_from(raw, 0)
    if payload_bytes > HDCH_MAX_INLINE_BYTES:
        raise ValueError(f"descriptor payload_bytes out of range: {payload_bytes}")
    return Message(
        payload=bytes(raw[64 : 64 + payload_bytes]),
        route=route,
        correlation_id=correlation_id,
        seq=seq,
        flags=flags,
    )


def open_channel(worker: Worker, worker_id: int = 0) -> Channel:
    region = worker.open_mapped_region(CHANNEL_BYTES, signal_count=2, worker_id=worker_id)
    try:
        info = worker.mapped_region_info(region, worker_id=worker_id)
        worker.mapped_region_datacopy_h2region(region, 0, make_initial_channel_image(), worker_id=worker_id)
    except Exception:
        worker.close_mapped_region(region, worker_id=worker_id)
        raise
    return Channel(worker=worker, region=region, info=info, worker_id=worker_id)


def read_lane_head_tail(channel: Channel, direction: int) -> tuple[int, int]:
    raw = channel.worker.mapped_region_datacopy_region2h(
        channel.region, lane_header_offset(direction), 8, worker_id=channel.worker_id
    )
    return struct.unpack("<II", raw)


def write_lane_head(channel: Channel, direction: int, head: int) -> None:
    channel.worker.mapped_region_datacopy_h2region(
        channel.region, lane_header_offset(direction), _pack_u32(head), worker_id=channel.worker_id
    )


def write_lane_tail(channel: Channel, direction: int, tail: int) -> None:
    channel.worker.mapped_region_datacopy_h2region(
        channel.region, lane_header_offset(direction) + 4, _pack_u32(tail), worker_id=channel.worker_id
    )


def write_desc(channel: Channel, direction: int, index: int, desc: Message) -> None:
    channel.worker.mapped_region_datacopy_h2region(
        channel.region, desc_offset(direction, index), pack_desc(desc), worker_id=channel.worker_id
    )


def read_desc(channel: Channel, direction: int, index: int) -> Message:
    raw = channel.worker.mapped_region_datacopy_region2h(
        channel.region, desc_offset(direction, index), DESC_SIZE, worker_id=channel.worker_id
    )
    return unpack_desc(raw)


def channel_send_cpu(
    channel: Channel,
    route: int,
    payload: bytes,
    correlation_id: int,
    seq: int,
) -> None:
    head, tail = read_lane_head_tail(channel, CPU_TO_L2)
    if tail - head >= LANE_DEPTH:
        raise RuntimeError("cpu_to_l2 queue is full")
    write_desc(
        channel,
        CPU_TO_L2,
        tail,
        Message(payload=payload, route=route, correlation_id=correlation_id, seq=seq),
    )
    write_lane_tail(channel, CPU_TO_L2, tail + 1)
    channel.worker.mapped_region_notify(channel.region, 0, seq, worker_id=channel.worker_id)


def channel_recv_cpu(channel: Channel, seq: int, timeout_us: int = TIMEOUT_US) -> Message | None:
    channel.worker.mapped_region_wait(channel.region, 1, seq, timeout_us, worker_id=channel.worker_id)
    head, tail = read_lane_head_tail(channel, L2_TO_CPU)
    if head == tail:
        return None
    msg = read_desc(channel, L2_TO_CPU, head)
    write_lane_head(channel, L2_TO_CPU, head + 1)
    return msg


def channel_empty(channel: Channel, direction: int) -> bool:
    head, tail = read_lane_head_tail(channel, direction)
    return head == tail


def make_payload(epoch: int, msg_idx: int) -> bytes:
    size = 17 + ((epoch * 7 + msg_idx * 11) % 96)
    return bytes(((epoch * 19 + msg_idx * 23 + i * 5) & 0xFF) for i in range(size))


def response_payload(seq: int, payload: bytes) -> bytes:
    return bytes((b ^ ((seq + i * 7) & 0xFF)) for i, b in enumerate(payload))


def build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(RUNTIME)

    kernel_bytes = kc.compile_incore(
        source_path=str(KERNEL_DIR / "aiv" / "host_device_spsc_queue_protocol.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=RUNTIME,
        source_path=str(KERNEL_DIR / "orchestration" / "host_device_spsc_queue_protocol_orch.cpp"),
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        func_name="host_device_spsc_queue_protocol_orch",
        config_name="host_device_spsc_queue_protocol_config",
        binary=orch_bytes,
        children=[(0, CoreCallable.build(signature=[], binary=kernel_bytes))],
    )


def _validate_response(expected: Message, got: Message) -> None:
    expected_route = expected.route ^ 0x80000000
    expected_payload = response_payload(expected.seq, expected.payload)
    assert got.seq == expected.seq
    assert got.correlation_id == expected.correlation_id
    assert got.route == expected_route
    assert got.payload == expected_payload


def run(
    platform: str,
    device_id: int,
    *,
    build: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    messages_per_epoch: int = DEFAULT_MESSAGES_PER_EPOCH,
) -> int:
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"unsupported platform: {platform}")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if messages_per_epoch <= 0:
        raise ValueError("messages_per_epoch must be positive")
    if messages_per_epoch > LANE_DEPTH:
        raise ValueError(f"messages_per_epoch must be <= {LANE_DEPTH}")

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
        with open_channel(worker) as channel:
            cfg = CallConfig()
            cfg.block_dim = 1
            cfg.aicpu_thread_num = 2

            total_responses = 0
            for epoch in range(epochs):
                expected: list[Message] = []
                for msg_idx in range(messages_per_epoch):
                    seq = epoch * messages_per_epoch + msg_idx + 1
                    payload = make_payload(epoch, msg_idx)
                    route = 0x1000 + msg_idx
                    correlation_id = (seq << 32) | msg_idx
                    msg = Message(payload=payload, route=route, correlation_id=correlation_id, seq=seq)
                    channel_send_cpu(channel, route, payload, correlation_id, seq)
                    expected.append(msg)

                def orch_fn(orch, _args, run_cfg):
                    args = TaskArgs()
                    args.add_scalar(channel.device_data_ptr)
                    args.add_scalar(channel.device_signal_ptr)
                    args.add_scalar(expected[-1].seq)
                    args.add_scalar(messages_per_epoch)
                    orch.submit_next_level(chip_cid, args, run_cfg, worker=0)

                worker.run(orch_fn, args=None, config=cfg)

                for msg in expected:
                    got = channel_recv_cpu(channel, msg.seq)
                    assert got is not None
                    _validate_response(msg, got)
                    total_responses += 1
                assert channel_recv_cpu(channel, expected[-1].seq) is None

            assert total_responses == epochs * messages_per_epoch
            assert channel_empty(channel, CPU_TO_L2)
            assert channel_empty(channel, L2_TO_CPU)
        return 0
    finally:
        worker.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", required=True, choices=sorted(SUPPORTED_PLATFORMS))
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--build", action="store_true", help="Rebuild runtime from source.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--messages-per-epoch", type=int, default=DEFAULT_MESSAGES_PER_EPOCH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rc = run(
        args.platform,
        args.device,
        build=args.build,
        epochs=args.epochs,
        messages_per_epoch=args.messages_per_epoch,
    )
    print(
        "[host_device_spsc_queue_protocol] "
        f"platform={args.platform} device={args.device} epochs={args.epochs} "
        f"messages_per_epoch={args.messages_per_epoch} PASSED"
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
