#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host CPU to device NPU round-trip through HostDeviceMappedRegion."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from simpler.task_interface import ArgDirection, CallConfig, ChipCallable, CoreCallable, TaskArgs
from simpler.worker import Worker
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.runtime_builder import RuntimeBuilder


HERE = Path(__file__).resolve().parent
KERNEL_DIR = HERE / "kernels"
RUNTIME = "tensormap_and_ringbuffer"
DEFAULT_DATA_BYTES = 256
DEFAULT_ITERS = 10


def _build_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(RUNTIME)

    incore = kc.compile_incore(
        source_path=str(KERNEL_DIR / "aiv" / "host_device_mapped_region_round_trip.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        incore = extract_text_section(incore)

    orch = kc.compile_orchestration(
        runtime_name=RUNTIME,
        source_path=str(KERNEL_DIR / "orchestration" / "host_device_mapped_region_round_trip_orch.cpp"),
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.IN],
        func_name="host_device_mapped_region_round_trip_orch",
        binary=orch,
        children=[(0, CoreCallable.build(signature=[], binary=incore))],
    )


def _pattern(seq: int, data_bytes: int) -> bytes:
    return bytes(((seq * 17 + i * 5) & 0xFF) for i in range(data_bytes))


def _expected(seq: int, payload: bytes) -> bytes:
    return bytes((b ^ ((seq + i * 3) & 0xFF)) for i, b in enumerate(payload))


def run(
    platform: str,
    device_id: int,
    *,
    build: bool = False,
    iters: int = DEFAULT_ITERS,
    data_bytes: int = DEFAULT_DATA_BYTES,
) -> None:
    if platform not in {"a2a3sim", "a2a3"}:
        raise ValueError(f"unsupported platform: {platform}")
    if iters <= 0:
        raise ValueError("iters must be positive")
    if data_bytes <= 0:
        raise ValueError("data_bytes must be positive")

    os.environ["PTO_ISA_ROOT"] = ensure_pto_isa_root(clone_protocol="https")
    RuntimeBuilder(platform=platform).get_binaries(RUNTIME, build=build)
    chip_callable = _build_callable(platform)

    worker = Worker(level=2, platform=platform, runtime=RUNTIME, device_id=device_id, build=build)
    worker.init()
    region = None
    try:
        chip_cid = worker.register(chip_callable)
        region = worker.open_mapped_region(data_bytes * 2, signal_count=2)
        info = worker.mapped_region_info(region)
        assert info.host_data_ptr == 0
        assert info.host_signal_ptr == 0
        assert info.device_data_ptr != 0
        assert info.device_signal_ptr != 0

        cfg = CallConfig()
        cfg.block_dim = 1
        cfg.aicpu_thread_num = 2

        for seq in range(1, iters + 1):
            payload = _pattern(seq, data_bytes)
            worker.mapped_region_datacopy_h2region(region, 0, payload)
            worker.mapped_region_notify(region, 0, seq)

            args = TaskArgs()
            args.add_scalar(info.device_data_ptr)
            args.add_scalar(info.device_signal_ptr)
            args.add_scalar(seq)
            args.add_scalar(data_bytes)
            worker.run(chip_cid, args, cfg)

            worker.mapped_region_wait(region, 1, seq, 1_000_000)
            got = worker.mapped_region_datacopy_region2h(region, data_bytes, data_bytes)
            assert got == _expected(seq, payload)
    finally:
        if region is not None:
            worker.close_mapped_region(region)
        worker.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3sim", "a2a3"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--build", action="store_true", help="Rebuild runtime from source.")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--data-bytes", type=int, default=DEFAULT_DATA_BYTES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args.platform, args.device, build=args.build, iters=args.iters, data_bytes=args.data_bytes)
    print(
        "[host_device_mapped_region_round_trip] "
        f"platform={args.platform} device={args.device} iters={args.iters} data_bytes={args.data_bytes} PASSED"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
