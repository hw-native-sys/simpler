#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""No-mpirun A3 L4 -> remote L3 -> L2 Fabric TLOAD smoke."""

from __future__ import annotations

import argparse
import os
import struct
import sys

from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    TaskArgs,
)
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_AIV = os.path.join(HERE, "kernels", "aiv", "global_tload_kernel.cpp")
KERNEL_ORCH = os.path.join(HERE, "kernels", "orchestration", "global_tload_orch.cpp")
COUNT = 256
FLOAT_BYTES = 4
MAX_RANKS = 16
WINDOW_SIZE = 4096


def build_chip_callable(platform: str, runtime: str) -> ChipCallable:
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = compiler.get_orchestration_include_dirs(runtime)
    kernel_include_dirs = list(include_dirs) + [str(compiler.project_root / "src" / "common")]
    kernel_bytes = compiler.compile_incore(
        source_path=KERNEL_AIV,
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)
    orch_bytes = compiler.compile_orchestration(runtime_name=runtime, source_path=KERNEL_ORCH)
    core = CoreCallable.build(signature=[ArgDirection.IN, ArgDirection.OUT], binary=kernel_bytes)
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT],
        func_name="global_tload_orchestration",
        config_name="global_tload_orchestration_config",
        binary=orch_bytes,
        children=[(0, core)],
    )


def _digest_scalars(digest: bytes) -> tuple[int, ...]:
    if len(digest) != 32:
        raise ValueError("callable digest must be 32 bytes")
    return tuple(int.from_bytes(digest[offset : offset + 8], "little") for offset in range(0, 32, 8))


def _input_values(rank: int) -> tuple[float, ...]:
    return tuple(float(rank * 100 + index) for index in range(COUNT))


def _expected_values(rank_count: int) -> tuple[float, ...]:
    rank_bias = 100 * rank_count * (rank_count - 1) // 2
    return tuple(float(rank_count * index + rank_bias) for index in range(COUNT))


def run(endpoints: list[str], device_ids: list[int], platform: str, runtime: str) -> int:
    if len(endpoints) != len(device_ids):
        raise ValueError("--endpoint and --device-id counts must match")
    if not 2 <= len(endpoints) <= MAX_RANKS:
        raise ValueError(f"the smoke requires between 2 and {MAX_RANKS} nodes")

    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=120)
    node_ids = tuple(
        worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=endpoint,
                platform=platform,
                runtime=runtime,
                device_ids=(device_id,),
                comm_profile="a3-fabric-v1",
            )
        )
        for endpoint, device_id in zip(endpoints, device_ids, strict=True)
    )
    print(f"[l4-global-tload] compiling for {platform}/{runtime}")
    chip_handle = worker.register(build_chip_callable(platform, runtime))
    remote_handle = worker.register(
        RemoteCallable("simpler.global_comm_smoke:remote_rank_orch"),
        workers=list(node_ids),
    )
    captured: dict[str, object] = {}
    try:
        worker.init()

        def build_and_run(orch, _args, cfg):
            domain = orch.allocate_global_domain(
                name="a3-l4-tload",
                members=tuple((node_id, 0) for node_id in node_ids),
                window_size=WINDOW_SIZE,
                buffers=(
                    CommBufferSpec("input", "float32", COUNT, COUNT * FLOAT_BYTES),
                    CommBufferSpec("result", "float32", COUNT, COUNT * FLOAT_BYTES),
                ),
                retain_after_run=True,
            )
            for rank in range(len(node_ids)):
                orch.copy_to_global_domain(
                    domain,
                    rank,
                    struct.pack(f"<{COUNT}f", *_input_values(rank)),
                    buffer="input",
                )
            digest_scalars = _digest_scalars(chip_handle.digest)
            for node_id in node_ids:
                rank_args = TaskArgs()
                rank_args.add_scalar(domain.domain_id)
                rank_args.add_scalar(0)
                for value in digest_scalars:
                    rank_args.add_scalar(value)
                orch.submit_next_level(remote_handle, rank_args, cfg, worker=node_id)
            captured["domain"] = domain

        worker.run(build_and_run, args=None, config=CallConfig())
        domain = captured["domain"]
        expected = _expected_values(len(node_ids))
        observed: list[tuple[float, ...]] = []

        def read_and_release(orch, _args, _cfg):
            for rank in range(len(node_ids)):
                raw = orch.copy_from_global_domain(
                    domain,
                    rank,
                    COUNT * FLOAT_BYTES,
                    buffer="result",
                )
                observed.append(tuple(float(value) for value in struct.unpack(f"<{COUNT}f", raw)))
            domain.release()

        worker.run(read_and_release, args=None, config=CallConfig())
        for rank, result in enumerate(observed):
            max_diff = max(abs(actual - wanted) for actual, wanted in zip(result, expected, strict=True))
            print(f"[l4-global-tload] rank={rank} max_diff={max_diff:.3e}")
            if max_diff > 1e-3:
                print("[l4-global-tload] FAILED")
                return 1
        print("[l4-global-tload] PASS: L4-brokered Fabric descriptors and peer TLOAD both succeeded")
        return 0
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", required=True, help="Remote L3 daemon endpoint, HOST:PORT")
    parser.add_argument("--device-id", action="append", required=True, type=int, help="One device id per endpoint")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    args = parser.parse_args()
    return run(args.endpoint, args.device_id, args.platform, args.runtime)


if __name__ == "__main__":
    sys.exit(main())
