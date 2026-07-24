#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run one L2 compute task and one cross-machine communication task from L4."""

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
LOCAL_ADD_AIV = os.path.join(HERE, "kernels", "aiv", "local_add_kernel.cpp")
LOCAL_ADD_ORCH = os.path.join(HERE, "kernels", "orchestration", "local_add_orch.cpp")
GLOBAL_TLOAD_AIV = os.path.join(HERE, "kernels", "aiv", "global_tload_kernel.cpp")
GLOBAL_TLOAD_ORCH = os.path.join(HERE, "kernels", "orchestration", "global_tload_orch.cpp")
COUNT = 256
FLOAT_BYTES = 4
MAX_RANKS = 16
WINDOW_SIZE = 4096


def _compile_aiv(compiler: KernelCompiler, platform: str, runtime: str, source: str) -> bytes:
    include_dirs = compiler.get_orchestration_include_dirs(runtime)
    kernel_include_dirs = list(include_dirs) + [str(compiler.project_root / "src" / "common")]
    kernel_bytes = compiler.compile_incore(
        source_path=source,
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=kernel_include_dirs,
    )
    return kernel_bytes if platform.endswith("sim") else extract_text_section(kernel_bytes)


def _build_chip_callable(
    *,
    platform: str,
    runtime: str,
    kernel_source: str,
    orchestration_source: str,
    signature: list[ArgDirection],
    func_name: str,
    config_name: str,
) -> ChipCallable:
    compiler = KernelCompiler(platform=platform)
    kernel_bytes = _compile_aiv(compiler, platform, runtime, kernel_source)
    orchestration_bytes = compiler.compile_orchestration(
        runtime_name=runtime,
        source_path=orchestration_source,
    )
    core = CoreCallable.build(signature=signature, binary=kernel_bytes)
    return ChipCallable.build(
        signature=signature,
        func_name=func_name,
        config_name=config_name,
        binary=orchestration_bytes,
        children=[(0, core)],
    )


def build_compute_callable(platform: str, runtime: str) -> ChipCallable:
    return _build_chip_callable(
        platform=platform,
        runtime=runtime,
        kernel_source=LOCAL_ADD_AIV,
        orchestration_source=LOCAL_ADD_ORCH,
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="local_add_orchestration",
        config_name="local_add_orchestration_config",
    )


def build_communication_callable(platform: str, runtime: str) -> ChipCallable:
    return _build_chip_callable(
        platform=platform,
        runtime=runtime,
        kernel_source=GLOBAL_TLOAD_AIV,
        orchestration_source=GLOBAL_TLOAD_ORCH,
        signature=[ArgDirection.IN, ArgDirection.OUT],
        func_name="global_tload_orchestration",
        config_name="global_tload_orchestration_config",
    )


def _digest_scalars(digest: bytes) -> tuple[int, ...]:
    if len(digest) != 32:
        raise ValueError("callable digest must be 32 bytes")
    return tuple(int.from_bytes(digest[offset : offset + 8], "little") for offset in range(0, 32, 8))


def _lhs_values(rank: int) -> tuple[float, ...]:
    return tuple(float(rank * 100 + index) for index in range(COUNT))


def _rhs_values(rank: int) -> tuple[float, ...]:
    return tuple(float(rank * 10 + 2 * index) for index in range(COUNT))


def _expected_compute(rank: int) -> tuple[float, ...]:
    return tuple(lhs + rhs for lhs, rhs in zip(_lhs_values(rank), _rhs_values(rank), strict=True))


def _expected_communication(rank_count: int) -> tuple[float, ...]:
    rank_results = tuple(_expected_compute(rank) for rank in range(rank_count))
    return tuple(sum(rank_result[index] for rank_result in rank_results) for index in range(COUNT))


def _remote_args(domain_id: int, chip_digest: bytes) -> TaskArgs:
    args = TaskArgs()
    args.add_scalar(domain_id)
    args.add_scalar(0)
    for value in _digest_scalars(chip_digest):
        args.add_scalar(value)
    return args


def _unpack_floats(raw: bytes) -> tuple[float, ...]:
    return tuple(float(value) for value in struct.unpack(f"<{COUNT}f", raw))


def _max_diff(actual: tuple[float, ...], expected: tuple[float, ...]) -> float:
    return max(abs(observed - wanted) for observed, wanted in zip(actual, expected, strict=True))


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
    print(f"[l4-compute-comm] compiling for {platform}/{runtime}")
    compute_handle = worker.register(build_compute_callable(platform, runtime))
    communication_handle = worker.register(build_communication_callable(platform, runtime))
    remote_compute_handle = worker.register(
        RemoteCallable("simpler.global_comm_smoke:remote_compute_orch"),
        workers=list(node_ids),
    )
    remote_communication_handle = worker.register(
        RemoteCallable("simpler.global_comm_smoke:remote_rank_orch"),
        workers=list(node_ids),
    )
    captured: dict[str, object] = {}
    observed_compute: list[tuple[float, ...]] = []
    observed_communication: list[tuple[float, ...]] = []
    try:
        worker.init()

        def compute_phase(orch, _args, cfg):
            domain = orch.allocate_global_domain(
                name="a3-l4-compute-then-tload",
                members=tuple((node_id, 0) for node_id in node_ids),
                window_size=WINDOW_SIZE,
                buffers=(
                    CommBufferSpec("lhs", "float32", COUNT, COUNT * FLOAT_BYTES),
                    CommBufferSpec("rhs", "float32", COUNT, COUNT * FLOAT_BYTES),
                    CommBufferSpec("input", "float32", COUNT, COUNT * FLOAT_BYTES),
                    CommBufferSpec("result", "float32", COUNT, COUNT * FLOAT_BYTES),
                ),
                retain_after_run=True,
            )
            for rank, node_id in enumerate(node_ids):
                orch.copy_to_global_domain(
                    domain,
                    rank,
                    struct.pack(f"<{COUNT}f", *_lhs_values(rank)),
                    buffer="lhs",
                )
                orch.copy_to_global_domain(
                    domain,
                    rank,
                    struct.pack(f"<{COUNT}f", *_rhs_values(rank)),
                    buffer="rhs",
                )
                orch.submit_next_level(
                    remote_compute_handle,
                    _remote_args(domain.domain_id, compute_handle.digest),
                    cfg,
                    worker=node_id,
                )
            captured["domain"] = domain

        worker.run(compute_phase, args=None, config=CallConfig())
        domain = captured["domain"]

        def communication_phase(orch, _args, cfg):
            for rank, node_id in enumerate(node_ids):
                raw = orch.copy_from_global_domain(
                    domain,
                    rank,
                    COUNT * FLOAT_BYTES,
                    buffer="input",
                )
                observed_compute.append(_unpack_floats(raw))
                orch.submit_next_level(
                    remote_communication_handle,
                    _remote_args(domain.domain_id, communication_handle.digest),
                    cfg,
                    worker=node_id,
                )

        worker.run(communication_phase, args=None, config=CallConfig())

        def verify_phase(orch, _args, _cfg):
            try:
                for rank in range(len(node_ids)):
                    raw = orch.copy_from_global_domain(
                        domain,
                        rank,
                        COUNT * FLOAT_BYTES,
                        buffer="result",
                    )
                    observed_communication.append(_unpack_floats(raw))
            finally:
                domain.release()

        worker.run(verify_phase, args=None, config=CallConfig())

        for rank, result in enumerate(observed_compute):
            max_diff = _max_diff(result, _expected_compute(rank))
            print(f"[l4-compute-comm] compute rank={rank} max_diff={max_diff:.3e}")
            if max_diff > 1e-5:
                raise AssertionError(f"rank {rank} compute golden mismatch: max_diff={max_diff}")

        expected_communication = _expected_communication(len(node_ids))
        for rank, result in enumerate(observed_communication):
            max_diff = _max_diff(result, expected_communication)
            print(f"[l4-compute-comm] communication rank={rank} max_diff={max_diff:.3e}")
            if max_diff > 1e-3:
                raise AssertionError(f"rank {rank} communication golden mismatch: max_diff={max_diff}")

        print("[l4-compute-comm] PASS: L4 -> L3 -> L2 compute and peer TLOAD succeeded")
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
