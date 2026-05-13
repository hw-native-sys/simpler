#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small L3 multi-communication-domain communication demo.

Three chip workers declare two overlapping domains:

  even = workers [0, 2]
  tail = workers [1, 2]

The example bootstraps the worker, prints the domain-local rank view seen by
each chip child, then runs one small allreduce in each domain.  It verifies
that:

  * non-member domains are absent and raise KeyError through ctx.domains;
  * worker_indices order defines the dense domain_rank;
  * overlapping domains have separate buffer pointers.
  * each domain transfers data only among its own participants.

Run:
    python examples/workers/l3/domain_rank_map/main.py -p a2a3    -d 0-2
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (  # noqa: E402
    ArgDirection,
    CallConfig,
    ChipBufferSpec,
    ChipCallable,
    ChipCommDomainContext,
    CommDomain,
    CommDomainPlan,
    ContinuousTensor,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker  # noqa: E402

from simpler_setup.elf_parser import extract_text_section  # noqa: E402
from simpler_setup.kernel_compiler import KernelCompiler  # noqa: E402
from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: E402
from simpler_setup.torch_interop import make_tensor_arg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAP_DIR = os.path.join(HERE, "../dual_domain_overlap")
COUNT = 256
DTYPE_NBYTES = 4
SIGNAL_TAIL_NBYTES = 16 * 4
SCRATCH_NBYTES = COUNT * DTYPE_NBYTES + SIGNAL_TAIL_NBYTES
DOMAINS = {
    "even": [0, 2],
    "tail": [1, 2],
}


def parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if len(ids) != 3:
        raise ValueError(f"domain_rank_map needs exactly 3 devices, got {ids}")
    return ids


def _make_comm_plan() -> CommDomainPlan:
    return CommDomainPlan(
        domains=[
            CommDomain(
                name=name,
                worker_indices=worker_indices,
                window_size=max(SCRATCH_NBYTES, 4096),
                buffers=[
                    ChipBufferSpec(
                        name="scratch",
                        dtype="float32",
                        count=COUNT,
                        nbytes=SCRATCH_NBYTES,
                    ),
                ],
            )
            for name, worker_indices in DOMAINS.items()
        ]
    )


def build_allreduce_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]
    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(OVERLAP_DIR, "kernels/aiv/domain_allreduce_sum.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    kernel_bytes = extract_text_section(kernel_bytes)
    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(OVERLAP_DIR, "kernels/orchestration/domain_allreduce_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        func_name="domain_allreduce_orchestration",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def _add_domain_scratch(args: TaskArgs, domain: ChipCommDomainContext) -> None:
    args.add_tensor(
        ContinuousTensor.make(
            data=domain.buffer_ptrs["scratch"],
            shapes=(COUNT,),
            dtype=DataType.FLOAT32,
            child_memory=True,
        ),
        TensorArgType.INOUT,
    )
    args.add_scalar(domain.domain_size)
    args.add_scalar(domain.device_ctx)


def _check_missing(ctx_domains: dict, name: str) -> bool:
    try:
        _ = ctx_domains[name]
    except KeyError:
        return True
    return False


def run(platform: str, device_ids: list[int]) -> int:
    if platform != "a2a3":
        raise ValueError("domain_rank_map uses PTO-ISA remote-window kernels and requires a2a3 hardware")

    print(f"[domain_rank_map] platform={platform} devices={device_ids}")
    host_inputs = {
        worker_idx: torch.tensor(
            [worker_idx * 100 + i for i in range(COUNT)],
            dtype=torch.float32,
        ).share_memory_()
        for worker_idx in range(len(device_ids))
    }
    outputs = {
        name: {
            worker_idx: torch.zeros(COUNT, dtype=torch.float32).share_memory_() for worker_idx in worker_indices
        }
        for name, worker_indices in DOMAINS.items()
    }

    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        comm_plan=_make_comm_plan(),
    )
    print("[domain_rank_map] compiling communication kernel...")
    allreduce_cid = worker.register(build_allreduce_callable(platform))

    try:
        print("[domain_rank_map] init worker...")
        worker.init()
        contexts = worker.chip_contexts

        expected = {
            0: {"even": 0},
            1: {"tail": 0},
            2: {"even": 1, "tail": 1},
        }
        ok = len(contexts) == 3

        for worker_idx, ctx in enumerate(contexts):
            actual_names = sorted(ctx.domains)
            print(f"[domain_rank_map] worker {worker_idx}: domains={actual_names}")

            expected_names = sorted(expected[worker_idx])
            if actual_names != expected_names:
                print(f"  expected domains {expected_names}, got {actual_names}")
                ok = False

            for name, expected_rank in expected[worker_idx].items():
                domain = ctx.domains[name]
                print(
                    f"  {name}: domain_rank={domain.domain_rank} "
                    f"domain_size={domain.domain_size} "
                    f"scratch=0x{domain.buffer_ptrs['scratch']:x}"
                )
                if domain.domain_rank != expected_rank or domain.domain_size != 2:
                    ok = False
                if domain.device_ctx == 0 or domain.buffer_ptrs["scratch"] == 0:
                    ok = False

        if not _check_missing(contexts[0].domains, "tail"):
            print("[domain_rank_map] worker 0 unexpectedly has tail domain")
            ok = False
        if not _check_missing(contexts[1].domains, "even"):
            print("[domain_rank_map] worker 1 unexpectedly has even domain")
            ok = False

        worker2 = contexts[2]
        if worker2.domains["even"].buffer_ptrs["scratch"] == worker2.domains["tail"].buffer_ptrs["scratch"]:
            print("[domain_rank_map] worker 2 domains share a scratch pointer")
            ok = False

        def reduce_orch_fn(domain_name: str):
            def _orch_fn(orch, _args, cfg):
                for worker_idx in DOMAINS[domain_name]:
                    domain = contexts[worker_idx].domains[domain_name]
                    args = TaskArgs()
                    args.add_tensor(make_tensor_arg(host_inputs[worker_idx]), TensorArgType.INPUT)
                    args.add_tensor(make_tensor_arg(outputs[domain_name][worker_idx]), TensorArgType.OUTPUT_EXISTING)
                    _add_domain_scratch(args, domain)
                    orch.submit_next_level(allreduce_cid, args, cfg, worker=worker_idx)

            return _orch_fn

        print("[domain_rank_map] running one allreduce per domain...")
        for domain_name in DOMAINS:
            worker.run(reduce_orch_fn(domain_name), args=None, config=CallConfig())

        for domain_name, worker_indices in DOMAINS.items():
            expected_tensor = sum(host_inputs[worker_idx] for worker_idx in worker_indices)
            for worker_idx in worker_indices:
                got = outputs[domain_name][worker_idx]
                max_diff = float(torch.max(torch.abs(got - expected_tensor)))
                print(f"[domain_rank_map] {domain_name} worker {worker_idx}: max_diff={max_diff:.3e}")
                if max_diff > 1e-3:
                    ok = False

        if not ok:
            print("[domain_rank_map] checks FAILED")
            return 1
        print("[domain_rank_map] communication checks PASSED")
        return 0
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3"])
    parser.add_argument("-d", "--device", default="0-2", help="Device range, e.g. '0-2'. Three chips required.")
    cli = parser.parse_args()
    return run(cli.platform, parse_device_range(cli.device))


if __name__ == "__main__":
    sys.exit(main())
