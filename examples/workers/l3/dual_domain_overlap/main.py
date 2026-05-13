#!/ usr / bin / env python3
#Copyright(c) PyPTO Contributors.
#This program is free software, you can redistribute it and / or modify it under the terms and conditions of
#CANN Open Software License Agreement Version 2.0(the "License").
#Please refer to the License for details.You may not use this file except in compliance with the License.
#THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#INCLUDING BUT NOT LIMITED TO NON - INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#See LICENSE in the root of the software repository for the full text of the License.
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
"""L3 multi-communication-domain demo with overlapping domains.

Three chip workers form two communication domains:

  left   = workers [0, 1]
  right  = workers [1, 2]

Worker 1 participates in both domains and receives two independent domain
contexts.  The L3 orchestration submits communication in both domains and
then submits affine compute tasks that depend only on their own domain's
reduced tensor.

Hardware only.  Run:
    python examples/workers/l3/dual_domain_overlap/main.py -d 0-2
"""

from __future__ import annotations

import argparse
import ctypes
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
    ChipContext,
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

COUNT = 256
DTYPE_NBYTES = 4
SIGNAL_TAIL_NBYTES = 16 * 4
SCRATCH_NBYTES = COUNT * DTYPE_NBYTES + SIGNAL_TAIL_NBYTES
COMM_MAX_RANK_NUM = 64
DOMAINS = {
    "left": [0, 1],
    "right": [1, 2],
}


def parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if len(ids) != 3:
        raise ValueError(f"dual_domain_overlap needs exactly 3 devices, got {ids}")
    return ids


def _kernel_compiler(platform: str) -> tuple[KernelCompiler, str, list[str], list[str]]:
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]
    return kc, pto_isa_root, list(include_dirs), kernel_include_dirs


def build_allreduce_callable(platform: str) -> ChipCallable:
    kc, pto_isa_root, _, kernel_include_dirs = _kernel_compiler(platform)
    runtime = "tensormap_and_ringbuffer"
    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/domain_allreduce_sum.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    kernel_bytes = extract_text_section(kernel_bytes)
    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/domain_allreduce_orch.cpp"),
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


def build_affine_callable(platform: str) -> ChipCallable:
    kc, pto_isa_root, _, kernel_include_dirs = _kernel_compiler(platform)
    runtime = "tensormap_and_ringbuffer"
    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/affine_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    kernel_bytes = extract_text_section(kernel_bytes)
    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/affine_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="affine_orchestration",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def _domain_tensor_map(fill_value: float = 0.0) -> dict[str, dict[int, torch.Tensor]]:
    return {
name: {
worker_idx: torch.full((COUNT,), fill_value, dtype=torch.float32).share_memory_() for worker_idx in ids
}
        for
            name, ids in DOMAINS.items()
}

def _make_comm_plan() -> CommDomainPlan:
    window_size = max(SCRATCH_NBYTES, 4 * 1024)
    return CommDomainPlan(
        domains=[
            CommDomain(
                name=name,
                worker_indices=worker_indices,
                window_size=window_size,
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


def run(device_ids: list[int]) -> int:
    nranks = len(device_ids)
    assert nranks == 3
    print(f"[dual_domain_overlap] devices={device_ids}")

    host_x = [
        torch.tensor([rank * 100 + i for i in range(COUNT)], dtype=torch.float32).share_memory_()
        for rank in range(nranks)
    ]
    scale = [
        torch.full((COUNT,), 0.5 + rank * 0.01, dtype=torch.float32).share_memory_() for rank in range(nranks)
    ]
    bias = [torch.full((COUNT,), 10.0 + rank, dtype=torch.float32).share_memory_() for rank in range(nranks)]
    reduce_out = _domain_tensor_map()
    affine_out = _domain_tensor_map()

    print("[dual_domain_overlap] compiling kernels...")
    allreduce_cc = build_allreduce_callable("a2a3")
    affine_cc = build_affine_callable("a2a3")

    worker = Worker(
        level=3,
        platform="a2a3",
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        comm_plan=_make_comm_plan(),
    )
    allreduce_cid = worker.register(allreduce_cc)
    affine_cid = worker.register(affine_cc)

    try:
        print("[dual_domain_overlap] init worker...")
        worker.init()
        contexts: list[ChipContext] = worker.chip_contexts
        assert len(contexts) == nranks
        for worker_idx, ctx in enumerate(contexts):
            names = ", ".join(sorted(ctx.domains))
            print(f"[dual_domain_overlap] worker {worker_idx}: device={ctx.device_id} domains={names}")
            for name, domain in sorted(ctx.domains.items()):
                print(
                    f"  {name}: rank={domain.domain_rank}/{domain.domain_size} "
                    f"scratch=0x{domain.buffer_ptrs['scratch']:x} ctx=0x{domain.device_ctx:x}"
                )

        def reduce_orch_fn(domain_name: str):
            def _orch_fn(orch, _args, cfg):
                worker_indices = DOMAINS[domain_name]
                for worker_idx in worker_indices:
                    domain = contexts[worker_idx].domains[domain_name]
                    args = TaskArgs()
                    args.add_tensor(make_tensor_arg(host_x[worker_idx]), TensorArgType.INPUT)
                    args.add_tensor(make_tensor_arg(reduce_out[domain_name][worker_idx]), TensorArgType.OUTPUT_EXISTING)
                    _add_domain_scratch(args, domain)
                    orch.submit_next_level(allreduce_cid, args, cfg, worker=worker_idx)

            return _orch_fn

        def affine_orch_fn(orch, _args, cfg):
            for domain_name, worker_indices in DOMAINS.items():
                args_list = []
                for worker_idx in worker_indices:
                    args = TaskArgs()
                    args.add_tensor(make_tensor_arg(reduce_out[domain_name][worker_idx]), TensorArgType.INPUT)
                    args.add_tensor(make_tensor_arg(scale[worker_idx]), TensorArgType.INPUT)
                    args.add_tensor(make_tensor_arg(bias[worker_idx]), TensorArgType.INPUT)
                    args.add_tensor(make_tensor_arg(affine_out[domain_name][worker_idx]), TensorArgType.OUTPUT_EXISTING)
                    args_list.append(args)
                orch.submit_next_level_group(affine_cid, args_list, cfg, workers=worker_indices)

        print("[dual_domain_overlap] running two-domain DAG...")
        for domain_name in DOMAINS:
            worker.run(reduce_orch_fn(domain_name), args=None, config=CallConfig())
        worker.run(affine_orch_fn, args=None, config=CallConfig())

        ok = True
        for domain_name, worker_indices in DOMAINS.items():
            expected_reduce = sum(host_x[worker_idx] for worker_idx in worker_indices)
            for worker_idx in worker_indices:
                expected_affine = expected_reduce * scale[worker_idx] + bias[worker_idx]
                reduce_diff = float(torch.max(torch.abs(reduce_out[domain_name][worker_idx] - expected_reduce)))
                affine_diff = float(torch.max(torch.abs(affine_out[domain_name][worker_idx] - expected_affine)))
                print(
                    f"[dual_domain_overlap] {domain_name} worker {worker_idx}: "
                    f"reduce_diff={reduce_diff:.3e} affine_diff={affine_diff:.3e}"
                )
                if not torch.isfinite(reduce_out[domain_name][worker_idx]).all():
                    ok = False
                    print(f"  reduce_out sample: {reduce_out[domain_name][worker_idx][:4].tolist()}")
                if not torch.isfinite(affine_out[domain_name][worker_idx]).all():
                    ok = False
                    print(f"  affine_out sample: {affine_out[domain_name][worker_idx][:4].tolist()}")
                if reduce_diff > 1e-3 or affine_diff > 1e-3:
                    ok = False

        if not ok:
            print("[dual_domain_overlap] golden check FAILED")
            return 1
        print("[dual_domain_overlap] all golden checks PASSED")
        return 0
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--device", default="0-2", help="Device range, e.g. '0-2'. Three chips required.")
    cli = parser.parse_args()
    return run(parse_device_range(cli.device))


if __name__ == "__main__":
    sys.exit(main())
