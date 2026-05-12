#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end FFN tensor-parallel demo — two-stage orchestration.

Per rank, in one orch_fn:

  Stage 1 (AIC matmul):  partial_local = x_shard @ w_shard
  Stage 2 (AIV reduce):  y             = sum_over_ranks(partial_local)

partial_local is a per-rank torch.share_memory_() tensor; it is the OUTPUT of
stage 1 and the INPUT of stage 2.  Because both submits see the same
``buffer.addr``, the framework's TensorMap discovers the producer/consumer
edge automatically — no manual barriers in Python.  Cross-rank exchange in
stage 2 still goes through a per-chip ``scratch`` HCCL-window buffer (laid
out as ``[mailbox: nranks * M*N floats | signal tail: nranks int32 slots]``).

Hardware only.  Run:
    python examples/workers/l3/ffn_tp_parallel/main.py -d 0-1
"""

from __future__ import annotations

import argparse
import os
import sys

# Workaround for the duplicate-libomp abort when homebrew numpy and pip torch
# coexist in one macOS process. Harmless on Linux. Must be set before
# ``import torch``. See docs/troubleshooting/macos-libomp-collision.md.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (  # noqa: E402
    ArgDirection,
    CallConfig,
    ChipBootstrapConfig,
    ChipBufferSpec,
    ChipCallable,
    ChipCommBootstrapConfig,
    ChipContext,
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

# Must match TILE / kRows / kCols in the AIC and AIV kernels.
M = 64
K = 64
N = 64
DTYPE_NBYTES = 4  # float32
PARTIAL_NBYTES = M * N * DTYPE_NBYTES


def parse_device_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if len(ids) != 2:
        raise ValueError(f"ffn_tp_parallel needs exactly 2 devices, got {ids}")
    return ids


def _kernel_compiler(platform: str) -> tuple[KernelCompiler, str, list[str], list[str]]:
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root(clone_protocol="https")
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    # The allreduce_sum kernel resolves CommContext from
    # "platform_comm/comm_context.h" under src/common/.
    kernel_include_dirs = list(include_dirs) + [str(kc.project_root / "src" / "common")]
    return kc, pto_isa_root, list(include_dirs), kernel_include_dirs


def build_ffn_local_callable(platform: str) -> ChipCallable:
    """AIC matmul: x_shard @ w_shard -> partial_local."""
    kc, pto_isa_root, _, kernel_include_dirs = _kernel_compiler(platform)
    runtime = "tensormap_and_ringbuffer"

    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aic/kernel_local_linear.cpp"),
        core_type="aic",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/ffn_local_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="ffn_local_orchestration",
        config_name="ffn_local_orchestration_config",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def build_allreduce_sum_callable(platform: str) -> ChipCallable:
    """AIV cross-rank sum (4-phase publish/notify/wait/accumulate)."""
    kc, pto_isa_root, _, kernel_include_dirs = _kernel_compiler(platform)
    runtime = "tensormap_and_ringbuffer"

    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/kernel_allreduce_sum.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/allreduce_sum_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT],
        func_name="allreduce_sum_orchestration",
        config_name="allreduce_sum_orchestration_config",
        binary=orch_bytes,
        children=[(1, core_callable)],
    )


def make_rank_inputs(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Match golden formula from PR #522 (golden.py)."""
    x = (torch.arange(M * K, dtype=torch.float32).reshape(M, K) + float(rank) * 0.25) / 32.0
    w = (torch.arange(K * N, dtype=torch.float32).reshape(K, N) + float(rank + 1) * 0.5) / 48.0
    return x, w


def run(device_ids: list[int]) -> int:
    nranks = len(device_ids)
    # scratch = mailbox(nranks * M*N floats) + signal tail (nranks int32).
    scratch_count = nranks * M * N
    scratch_nbytes = scratch_count * DTYPE_NBYTES + nranks * 4
    window_size = max(scratch_nbytes, 4 * 1024)

    rootinfo_path = f"/tmp/pto_ffn_tp_parallel_rootinfo_{os.getpid()}.bin"
    try:
        os.unlink(rootinfo_path)
    except FileNotFoundError:
        pass

    print(f"[ffn_tp_parallel] devices={device_ids} nranks={nranks} M={M} K={K} N={N}")

    # Per-rank host tensors via torch.share_memory_(): inputs, partial_local
    # (stage1 output / stage2 input), and final y (stage2 output).
    host_x_shards = [make_rank_inputs(r)[0].share_memory_() for r in range(nranks)]
    host_w_shards = [make_rank_inputs(r)[1].share_memory_() for r in range(nranks)]
    host_partial = [torch.zeros(M, N, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    host_y = [torch.zeros(M, N, dtype=torch.float32).share_memory_() for _ in range(nranks)]

    cfgs = [
        ChipBootstrapConfig(
            comm=ChipCommBootstrapConfig(
                rank=rank,
                nranks=nranks,
                rootinfo_path=rootinfo_path,
                window_size=window_size,
            ),
            buffers=[
                ChipBufferSpec(
                    name="scratch",
                    dtype="float32",
                    count=scratch_count,
                    nbytes=scratch_nbytes,
                ),
            ],
        )
        for rank in range(nranks)
    ]

    print("[ffn_tp_parallel] compiling kernels...")
    ffn_local_cc = build_ffn_local_callable("a2a3")
    allreduce_cc = build_allreduce_sum_callable("a2a3")

    worker = Worker(
        level=3,
        platform="a2a3",
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
        chip_bootstrap_configs=cfgs,
    )
    ffn_cid = worker.register(ffn_local_cc)
    allreduce_cid = worker.register(allreduce_cc)

    try:
        print("[ffn_tp_parallel] init worker (forks chip children + bootstraps HCCL)...")
        worker.init()

        contexts: list[ChipContext] = worker.chip_contexts
        assert len(contexts) == nranks
        for i, ctx in enumerate(contexts):
            print(
                f"[ffn_tp_parallel] chip {i}: device={ctx.device_id} rank={ctx.rank}/{ctx.nranks} "
                f"window=[0x{ctx.local_window_base:x} +{ctx.actual_window_size}B] "
                f"scratch=0x{ctx.buffer_ptrs['scratch']:x}"
            )

        def orch_fn(orch, _args, cfg):
            for i, ctx in enumerate(contexts):
                # Stage 1: AIC matmul. partial_local is OUTPUT_EXISTING here;
                # the framework records its buffer.addr as a producer.
                a1 = TaskArgs()
                a1.add_tensor(make_tensor_arg(host_x_shards[i]), TensorArgType.INPUT)
                a1.add_tensor(make_tensor_arg(host_w_shards[i]), TensorArgType.INPUT)
                a1.add_tensor(make_tensor_arg(host_partial[i]), TensorArgType.OUTPUT_EXISTING)
                orch.submit_next_level(ffn_cid, a1, cfg, worker=i)

                # Stage 2: AIV cross-rank sum. Tagging partial_local INPUT
                # with the same buffer.addr makes TensorMap auto-link this
                # task as a consumer of stage 1, no explicit barrier needed.
                a2 = TaskArgs()
                a2.add_tensor(make_tensor_arg(host_partial[i]), TensorArgType.INPUT)
                a2.add_tensor(make_tensor_arg(host_y[i]), TensorArgType.OUTPUT_EXISTING)
                a2.add_tensor(
                    ContinuousTensor.make(
                        data=ctx.buffer_ptrs["scratch"],
                        shapes=(scratch_count,),
                        dtype=DataType.FLOAT32,
                        child_memory=True,
                    ),
                    TensorArgType.INOUT,
                )
                a2.add_scalar(ctx.nranks)
                a2.add_scalar(ctx.device_ctx)
                orch.submit_next_level(allreduce_cid, a2, cfg, worker=i)

        print("[ffn_tp_parallel] running 2-chip 2-stage DAG...")
        worker.run(orch_fn, args=None, config=CallConfig())

        # Golden: every rank's y should equal sum over r of x_shard[r] @ w_shard[r].
        expected = torch.zeros(M, N, dtype=torch.float32)
        for r in range(nranks):
            x, w = make_rank_inputs(r)
            expected += x @ w

        # Match scene_test's _compare_outputs: torch.allclose(rtol, atol),
        # which evaluates |a-e| <= atol + rtol*|e|. #522's golden.py uses
        # rtol=atol=1e-4.
        rtol, atol = 1e-4, 1e-4
        ok = True
        for i in range(nranks):
            diff = torch.abs(host_y[i] - expected)
            rel = diff / torch.clamp(torch.abs(expected), min=1e-12)
            print(f"[ffn_tp_parallel] chip {i}: max|y-exp|={float(diff.max()):.3e} max_rel={float(rel.max()):.3e}")
            if not torch.allclose(host_y[i], expected, rtol=rtol, atol=atol):
                ok = False
                for j in range(min(4, M * N)):
                    flat_y = host_y[i].flatten()
                    flat_e = expected.flatten()
                    print(f"  y[{j}]={float(flat_y[j])!r} expected={float(flat_e[j])!r}")

        if not ok:
            print("[ffn_tp_parallel] golden check FAILED")
            return 1
        print("[ffn_tp_parallel] all ranks matched golden ✅")
        return 0
    finally:
        worker.close()
        try:
            os.unlink(rootinfo_path)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--device", default="0-1", help="Device range, e.g. '0-1'. Two chips required.")
    cli = parser.parse_args()
    return run(parse_device_range(cli.device))


if __name__ == "__main__":
    sys.exit(main())
