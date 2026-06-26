#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime overhead benchmark for the fully_distributed_within_core runtime.

Goal: isolate the cost of *on-core orchestration + claim race + scheduling*
(everything the distributed runtime does instead of an AICPU scheduler) from
the cost of the kernels themselves, and see how that cost scales with the
number of physical blocks (cores).

Method:
  * Reuse the ``benchmark_bgemm`` workload (same orchestration + GEMM/ADD
    incores) — referenced directly, not duplicated.
  * Set ``PTO_DIST_SKIP_EXEC=1`` so the engine skips every incore kernel call
    and treats each (sub)task as 0-cost, while keeping all ownership/completion
    bookkeeping. The wall clock then reflects orchestration/scheduling only.
  * Sweep ``block_dim`` (1 block = 1 AIC + 2 AIV) and report the program wall
    clock for each, so the relative overhead across core counts is visible.

a2a3sim caps ``block_dim`` at PLATFORM_MAX_BLOCKDIM = 24 (24 AIC + 48 AIV = 72
cores); 48 *blocks* is not representable (that 48 is the AIV-core count at the
24-block max). The default sweep is the full ramp 1..24 (``--blocks 1-24``);
pass an explicit list/range to narrow it (e.g. ``--blocks 1,2,12,24``).

Run (standalone driver produces the comparison table)::

    python test_runtime_overhead.py -p a2a3sim
    python test_runtime_overhead.py -p a2a3sim --blocks 1,12,24 --rounds 5 --tasks 480
    python test_runtime_overhead.py -p a2a3sim --exec        # include kernel work (baseline)
    python test_runtime_overhead.py -p a2a3sim --bind node:0,1   # pin sim threads to NUMA nodes 0,1
    python test_runtime_overhead.py -p a2a3sim --bind cpu:0-79    # pin to an explicit CPU range
    # Confine the AICore working set to ONE NUMA node (1:1 thread->cpu) while
    # auxiliary threads ride the wider --bind set; needs cores=block*3 <= node size:
    python test_runtime_overhead.py -p a2a3sim --blocks 1-13 --bind node:1,2,3 --aicore-numa 2

The class is also a valid SceneTestCase (cases marked manual), so the workload
can be golden-checked the normal way with kernels enabled::

    python test_runtime_overhead.py -p a2a3sim --case Blk24 --manual only
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

# The bgemm incore/orchestration sources live in the sibling example; reference
# them so this benchmark exercises exactly that workload without duplication.
_BGEMM = "../benchmark_bgemm/kernels"


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestRuntimeOverhead(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": f"{_BGEMM}/orchestration/bgemm_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "GEMM",
                "source": f"{_BGEMM}/aic/kernel_gemm_tile.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ADD",
                "source": f"{_BGEMM}/aiv/kernel_tile_add.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN],
            },
        ],
    }

    # Cases for the normal (golden-checked, kernels-on) pytest path. All manual
    # so the benchmark never slows the default suite; the headline artifact is
    # the standalone comparison table below.
    _BENCH_PARAMS = {"matmul_add_task_num": 1000, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2}
    CASES = [
        {
            "name": "Blk1",
            "manual": True,
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": _BENCH_PARAMS,
        },
        {
            "name": "Blk2",
            "manual": True,
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 2},
            "params": _BENCH_PARAMS,
        },
        {
            "name": "Blk12",
            "manual": True,
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 12},
            "params": _BENCH_PARAMS,
        },
        {
            "name": "Blk24",
            "manual": True,
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": _BENCH_PARAMS,
        },
    ]

    def generate_args(self, params):
        tile_size = params["incore_data_size"]
        incore_loop = params["incore_loop"]
        grid_k = params["grid_k"]
        num_groups = params["matmul_add_task_num"] // grid_k
        A = torch.randn(num_groups, grid_k, incore_loop, tile_size, tile_size, dtype=torch.float32) * 0.01
        B = torch.randn(num_groups, grid_k, incore_loop, tile_size, tile_size, dtype=torch.float32) * 0.01
        C = torch.zeros(incore_loop * num_groups, tile_size, tile_size, dtype=torch.float32)
        config = torch.tensor([tile_size, grid_k, num_groups, incore_loop], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("A", A.flatten()), Tensor("B", B.flatten()), Tensor("C", C.flatten()), Tensor("config", config)
        )

    def compute_golden(self, args, params):
        tile_size = params["incore_data_size"]
        incore_loop = params["incore_loop"]
        grid_k = params["grid_k"]
        num_groups = params["matmul_add_task_num"] // grid_k
        A = args.A.reshape(num_groups, grid_k, incore_loop, tile_size, tile_size)
        B = args.B.reshape(num_groups, grid_k, incore_loop, tile_size, tile_size)
        C = args.C.reshape(incore_loop * num_groups, tile_size, tile_size)
        C[:] = 0.0
        for group in range(num_groups):
            for k_idx in range(grid_k):
                for i in range(incore_loop):
                    C[group * incore_loop + i] += torch.matmul(A[group, k_idx, i], B[group, k_idx, i])


# ---------------------------------------------------------------------------
# CPU-affinity (core-binding) control.
#
# The sim runs every AICore/AICPU "core" as a host std::thread; those threads
# inherit the launching process's CPU-affinity mask, so binding the Python
# process here pins the whole simulation without external numactl/taskset (and
# without numactl --membind, whose memory pinning starved allocations and added
# noise). Threads are created lazily inside worker.run(), so applying the mask
# before the first run is sufficient for all of them.
# ---------------------------------------------------------------------------


def _parse_cpu_list(spec):
    """Parse a CPU list like '0-3,8,10-12' into a set of ints."""
    cpus = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = (int(v) for v in tok.split("-", 1))
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(tok))
    return cpus


def _node_cpus(nodes_spec):
    """Union the online CPUs of the given NUMA node(s), e.g. '0,1'."""
    cpus = set()
    for node in _parse_cpu_list(nodes_spec):
        path = f"/sys/devices/system/node/node{node}/cpulist"
        with open(path) as f:  # noqa: PTH123
            cpus |= _parse_cpu_list(f.read().strip())
    return cpus


def _apply_cpu_binding(bind):
    """Apply a core-binding strategy to this process; return the bound cpu set.

    Strategies (``--bind``):
      * ``none``                : no pinning (sim threads float over all CPUs).
      * ``node:<nodes>``        : pin to all CPUs of the given NUMA node(s),
                                  e.g. ``node:0`` or ``node:0,1``.
      * ``cpu:<list>`` / ``<list>`` : pin to an explicit CPU list/range,
                                  e.g. ``cpu:0-79`` or ``0,1,2``.
    """
    import os  # noqa: PLC0415

    spec = (bind or "none").strip()
    online = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None

    if spec.lower() in ("", "none"):
        cpus = online
    elif spec.lower().startswith("node:"):
        cpus = _node_cpus(spec[len("node:") :])
    elif spec.lower().startswith("cpu:"):
        cpus = _parse_cpu_list(spec[len("cpu:") :])
    else:
        cpus = _parse_cpu_list(spec)

    if spec.lower() not in ("", "none") and cpus:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("os.sched_setaffinity unavailable on this platform")
        os.sched_setaffinity(0, cpus)
        cpus = os.sched_getaffinity(0)  # echo back what the OS actually accepted

    n = len(cpus) if cpus else 0
    print(f"CPU binding: strategy='{spec}' -> {n} physical cores" + (f" {sorted(cpus)}" if cpus and n <= 32 else ""))
    return cpus


# ---------------------------------------------------------------------------
# Standalone comparison driver: sweep block_dim, print a wall-clock table.
# ---------------------------------------------------------------------------


def _bench(platform, block_dims, params, rounds, skip_exec, warmup, device):
    """Run the workload once per block_dim and return per-config timings (us)."""
    import os  # noqa: PLC0415
    import statistics  # noqa: PLC0415
    import time  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from simpler_setup.scene_test import _build_chip_task_args, _resolve_callable_paths  # noqa: PLC0415

    # Engine reads PTO_DIST_SKIP_EXEC at dist_engine_register (once per run()).
    if skip_exec:
        os.environ["PTO_DIST_SKIP_EXEC"] = "1"
    else:
        os.environ.pop("PTO_DIST_SKIP_EXEC", None)

    # The standalone path skips scene_test's per-class setup, so resolve the
    # (relative) bgemm kernel sources against this file's directory ourselves.
    _resolve_callable_paths(TestRuntimeOverhead, Path(__file__).parent)

    inst = TestRuntimeOverhead()
    orch_sig = TestRuntimeOverhead.CALLABLE["orchestration"]["signature"]

    worker = TestRuntimeOverhead._create_worker(platform, device)
    results = []
    try:
        callable_obj = inst.build_callable(platform)
        handle = worker.register(callable_obj)

        for bd in block_dims:
            cfg = inst._build_config({"aicpu_thread_num": 4, "block_dim": bd})

            # Build args/chip_args once per block_dim (data content is irrelevant
            # to orchestration/scheduling timing, and skip-exec never reads it).
            # Hoisting it out of the timed loop keeps large --tasks sweeps fast:
            # otherwise every round re-runs torch.randn over multi-GB tensors.
            args = inst.generate_args(params)
            chip_args, _ = _build_chip_task_args(args, orch_sig)

            def _one_run():
                t0 = time.perf_counter()
                timing = worker.run(handle, chip_args, config=cfg)
                wall_us = (time.perf_counter() - t0) * 1e6
                dev_us = float(getattr(timing, "device_wall_us", 0.0) or 0.0)
                host_us = float(getattr(timing, "host_wall_us", 0.0) or 0.0)
                return wall_us, host_us, dev_us

            for _ in range(warmup):
                _one_run()
            samples = [_one_run() for _ in range(rounds)]
            wall = statistics.median(s[0] for s in samples)
            host = statistics.median(s[1] for s in samples)
            dev = statistics.median(s[2] for s in samples)
            results.append((bd, wall, host, dev))
            print(f"  block_dim={bd:>2} ({bd * 3:>3} cores): wall={wall / 1000:8.3f} ms  device={dev / 1000:8.3f} ms")
    finally:
        worker.close()
    return results


def _print_table(results, params, rounds, skip_exec, bind_spec="none", bind_ncores=0):
    task_num = params["matmul_add_task_num"]
    # bgemm submits one GEMM (1C) and one ADD (1V) per matmul-add unit.
    total_tasks = task_num * 2
    # The on-device orchestrator wall is the metric of interest: it is the pure
    # on-core orchestration + claim race + scheduling cost. The host wall is
    # dominated by fixed Python/sim-launch overhead and is shown only for context.
    base_dev = results[0][3] if results else 0.0
    mode = "skip-exec (orchestration/scheduling only)" if skip_exec else "with kernels"
    # Echo the active core-binding so the table is self-describing (the bound
    # physical-core count is the key axis when comparing pinned vs unpinned runs
    # and over/under-subscription effects — see docs §6.3).
    bind_str = f"unpinned ({bind_ncores} cores available)" if bind_spec in ("", "none") else (
        f"{bind_ncores} physical cores (strategy='{bind_spec}')"
    )
    print()
    print(f"Runtime overhead — fully_distributed_within_core [{mode}]")
    print(f"workload=bgemm  matmul_add_task_num={task_num}  (~{total_tasks} tasks)  rounds={rounds} (median)")
    print(f"cpu_bind={bind_str}")
    print()
    header = (
        f"| {'blocks':>6} | {'cores':>5} | {'device (ms)':>11} | {'us/task':>8} "
        f"| {'dev vs 1blk':>11} | {'host (ms)':>10} |"
    )
    sep = "|" + "-" * 8 + "|" + "-" * 7 + "|" + "-" * 13 + "|" + "-" * 10 + "|" + "-" * 13 + "|" + "-" * 12 + "|"
    print(header)
    print(sep)
    for bd, wall, _host, dev in results:
        ratio = (dev / base_dev) if base_dev > 0 else 0.0
        us_task = dev / total_tasks if total_tasks else 0.0
        print(
            f"| {bd:>6} | {bd * 3:>5} | {dev / 1000:>11.3f} | {us_task:>8.2f} "
            f"| {ratio:>10.2f}× | {wall / 1000:>10.3f} |"
        )
    print()
    print("device (ms) = on-core orchestration + claim race + scheduling wall (PTO2 profiling).")
    print("host (ms)   = Python wall incl. fixed sim-launch overhead (context only).")


def main():
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(description="fully_distributed_within_core runtime-overhead benchmark")
    p.add_argument("-p", "--platform", required=True)
    p.add_argument("-d", "--device", type=int, default=0)
    p.add_argument(
        "--blocks",
        default="1-24",
        help="block_dim values: comma list and/or a-b ranges, e.g. '1,2,12,24' or '1-24' (a2a3sim max 24)",
    )
    p.add_argument("--rounds", type=int, default=5, help="timed rounds per config (median reported)")
    p.add_argument("--warmup", type=int, default=1, help="untimed warmup rounds per config")
    p.add_argument("--tasks", type=int, default=1000, help="matmul_add_task_num (total tasks = 2x; batch)")
    p.add_argument("--data-size", type=int, default=128, help="incore tile shape (NxN)")
    p.add_argument("--loop", type=int, default=4)
    p.add_argument("--grid-k", type=int, default=2)
    p.add_argument("--exec", action="store_true", help="actually run kernels (default: skip for overhead isolation)")
    p.add_argument(
        "--bind",
        default="none",
        help="CPU core-binding strategy: 'none' | 'node:<nodes>' (e.g. node:0,1) | "
        "'cpu:<list>' or bare '<list>' (e.g. cpu:0-79). Pins all sim threads via "
        "sched_setaffinity, no external numactl needed.",
    )
    p.add_argument(
        "--aicore-numa",
        type=int,
        default=None,
        help="Pin every AICore sim thread 1:1 into this single NUMA node (sets "
        "PTO_SIM_AICORE_NUMA_NODE), keeping the AICore working set inside one node. "
        "Use a node with >= cores (=block_dim*3) CPUs; combine with --bind on a few "
        "idle nodes so auxiliary threads don't oversubscribe the AICore node.",
    )
    args = p.parse_args()

    import os  # noqa: PLC0415

    bound_cpus = _apply_cpu_binding(args.bind)
    bind_ncores = len(bound_cpus) if bound_cpus else 0

    if args.aicore_numa is not None:
        os.environ["PTO_SIM_AICORE_NUMA_NODE"] = str(args.aicore_numa)
        node_cpus = sorted(_node_cpus(str(args.aicore_numa)))
        print(
            f"AICore pinning: every AICore thread -> NUMA node {args.aicore_numa} "
            f"({len(node_cpus)} cpus: {min(node_cpus)}..{max(node_cpus)}), 1:1 exclusive"
        )
    else:
        os.environ.pop("PTO_SIM_AICORE_NUMA_NODE", None)

    block_dims = []
    for tok in args.blocks.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = (int(v) for v in tok.split("-", 1))
            block_dims.extend(range(lo, hi + 1))
        else:
            block_dims.append(int(tok))
    params = {
        "matmul_add_task_num": args.tasks,
        "incore_data_size": args.data_size,
        "incore_loop": args.loop,
        "grid_k": args.grid_k,
    }
    skip_exec = not args.exec
    print(f"Benchmarking block_dims={block_dims} on {args.platform} (skip_exec={skip_exec}) ...")
    results = _bench(args.platform, block_dims, params, args.rounds, skip_exec, args.warmup, args.device)
    _print_table(results, params, args.rounds, skip_exec, args.bind, bind_ncores)


if __name__ == "__main__":
    main()
