#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Batch CI test runner using ChipWorker for efficient device reuse.

Replaces ci.sh by running all test tasks (sim + HW) in a single Python process
per device, reusing ChipWorker across tasks that share the same runtime.

Usage:
    python tools/ci.py -p a2a3 -d 5-8 --parallel -c 6622890 -t 600
    python tools/ci.py -p a2a3sim -r tensormap_and_ringbuffer -c 6622890 -t 600
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup — mirrors run_example.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "examples" / "scripts"
PYTHON_DIR = PROJECT_ROOT / "python"
GOLDEN_DIR = PROJECT_ROOT / "golden"

for d in (PYTHON_DIR, SCRIPTS_DIR, GOLDEN_DIR):
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

logger = logging.getLogger("ci")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

EXAMPLES_DIR = PROJECT_ROOT / "examples"
DEVICE_TESTS_DIR = PROJECT_ROOT / "tests" / "st"
MAX_RETRIES = 3


@dataclass
class TaskSpec:
    name: str
    task_dir: Path
    kernels_dir: Path
    golden_path: Path
    platform: str
    runtime_name: str


@dataclass
class CompiledTask:
    spec: TaskSpec
    chip_callable: object  # ChipCallable
    cases: list[dict]
    runtime_bins: object  # RuntimeBinaries
    golden_module: object
    kernel_config: object
    rtol: float = 1e-5
    atol: float = 1e-5
    output_names: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    name: str
    platform: str
    passed: bool
    device: str
    attempt: int
    elapsed_s: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Module loading helpers (from code_runner.py)
# ---------------------------------------------------------------------------


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------


def _discover_runtimes_for_platform(platform: str) -> list[str]:
    from platform_info import discover_runtimes, parse_platform

    arch, _ = parse_platform(platform)
    return discover_runtimes(arch)


def discover_tasks(platform: str, runtime_filter: Optional[str] = None) -> list[TaskSpec]:
    """Scan examples/ and tests/st/ for test directories matching the given platform."""
    from platform_info import parse_platform

    arch, variant = parse_platform(platform)
    is_sim = variant == "sim"
    supported_runtimes = set(_discover_runtimes_for_platform(platform))

    if runtime_filter:
        if runtime_filter not in supported_runtimes:
            raise ValueError(
                f"Runtime '{runtime_filter}' not available for '{platform}'. "
                f"Available: {sorted(supported_runtimes)}"
            )
        supported_runtimes = {runtime_filter}

    tasks: list[TaskSpec] = []

    search_dirs = [EXAMPLES_DIR]
    if not is_sim:
        search_dirs.append(DEVICE_TESTS_DIR)

    for base_dir in search_dirs:
        if not base_dir.is_dir():
            continue
        arch_dir = base_dir / arch
        if not arch_dir.is_dir():
            continue
        for runtime_dir in sorted(arch_dir.iterdir()):
            if not runtime_dir.is_dir():
                continue
            rt_name = runtime_dir.name
            if rt_name not in supported_runtimes:
                continue
            for example_dir in sorted(runtime_dir.iterdir()):
                if not example_dir.is_dir():
                    continue
                kernels_dir = example_dir / "kernels"
                golden_path = example_dir / "golden.py"
                kernel_config_path = kernels_dir / "kernel_config.py"
                if not (kernel_config_path.is_file() and golden_path.is_file()):
                    continue

                rel = example_dir.relative_to(base_dir)
                prefix = "device_test" if base_dir == DEVICE_TESTS_DIR else "example"
                name = f"{prefix}:{rel}"

                tasks.append(
                    TaskSpec(
                        name=name,
                        task_dir=example_dir,
                        kernels_dir=kernels_dir,
                        golden_path=golden_path,
                        platform=platform,
                        runtime_name=rt_name,
                    )
                )

    return tasks


# ---------------------------------------------------------------------------
# PTO-ISA management (reuses code_runner logic)
# ---------------------------------------------------------------------------


def ensure_pto_isa(commit: Optional[str], clone_protocol: str) -> str:
    from code_runner import _ensure_pto_isa_root

    root = _ensure_pto_isa_root(verbose=True, commit=commit, clone_protocol=clone_protocol)
    if root is None:
        raise OSError(
            "PTO_ISA_ROOT could not be resolved.\n"
            "Set it manually or let auto-clone run:\n"
            "  export PTO_ISA_ROOT=$(pwd)/examples/scripts/_deps/pto-isa"
        )
    return root


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_task(
    spec: TaskSpec,
    pto_isa_root: str,
    build_runtime: bool = False,
) -> CompiledTask:
    """Compile orchestration + kernels for a single task, return CompiledTask."""
    from elf_parser import extract_text_section
    from kernel_compiler import KernelCompiler
    from runtime_builder import RuntimeBuilder
    from task_interface import ChipCallable, CoreCallable

    # Load kernel_config and golden
    kc = _load_module(spec.kernels_dir / "kernel_config.py", f"kc_{id(spec)}")
    golden = _load_module(spec.golden_path, f"golden_{id(spec)}")

    kernels = kc.KERNELS
    orchestration = kc.ORCHESTRATION

    builder = RuntimeBuilder(platform=spec.platform)
    compiler = KernelCompiler(platform=spec.platform)

    # Resolve runtime include dirs
    from platform_info import parse_platform

    arch, _ = parse_platform(spec.platform)
    runtime_base = PROJECT_ROOT / "src" / arch / "runtime" / spec.runtime_name
    build_config_path = runtime_base / "build_config.py"
    runtime_include_dirs = []
    if build_config_path.is_file():
        bc = _load_module(build_config_path, f"bc_{id(spec)}")
        aicore_cfg = bc.BUILD_CONFIG.get("aicore", {})
        for p in aicore_cfg.get("include_dirs", []):
            runtime_include_dirs.append(str((runtime_base / p).resolve()))
    else:
        runtime_include_dirs.append(str(runtime_base / "runtime"))
    runtime_include_dirs.append(str(PROJECT_ROOT / "src" / "common" / "task_interface"))

    is_sim = spec.platform.endswith("sim")

    # Compile runtime + orch + kernels in parallel
    def _build_runtime():
        return builder.get_binaries(spec.runtime_name, build=build_runtime)

    def _compile_orch():
        return compiler.compile_orchestration(spec.runtime_name, orchestration["source"])

    def _compile_kernel(kernel):
        incore_o = compiler.compile_incore(
            kernel["source"],
            core_type=kernel["core_type"],
            pto_isa_root=pto_isa_root,
            extra_include_dirs=runtime_include_dirs,
        )
        kernel_bin = incore_o if is_sim else extract_text_section(incore_o)
        sig = kernel.get("signature", [])
        return (kernel["func_id"], CoreCallable.build(signature=sig, binary=kernel_bin))

    max_w = 2 + len(kernels)
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        fut_rt = pool.submit(_build_runtime)
        fut_orch = pool.submit(_compile_orch)
        fut_kernels = [pool.submit(_compile_kernel, k) for k in kernels]

        runtime_bins = fut_rt.result()
        orch_binary = fut_orch.result()
        kernel_binaries = [f.result() for f in fut_kernels]

    orch_sig = orchestration.get("signature", [])
    callable_obj = ChipCallable.build(
        signature=orch_sig,
        func_name=orchestration["function_name"],
        binary=orch_binary,
        children=kernel_binaries,
    )

    all_cases = getattr(golden, "ALL_CASES", {"Default": {}})
    cases = [{"name": name, **params} for name, params in all_cases.items()]

    return CompiledTask(
        spec=spec,
        chip_callable=callable_obj,
        cases=cases,
        runtime_bins=runtime_bins,
        golden_module=golden,
        kernel_config=kc,
        rtol=getattr(golden, "RTOL", 1e-5),
        atol=getattr(golden, "ATOL", 1e-5),
        output_names=getattr(golden, "__outputs__", []),
    )


def compile_all_tasks(
    tasks: list[TaskSpec],
    pto_isa_root: str,
    build_runtime: bool = False,
    max_workers: int = 4,
) -> list[CompiledTask]:
    """Compile all tasks in parallel. Returns list in same order as input."""
    compiled: list[Optional[CompiledTask]] = [None] * len(tasks)
    errors: list[tuple[int, Exception]] = []
    lock = Lock()

    def _do(idx: int):
        try:
            result = compile_task(tasks[idx], pto_isa_root, build_runtime)
            with lock:
                compiled[idx] = result
        except Exception as e:
            with lock:
                errors.append((idx, e))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_do, range(len(tasks))))

    if errors:
        for idx, e in errors:
            logger.error(f"Failed to compile {tasks[idx].name}: {e}")
        raise RuntimeError(f"{len(errors)} task(s) failed to compile")

    return compiled  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Single task execution
# ---------------------------------------------------------------------------


def run_single_task(
    task: CompiledTask,
    worker,
    device_id: int,
) -> bool:
    """Run all cases in a compiled task on a given worker. Returns True if all pass."""
    import torch

    from code_runner import _kernel_config_runtime_env, _temporary_env
    from task_interface import CallConfig, ChipStorageTaskArgs, make_tensor_arg, scalar_to_uint64

    import ctypes
    import numpy as np

    golden_mod = task.golden_module
    kc = task.kernel_config
    runtime_config = getattr(kc, "RUNTIME_CONFIG", {})

    run_env = _kernel_config_runtime_env(kc, task.spec.kernels_dir)

    for params in task.cases:
        result = golden_mod.generate_inputs(params)

        if isinstance(result, list):
            # New-style: flat argument list
            orch_args = ChipStorageTaskArgs()
            args = {}
            inputs = {}
            outputs = {}
            output_set = set(task.output_names)

            for item in result:
                name, value = item
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    tensor = torch.as_tensor(value).cpu().contiguous() if not isinstance(value, torch.Tensor) else value.cpu().contiguous()
                    args[name] = tensor
                    orch_args.add_tensor(make_tensor_arg(tensor))
                    if name in output_set:
                        outputs[name] = tensor
                    else:
                        inputs[name] = tensor
                elif isinstance(value, ctypes._SimpleCData):
                    orch_args.add_scalar(scalar_to_uint64(value))
                    args[name] = value.value
                else:
                    raise TypeError(f"Unsupported arg type for '{name}': {type(value)}")
        else:
            raise TypeError("Legacy dict-style generate_inputs not supported in ci.py; use list-style")

        # Compute golden
        golden_outputs = {k: v.clone() for k, v in outputs.items()}
        golden_with_inputs = {**inputs, **golden_outputs}
        golden_mod.compute_golden(golden_with_inputs, params)

        # Run on device
        config = CallConfig()
        config.block_dim = runtime_config.get("block_dim", 24)
        config.aicpu_thread_num = runtime_config.get("aicpu_thread_num", 3)
        config.orch_thread_num = runtime_config.get("orch_thread_num", 1)

        with _temporary_env(run_env):
            worker.run(task.chip_callable, orch_args, config)

        # Compare
        for name in outputs:
            actual = outputs[name].cpu()
            expected = golden_outputs[name].cpu()
            if not torch.allclose(actual, expected, rtol=task.rtol, atol=task.atol):
                close_mask = torch.isclose(actual, expected, rtol=task.rtol, atol=task.atol)
                mismatches = (~close_mask).sum().item()
                total = actual.numel()
                raise AssertionError(
                    f"Output '{name}' mismatch in case '{params.get('name', '?')}': "
                    f"{mismatches}/{total} elements differ (rtol={task.rtol}, atol={task.atol})"
                )

    return True


# ---------------------------------------------------------------------------
# Group tasks by runtime for ChipWorker reuse
# ---------------------------------------------------------------------------


def group_by_runtime(tasks: list[CompiledTask]) -> dict[str, list[CompiledTask]]:
    groups: dict[str, list[CompiledTask]] = {}
    for t in tasks:
        groups.setdefault(t.spec.runtime_name, []).append(t)
    return groups


# ---------------------------------------------------------------------------
# Device worker
# ---------------------------------------------------------------------------


def device_worker(
    device_id: int,
    task_queue: Queue,
    results: list,
    results_lock: Lock,
    quarantined: set,
    quarantine_lock: Lock,
):
    """Worker thread: pull tasks from queue, run them, handle retries."""
    from task_interface import ChipWorker

    while True:
        try:
            item = task_queue.get_nowait()
        except Empty:
            break

        runtime_name, compiled_tasks, attempt = item
        rt_bins = compiled_tasks[0].runtime_bins

        # Init worker for this runtime group
        worker = ChipWorker()
        try:
            worker.init(
                device_id,
                str(rt_bins.host_path),
                rt_bins.aicpu_path.read_bytes(),
                rt_bins.aicore_path.read_bytes(),
            )
        except Exception as e:
            logger.error(f"[dev{device_id}] Failed to init ChipWorker for {runtime_name}: {e}")
            for ct in compiled_tasks:
                with results_lock:
                    results.append(
                        TaskResult(
                            name=ct.spec.name,
                            platform=ct.spec.platform,
                            passed=False,
                            device=str(device_id),
                            attempt=attempt,
                            elapsed_s=0,
                            error=str(e),
                        )
                    )
            with quarantine_lock:
                quarantined.add(device_id)
            task_queue.task_done()
            break

        failed_tasks = []
        for ct in compiled_tasks:
            start = time.monotonic()
            logger.info(f"[dev{device_id}] Running: {ct.spec.name} (attempt {attempt})")
            try:
                run_single_task(ct, worker, device_id)
                elapsed = time.monotonic() - start
                logger.info(f"[dev{device_id}] PASS: {ct.spec.name} ({elapsed:.1f}s)")
                with results_lock:
                    results.append(
                        TaskResult(
                            name=ct.spec.name,
                            platform=ct.spec.platform,
                            passed=True,
                            device=str(device_id),
                            attempt=attempt,
                            elapsed_s=elapsed,
                        )
                    )
            except Exception as e:
                elapsed = time.monotonic() - start
                logger.error(f"[dev{device_id}] FAIL: {ct.spec.name} ({elapsed:.1f}s): {e}")
                with results_lock:
                    results.append(
                        TaskResult(
                            name=ct.spec.name,
                            platform=ct.spec.platform,
                            passed=False,
                            device=str(device_id),
                            attempt=attempt,
                            elapsed_s=elapsed,
                            error=str(e),
                        )
                    )
                failed_tasks.append(ct)

        worker.reset()

        # Re-enqueue failed tasks for retry (individually, not as a group)
        if failed_tasks and attempt + 1 < MAX_RETRIES:
            for ct in failed_tasks:
                task_queue.put((ct.spec.runtime_name, [ct], attempt + 1))
        elif failed_tasks and attempt + 1 >= MAX_RETRIES:
            logger.warning(f"[dev{device_id}] Quarantined after exhausting retries")
            with quarantine_lock:
                quarantined.add(device_id)
            task_queue.task_done()
            break

        task_queue.task_done()


# ---------------------------------------------------------------------------
# Orchestrators: sim and HW
# ---------------------------------------------------------------------------


def run_sim_tasks(compiled: list[CompiledTask], parallel: bool = False) -> list[TaskResult]:
    """Run simulation tasks with ChipWorker reuse per runtime group."""
    from task_interface import ChipWorker

    groups = group_by_runtime(compiled)
    results: list[TaskResult] = []
    lock = Lock()

    def _run_group(runtime_name: str, group_tasks: list[CompiledTask]):
        worker = ChipWorker()
        rt_bins = group_tasks[0].runtime_bins
        try:
            worker.init(0, str(rt_bins.host_path), rt_bins.aicpu_path.read_bytes(), rt_bins.aicore_path.read_bytes())
        except Exception as e:
            logger.error(f"[sim] Failed to init ChipWorker for {runtime_name}: {e}")
            with lock:
                results.extend(
                    TaskResult(
                        name=ct.spec.name,
                        platform=ct.spec.platform,
                        passed=False,
                        device="sim",
                        attempt=0,
                        elapsed_s=0,
                        error=str(e),
                    )
                    for ct in group_tasks
                )
            return

        try:
            for ct in group_tasks:
                start = time.monotonic()
                try:
                    run_single_task(ct, worker, 0)
                    elapsed = time.monotonic() - start
                    logger.info(f"[sim] PASS: {ct.spec.name} ({elapsed:.1f}s)")
                    r = TaskResult(
                        name=ct.spec.name,
                        platform=ct.spec.platform,
                        passed=True,
                        device="sim",
                        attempt=0,
                        elapsed_s=elapsed,
                    )
                except Exception as e:
                    elapsed = time.monotonic() - start
                    logger.error(f"[sim] FAIL: {ct.spec.name} ({elapsed:.1f}s): {e}")
                    r = TaskResult(
                        name=ct.spec.name,
                        platform=ct.spec.platform,
                        passed=False,
                        device="sim",
                        attempt=0,
                        elapsed_s=elapsed,
                        error=str(e),
                    )
                with lock:
                    results.append(r)
        finally:
            worker.reset()

    if parallel:
        threads = [Thread(target=_run_group, args=(rt_name, tasks)) for rt_name, tasks in groups.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for rt_name, tasks in groups.items():
            _run_group(rt_name, tasks)

    return results


def run_hw_tasks(
    compiled: list[CompiledTask],
    devices: list[int],
    is_a5: bool = False,
) -> list[TaskResult]:
    """Run hardware tasks across devices with ChipWorker reuse per runtime group."""
    groups = group_by_runtime(compiled)

    task_queue: Queue = Queue()
    for rt_name, tasks in groups.items():
        task_queue.put((rt_name, tasks, 0))

    results: list[TaskResult] = []
    results_lock = Lock()
    quarantined: set[int] = set()
    quarantine_lock = Lock()

    threads = []
    for dev_id in devices:
        t = Thread(
            target=device_worker,
            args=(dev_id, task_queue, results, results_lock, quarantined, quarantine_lock),
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return results


def run_hw_tasks_a5(
    compiled: list[CompiledTask],
    devices: list[int],
    args: argparse.Namespace,
) -> list[TaskResult]:
    """Run HW tasks on A5 — wraps each device worker in npu-lock."""
    # Build device-worker sub-command args
    base_args = [
        sys.executable, str(Path(__file__).resolve()),
        "--device-worker",
        "-p", args.platform,
        "--clone-protocol", args.clone_protocol,
    ]
    if args.pto_isa_commit:
        base_args += ["-c", args.pto_isa_commit]

    results: list[TaskResult] = []
    lock = Lock()

    def _run_device(dev_id: int):
        cmd = base_args + ["-d", str(dev_id)]
        is_root = os.getuid() == 0
        has_task_submit = shutil.which("task-submit") is not None

        if is_root:
            full_cmd = ["npu-lock", str(dev_id), "--"] + cmd
        elif has_task_submit:
            inner = f"npu-lock {dev_id} -- " + " ".join(cmd)
            task_id = subprocess.check_output(["task-submit", inner], text=True).strip()
            full_cmd = ["task-submit", "--timeout", str(args.timeout), "--wait", task_id]
        else:
            full_cmd = cmd

        logger.info(f"[a5:dev{dev_id}] Launching: {' '.join(full_cmd)}")
        proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=args.timeout)
        # Parse results from stdout (simplified — rely on exit code)
        passed = proc.returncode == 0
        if not passed:
            logger.error(f"[a5:dev{dev_id}] Failed:\n{proc.stdout}\n{proc.stderr}")
        with lock:
            results.append(
                TaskResult(
                    name=f"a5-device-{dev_id}",
                    platform=args.platform,
                    passed=passed,
                    device=str(dev_id),
                    attempt=0,
                    elapsed_s=0,
                    error=proc.stderr if not passed else None,
                )
            )

    threads = [Thread(target=_run_device, args=(d,)) for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: list[TaskResult]) -> int:
    """Print results table. Returns exit code (0 = all pass, 1 = failures)."""
    # Deduplicate: keep last result per task name (retries produce multiple entries)
    final: dict[str, TaskResult] = {}
    for r in results:
        final[r.name] = r

    ordered = list(final.values())
    pass_count = sum(1 for r in ordered if r.passed)
    fail_count = sum(1 for r in ordered if not r.passed)
    total = len(ordered)

    is_tty = sys.stdout.isatty()
    red = "\033[31m" if is_tty else ""
    green = "\033[32m" if is_tty else ""
    reset = "\033[0m" if is_tty else ""

    # Column widths
    name_w = max((len(r.name) for r in ordered), default=40)
    name_w = max(40, min(72, name_w))

    border = "=" * (name_w + 40)

    # Print failure details first
    for r in ordered:
        if not r.passed and r.error:
            print(f"\n--- FAIL: {r.name} (dev{r.device}, attempt {r.attempt + 1}) ---")
            print(r.error)
            print("--- END ---")

    print(f"\n{border}")
    print(f"{'CI RESULTS SUMMARY':^{len(border)}}")
    print(border)
    print(f"{'TASK':<{name_w}} {'PLATFORM':<10} {'DEVICE':<8} {'ATTEMPT':<8} {'TIME':<8} RESULT")
    print(f"{'-' * name_w} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} ------")

    for r in ordered:
        name_display = r.name[:name_w - 3] + "..." if len(r.name) > name_w else r.name
        status_str = f"{green}PASS{reset}" if r.passed else f"{red}FAIL{reset}"
        print(
            f"{name_display:<{name_w}} {r.platform:<10} {r.device:<8} "
            f"{r.attempt + 1:<8} {r.elapsed_s:.0f}s{'':<5} {status_str}"
        )

    print(border)
    print(f"Total: {total}  Passed: {pass_count}  Failed: {fail_count}")
    print(border)

    if fail_count == 0:
        print("All tests passed!")
        return 0
    return 1


# ---------------------------------------------------------------------------
# PTO-ISA pin on failure (two-pass)
# ---------------------------------------------------------------------------


def reset_pto_isa(commit: str, clone_protocol: str) -> str:
    """Checkout PTO-ISA at the pinned commit (or re-clone if needed)."""
    from code_runner import _checkout_pto_isa_commit, _get_pto_isa_clone_path

    clone_path = _get_pto_isa_clone_path()
    if clone_path.exists():
        _checkout_pto_isa_commit(clone_path, commit, verbose=True)
        return str(clone_path.resolve())
    return ensure_pto_isa(commit, clone_protocol)


# ---------------------------------------------------------------------------
# Device-worker sub-command (for A5 npu-lock wrapping)
# ---------------------------------------------------------------------------


def device_worker_main(args: argparse.Namespace) -> int:
    """Entry point when invoked as --device-worker. Runs all tasks on one device."""
    device_id = args.devices[0] if args.devices else 0
    platform = args.platform

    pto_isa_root = ensure_pto_isa(args.pto_isa_commit, args.clone_protocol)

    tasks = discover_tasks(platform, runtime_filter=args.runtime)
    if not tasks:
        logger.info("No tasks found")
        return 0

    logger.info(f"Compiling {len(tasks)} tasks...")
    compiled = compile_all_tasks(tasks, pto_isa_root)

    groups = group_by_runtime(compiled)
    all_results: list[TaskResult] = []

    from task_interface import ChipWorker

    for rt_name, group_tasks in groups.items():
        rt_bins = group_tasks[0].runtime_bins
        worker = ChipWorker()
        worker.init(
            device_id,
            str(rt_bins.host_path),
            rt_bins.aicpu_path.read_bytes(),
            rt_bins.aicore_path.read_bytes(),
        )

        for ct in group_tasks:
            start = time.monotonic()
            try:
                run_single_task(ct, worker, device_id)
                elapsed = time.monotonic() - start
                logger.info(f"[dev{device_id}] PASS: {ct.spec.name} ({elapsed:.1f}s)")
                all_results.append(
                    TaskResult(
                        name=ct.spec.name, platform=platform, passed=True,
                        device=str(device_id), attempt=0, elapsed_s=elapsed,
                    )
                )
            except Exception as e:
                elapsed = time.monotonic() - start
                logger.error(f"[dev{device_id}] FAIL: {ct.spec.name} ({elapsed:.1f}s): {e}")
                all_results.append(
                    TaskResult(
                        name=ct.spec.name, platform=platform, passed=False,
                        device=str(device_id), attempt=0, elapsed_s=elapsed, error=str(e),
                    )
                )

        worker.reset()

    return print_summary(all_results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch CI test runner with ChipWorker reuse")
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", dest="device_range", default="0")
    parser.add_argument("-r", "--runtime", default=None)
    parser.add_argument(
        "--build-runtime",
        action="store_true",
        help="Rebuild runtime binaries from src/ instead of using pre-built build/lib artifacts",
    )
    parser.add_argument("-c", "--pto-isa-commit", default=None)
    parser.add_argument("-t", "--timeout", type=int, default=600)
    parser.add_argument("--clone-protocol", choices=["ssh", "https"], default="ssh")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--device-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_device_range(device_range: str) -> list[int]:
    if "-" in device_range:
        start, end = device_range.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(device_range)]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    args = parse_args()
    args.devices = parse_device_range(args.device_range)
    is_sim = args.platform.endswith("sim")
    is_a5 = args.platform in ("a5", "a5sim")

    # Device-worker sub-command (for A5 npu-lock wrapping)
    if args.device_worker:
        return device_worker_main(args)

    # Watchdog timer
    watchdog_fired = False

    def _watchdog_handler(signum, frame):
        nonlocal watchdog_fired
        watchdog_fired = True
        print(f"\n{'=' * 40}")
        print(f"[CI] TIMEOUT: exceeded {args.timeout}s ({args.timeout // 60}min) limit, aborting")
        print(f"{'=' * 40}")
        sys.exit(1)

    signal.signal(signal.SIGALRM, _watchdog_handler)
    signal.alarm(args.timeout)

    # Step 1: Ensure PTO-ISA (latest first)
    pto_isa_root = ensure_pto_isa(commit=None, clone_protocol=args.clone_protocol)

    # Step 2: Discover tasks
    tasks = discover_tasks(args.platform, runtime_filter=args.runtime)
    if not tasks:
        logger.info("No tasks found")
        return 0
    logger.info(f"Discovered {len(tasks)} tasks")

    # Step 3 & 4: Compile and run
    logger.info("Compiling all tasks...")
    compiled = compile_all_tasks(tasks, pto_isa_root, build_runtime=args.build_runtime)
    logger.info(f"Compiled {len(compiled)} tasks")

    if is_sim:
        all_results = run_sim_tasks(compiled, parallel=args.parallel)
    elif is_a5:
        all_results = run_hw_tasks_a5(compiled, args.devices, args)
    else:
        all_results = run_hw_tasks(compiled, args.devices)

    # Step 5: PTO-ISA pinned retry for failures
    failures = [r for r in all_results if not r.passed]
    if failures and args.pto_isa_commit:
        failed_names = {r.name for r in failures}
        logger.info(f"[CI] {len(failures)} failure(s), retrying with pinned PTO-ISA {args.pto_isa_commit}")
        pto_isa_root = reset_pto_isa(args.pto_isa_commit, args.clone_protocol)

        failed_specs = [ct.spec for ct in compiled if ct.spec.name in failed_names]
        retry_compiled = compile_all_tasks(failed_specs, pto_isa_root, build_runtime=args.build_runtime)

        if is_sim:
            retry_results = run_sim_tasks(retry_compiled, parallel=args.parallel)
        else:
            if is_a5:
                retry_results = run_hw_tasks_a5(retry_compiled, args.devices, args)
            else:
                retry_results = run_hw_tasks(retry_compiled, args.devices)

        all_results.extend(retry_results)

    # Step 6: Summary
    signal.alarm(0)
    return print_summary(all_results)


if __name__ == "__main__":
    sys.exit(main())
