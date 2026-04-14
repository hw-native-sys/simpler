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
    python ci.py                                                    # all sim platforms
    python ci.py -p a2a3sim -r tensormap_and_ringbuffer -c 6622890  # single platform
    python ci.py -p a2a3 -d 5-8 -c 6622890 -t 600                  # hardware with devices
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# macOS libomp collision workaround — MUST run before any import that may
# transitively load numpy or torch.  See docs/macos-libomp-collision.md for
# the full analysis.
#
# On macOS with a --system-site-packages venv, homebrew's numpy pulls in
# /opt/homebrew/opt/libomp/lib/libomp.dylib (via openblas), while pip's
# torch ships its own .venv/.../torch/lib/libomp.dylib under a different
# install name (/opt/llvm-openmp/lib/libomp.dylib).  Because the two
# dylibs have distinct install names, dyld loads them both, and Intel's
# libomp aborts the process with "OMP: Error #15 ... libomp already
# initialized" (SIGABRT).
#
# The officially-documented escape hatch is KMP_DUPLICATE_LIB_OK=TRUE.
# For our CI workload (numpy random + torch golden compute, no heavy
# parallel OMP regions) the two runtimes never actually race, so allowing
# the duplicate load is safe in practice.
# ---------------------------------------------------------------------------
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import importlib.util
import json
import logging
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Callable, Protocol, cast

PROJECT_ROOT = Path(__file__).resolve().parent

from simpler.task_interface import (  # noqa: E402  # type: ignore[import-not-found]
    ChipCallable,  # pyright: ignore[reportAttributeAccessIssue]
    ChipCallConfig,  # pyright: ignore[reportAttributeAccessIssue]
    ChipStorageTaskArgs,  # pyright: ignore[reportAttributeAccessIssue]
    ChipWorker,  # pyright: ignore[reportAttributeAccessIssue]
    CoreCallable,  # pyright: ignore[reportAttributeAccessIssue]
    make_tensor_arg,
    scalar_to_uint64,
)

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


class BinaryArtifactPathLike(Protocol):
    def read_bytes(self) -> bytes: ...

    def __str__(self) -> str: ...


class RuntimeBinariesLike(Protocol):
    host_path: BinaryArtifactPathLike
    aicpu_path: BinaryArtifactPathLike
    aicore_path: BinaryArtifactPathLike
    sim_context_path: Any


class GoldenModuleLike(Protocol):
    def generate_inputs(self, params: dict[str, Any]) -> object: ...

    def compute_golden(self, tensors: dict[str, Any], params: dict[str, Any]) -> None: ...


@dataclass
class CompiledTask:
    spec: TaskSpec
    chip_callable: Any  # ChipCallable
    cases: list[dict[str, Any]]
    runtime_bins: Any
    golden_module: Any
    kernel_config: Any
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
    error: str | None = None


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


def _write_results_json(results: list[TaskResult], output_path: str | None) -> None:
    if output_path is None:
        return
    Path(output_path).write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")


def _read_results_json(result_path: Path) -> list[TaskResult]:
    if not result_path.is_file():
        return []
    raw = result_path.read_text().strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring invalid result JSON from %s", result_path)
        return []
    return [TaskResult(**item) for item in payload]


def _write_task_list_json(tasks: list[TaskSpec], output_path: str | None) -> None:
    if output_path is None:
        return
    Path(output_path).write_text(json.dumps([task.name for task in tasks], indent=2) + "\n")


def _read_task_list_json(task_list_path: str | None) -> set[str] | None:
    if task_list_path is None:
        return None
    path = Path(task_list_path)
    if not path.is_file():
        return None
    return set(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------


def _discover_runtimes_for_platform(platform: str) -> list[str]:
    from simpler_setup.platform_info import discover_runtimes, parse_platform  # noqa: PLC0415

    arch, _ = parse_platform(platform)
    return discover_runtimes(arch)


def discover_tasks(platform: str, runtime_filter: str | None = None) -> list[TaskSpec]:
    """Scan examples/ and tests/st/ for test directories matching the given platform."""
    from simpler_setup.platform_info import parse_platform  # noqa: PLC0415

    arch, variant = parse_platform(platform)
    is_sim = variant == "sim"
    supported_runtimes = set(_discover_runtimes_for_platform(platform))

    if runtime_filter:
        if runtime_filter not in supported_runtimes:
            raise ValueError(
                f"Runtime '{runtime_filter}' not available for '{platform}'. Available: {sorted(supported_runtimes)}"
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


def ensure_pto_isa(commit: str | None, clone_protocol: str) -> str:
    from simpler_setup.code_runner import _ensure_pto_isa_root  # noqa: PLC0415

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
    run_all_cases: bool = False,
) -> CompiledTask:
    """Compile orchestration + kernels for a single task, return CompiledTask."""
    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

    # Load kernel_config and golden
    kc = _load_module(spec.kernels_dir / "kernel_config.py", f"kc_{id(spec)}")
    golden = _load_module(spec.golden_path, f"golden_{id(spec)}")

    kernels = kc.KERNELS
    orchestration = kc.ORCHESTRATION

    builder = RuntimeBuilder(platform=spec.platform)
    compiler = KernelCompiler(platform=spec.platform)

    # Resolve runtime include dirs
    from simpler_setup.platform_info import parse_platform  # noqa: PLC0415

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
        config_name=orchestration.get("config_name", ""),
    )

    all_cases = getattr(golden, "ALL_CASES", {"Default": {}})
    if run_all_cases:
        cases = [{"name": name, **params} for name, params in all_cases.items()]
    else:
        default_case = getattr(golden, "DEFAULT_CASE", "Default")
        cases = [{"name": default_case, **all_cases[default_case]}]

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
    run_all_cases: bool = False,
    max_workers: int = 4,
) -> list[CompiledTask]:
    """Compile all tasks in parallel. Returns list in same order as input."""
    compiled: list[CompiledTask | None] = [None] * len(tasks)
    errors: list[tuple[int, Exception]] = []
    lock = Lock()

    def _do(idx: int):
        try:
            result = compile_task(tasks[idx], pto_isa_root, build_runtime, run_all_cases)
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

    return cast(list[CompiledTask], compiled)


# ---------------------------------------------------------------------------
# Single task execution
# ---------------------------------------------------------------------------


def run_single_task(
    task: CompiledTask,
    worker,
    device_id: int,
) -> bool:
    """Run all cases in a compiled task on a given worker. Returns True if all pass."""
    import ctypes  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from simpler_setup.code_runner import _kernel_config_runtime_env, _temporary_env  # noqa: PLC0415

    golden_mod = cast(GoldenModuleLike, task.golden_module)
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
                    tensor = (
                        torch.as_tensor(value).cpu().contiguous()
                        if not isinstance(value, torch.Tensor)
                        else value.cpu().contiguous()
                    )
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
        config = ChipCallConfig()
        config.block_dim = runtime_config.get("block_dim", 24)
        config.aicpu_thread_num = runtime_config.get("aicpu_thread_num", 3)

        with _temporary_env(run_env):
            worker.run(task.chip_callable, orch_args, config)

        # Compare
        for name, actual_tensor in outputs.items():
            actual = actual_tensor.cpu()
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
    while True:
        try:
            item = task_queue.get_nowait()
        except Empty:
            break

        runtime_name, compiled_tasks, attempt = item
        rt_bins = cast(RuntimeBinariesLike, compiled_tasks[0].runtime_bins)

        # Init worker for this runtime group
        worker = ChipWorker()
        try:
            worker.init(
                str(rt_bins.host_path),
                str(rt_bins.aicpu_path),
                str(rt_bins.aicore_path),
                sim_context_lib_path=str(rt_bins.sim_context_path) if rt_bins.sim_context_path else "",
            )
            worker.set_device(device_id)
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

        worker.reset_device()
        worker.finalize()

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


def run_hw_tasks(
    compiled: list[CompiledTask],
    devices: list[int],
) -> list[TaskResult]:
    """Run hardware tasks in-process with ChipWorker reuse per runtime group."""
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

    if quarantined:
        logger.warning("[hw] Quarantined devices: %s", sorted(quarantined))

    return results


def _build_device_worker_base_args(args: argparse.Namespace) -> list[str]:
    base_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--device-worker",
        "-p",
        args.platform,
        "--clone-protocol",
        args.clone_protocol,
    ]
    if args.runtime:
        base_args += ["-r", args.runtime]
    if args.build_runtime:
        base_args.append("--build-runtime")
    if args.run_all_cases:
        base_args.append("--all")
    return base_args


def _run_device_worker_subprocess(
    tasks: list[TaskSpec],
    device_id: int,
    args: argparse.Namespace,
    tag: str,
    pto_isa_commit: str | None = None,
    print_log_on_fail: bool = False,
    quiet: bool = True,
    timeout: int | None = None,
) -> list[TaskResult]:
    """Run a task batch in one device-worker subprocess and return its reported results.

    When *quiet* is False, stdout streams to the terminal in real time
    (useful for serial sim runs).  When True, output is captured and only
    shown on failure if *print_log_on_fail* is set.
    """
    base_args = _build_device_worker_base_args(args)
    if pto_isa_commit:
        base_args += ["-c", pto_isa_commit]

    with tempfile.NamedTemporaryFile(
        prefix=f"ci_{tag}_tasks_dev{device_id}_",
        suffix=".json",
        delete=False,
    ) as task_file:
        task_list_path = Path(task_file.name)

    with tempfile.NamedTemporaryFile(
        prefix=f"ci_{tag}_dev{device_id}_",
        suffix=".json",
        delete=False,
    ) as result_file:
        result_path = Path(result_file.name)

    _write_task_list_json(tasks, str(task_list_path))
    full_cmd = base_args + [
        "-d",
        str(device_id),
        "--task-list-json",
        str(task_list_path),
        "--result-json",
        str(result_path),
    ]

    logger.info(f"[{tag}:dev{device_id}] Launching: {' '.join(full_cmd)}")
    try:
        if quiet:
            proc = subprocess.run(full_cmd, check=False, capture_output=True, text=True, timeout=timeout)
        else:
            proc = subprocess.run(
                full_cmd, check=False, stdout=None, stderr=subprocess.PIPE, text=True, timeout=timeout
            )
        device_results = _read_results_json(result_path)
        if proc.returncode != 0:
            if print_log_on_fail and quiet:
                logger.error(f"[{tag}:dev{device_id}] Failed:\n{proc.stdout}\n{proc.stderr}")
            elif print_log_on_fail and proc.stderr:
                logger.error(f"[{tag}:dev{device_id}] stderr:\n{proc.stderr}")
        # When the subprocess crashes without reporting per-task failures,
        # generate FAIL results for every task that has no result yet so
        # that pin-retry can match them by name.
        if proc.returncode != 0 and not any(not r.passed for r in device_results):
            reported_names = {r.name for r in device_results}
            error_msg = (proc.stderr or proc.stdout or f"Device worker exited with code {proc.returncode}").strip()
            for t in tasks:
                if t.name not in reported_names:
                    device_results.append(
                        TaskResult(
                            name=t.name,
                            platform=t.platform,
                            passed=False,
                            device=str(device_id),
                            attempt=0,
                            elapsed_s=0,
                            error=error_msg,
                        )
                    )
        return device_results
    except subprocess.TimeoutExpired:
        logger.error(f"[{tag}:dev{device_id}] Subprocess timed out after {timeout}s")
        device_results = _read_results_json(result_path)
        reported_names = {r.name for r in device_results}
        for t in tasks:
            if t.name not in reported_names:
                device_results.append(
                    TaskResult(
                        name=t.name,
                        platform=t.platform,
                        passed=False,
                        device=str(device_id),
                        attempt=0,
                        elapsed_s=0,
                        error=f"Timed out after {timeout}s",
                    )
                )
        return device_results
    finally:
        task_list_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def _normalize_task_result(
    task: TaskSpec,
    device_id: int,
    attempt: int,
    task_results: list[TaskResult],
) -> TaskResult:
    matching = [result for result in task_results if result.name == task.name]
    source = matching[-1] if matching else task_results[-1]
    return TaskResult(
        name=task.name,
        platform=task.platform,
        passed=source.passed,
        device=str(device_id),
        attempt=attempt,
        elapsed_s=source.elapsed_s,
        error=source.error,
    )


def run_hw_tasks_subprocess(
    tasks: list[TaskSpec],
    devices: list[int],
    args: argparse.Namespace,
    pto_isa_commit: str | None = None,
) -> list[TaskResult]:
    """Run hardware tasks: one subprocess per task.

    On any failure the device is immediately quarantined (worker exits). Healthy
    devices keep pulling from the shared queue. Tasks that were never run or failed
    are collected so the caller can re-run them in a pin-commit pass with all devices
    refreshed.
    """
    task_queue: Queue[tuple[TaskSpec, int]] = Queue()
    total = len(tasks)
    for task in tasks:
        task_queue.put((task, 0))

    results: list[TaskResult] = []
    results_lock = Lock()
    completed = [0]  # mutable counter for thread-safe increment
    quarantined: set[int] = set()
    quarantine_lock = Lock()
    tag = "hw"

    is_pin_retry = pto_isa_commit is not None

    def _run_device(dev_id: int):
        while True:
            try:
                task, attempt = task_queue.get_nowait()
            except Empty:
                return

            is_last_attempt = attempt + 1 >= MAX_RETRIES
            task_results = _run_device_worker_subprocess(
                [task],
                dev_id,
                args,
                tag=tag,
                pto_isa_commit=pto_isa_commit,
                print_log_on_fail=is_pin_retry and is_last_attempt,
            )
            normalized = _normalize_task_result(task, dev_id, attempt, task_results)
            with results_lock:
                results.append(normalized)
                if normalized.passed or is_last_attempt:
                    completed[0] += 1
                n = completed[0]
            status = "PASS" if normalized.passed else "FAIL"
            attempt_info = f" attempt {attempt + 1}" if attempt > 0 else ""
            logger.info(
                f"[{tag}:dev{dev_id}] [{n}/{total}] {status}: {task.name}{attempt_info} ({normalized.elapsed_s:.1f}s)"
            )

            if normalized.passed:
                continue

            # Failure: re-enqueue with attempt+1 if under limit, quarantine this device
            if not is_last_attempt:
                task_queue.put((task, attempt + 1))
            logger.warning(f"[{tag}:dev{dev_id}] Quarantined after failure on {task.name}")
            with quarantine_lock:
                quarantined.add(dev_id)
            return

    threads = [Thread(target=_run_device, args=(device_id,)) for device_id in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Tasks stranded in queue — all devices quarantined before queue emptied
    while True:
        try:
            task, attempt = task_queue.get_nowait()
        except Empty:
            break
        results.append(
            TaskResult(
                name=task.name,
                platform=task.platform,
                passed=False,
                device="N/A",
                attempt=attempt,
                elapsed_s=0,
                error="All devices quarantined",
            )
        )

    if quarantined:
        logger.warning(f"[{tag}] Quarantined devices: {sorted(quarantined)}")

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
        name_display = r.name[: name_w - 3] + "..." if len(r.name) > name_w else r.name
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
    from simpler_setup.code_runner import _checkout_pto_isa_commit, _get_pto_isa_clone_path  # noqa: PLC0415

    clone_path = _get_pto_isa_clone_path()
    if clone_path.exists():
        _checkout_pto_isa_commit(clone_path, commit, verbose=True)
        return str(clone_path.resolve())
    return ensure_pto_isa(commit, clone_protocol)


# ---------------------------------------------------------------------------
# Device-worker sub-command
# ---------------------------------------------------------------------------


def device_worker_main(args: argparse.Namespace) -> int:
    """Entry point when invoked as --device-worker. Runs all tasks on one device."""
    device_id = args.devices[0] if args.devices else 0
    platform = args.platform

    pto_isa_root = ensure_pto_isa(args.pto_isa_commit, args.clone_protocol)

    tasks = discover_tasks(platform, runtime_filter=args.runtime)
    selected_names = _read_task_list_json(args.task_list_json)
    if selected_names is not None:
        tasks = [task for task in tasks if task.name in selected_names]
    if not tasks:
        logger.info("No tasks found")
        return 0

    all_results = _run_tasks_on_device(tasks, device_id, platform, pto_isa_root, args)
    _write_results_json(all_results, args.result_json)
    return print_summary(all_results)


def _run_tasks_on_device(
    tasks: list[TaskSpec],
    device_id: int,
    platform: str,
    pto_isa_root: str,
    args: argparse.Namespace,
) -> list[TaskResult]:
    """Compile and run all tasks on a single device. Returns all TaskResults.

    For simulation platforms with sufficient CPUs, tasks are distributed
    across multiple virtual device IDs and executed in parallel threads.
    ChipWorker.run() internally uses std::thread + join, so GIL is released
    during execution, enabling true parallelism.
    """
    logger.info(f"Compiling {len(tasks)} tasks...")
    try:
        compiled = compile_all_tasks(
            tasks, pto_isa_root, build_runtime=args.build_runtime, run_all_cases=args.run_all_cases
        )
    except RuntimeError:
        return [
            TaskResult(
                name=t.name,
                platform=platform,
                passed=False,
                device=str(device_id),
                attempt=0,
                elapsed_s=0,
                error="compile failed",
            )
            for t in tasks
        ]

    is_sim = platform.endswith("sim")
    if is_sim:
        cpu_count = os.cpu_count() or 1
        max_workers = min(max(cpu_count // 20, 1), len(compiled))
    else:
        max_workers = 1

    if max_workers <= 1:
        return _run_compiled_tasks(compiled, device_id, platform)

    # Parallel: distribute tasks round-robin across virtual device IDs
    buckets: list[list[CompiledTask]] = [[] for _ in range(max_workers)]
    for i, ct in enumerate(compiled):
        buckets[i % max_workers].append(ct)

    logger.info(f"[sim] Parallel execution: {max_workers} workers, {len(compiled)} tasks")

    results: list[TaskResult] = []
    results_lock = Lock()
    completed_count = [0]
    total = len(compiled)

    def _worker(worker_id: int, worker_tasks: list[CompiledTask]):
        dev_id = worker_id
        worker_results = _run_compiled_tasks(worker_tasks, dev_id, platform)
        with results_lock:
            for r in worker_results:
                completed_count[0] += 1
                n = completed_count[0]
                results.append(r)
                status = "PASS" if r.passed else "FAIL"
                logger.info(f"[dev{dev_id}] [{n}/{total}] {status}: {r.name} ({r.elapsed_s:.1f}s)")

    threads = []
    for i in range(max_workers):
        if not buckets[i]:
            continue
        t = Thread(target=_worker, args=(i, buckets[i]))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return results


def _run_compiled_tasks(
    compiled: list[CompiledTask],
    device_id: int,
    platform: str,
) -> list[TaskResult]:
    """Run compiled tasks serially on a single device."""

    groups = group_by_runtime(compiled)
    all_results: list[TaskResult] = []

    for rt_name, group_tasks in groups.items():
        rt_bins = cast(RuntimeBinariesLike, group_tasks[0].runtime_bins)
        worker = ChipWorker()
        try:
            worker.init(
                str(rt_bins.host_path),
                str(rt_bins.aicpu_path),
                str(rt_bins.aicore_path),
                sim_context_lib_path=str(rt_bins.sim_context_path) if rt_bins.sim_context_path else "",
            )
            worker.set_device(device_id)
        except Exception as e:
            logger.error(f"[dev{device_id}] Failed to init ChipWorker for {rt_name}: {e}")
            all_results.extend(
                TaskResult(
                    name=ct.spec.name,
                    platform=platform,
                    passed=False,
                    device=str(device_id),
                    attempt=0,
                    elapsed_s=0,
                    error=str(e),
                )
                for ct in group_tasks
            )
            continue

        for ct in group_tasks:
            start = time.monotonic()
            try:
                run_single_task(ct, worker, device_id)
                elapsed = time.monotonic() - start
                logger.info(f"[dev{device_id}] PASS: {ct.spec.name} ({elapsed:.1f}s)")
                all_results.append(
                    TaskResult(
                        name=ct.spec.name,
                        platform=platform,
                        passed=True,
                        device=str(device_id),
                        attempt=0,
                        elapsed_s=elapsed,
                    )
                )
            except Exception as e:
                elapsed = time.monotonic() - start
                logger.error(f"[dev{device_id}] FAIL: {ct.spec.name} ({elapsed:.1f}s): {e}")
                all_results.append(
                    TaskResult(
                        name=ct.spec.name,
                        platform=platform,
                        passed=False,
                        device=str(device_id),
                        attempt=0,
                        elapsed_s=elapsed,
                        error=str(e),
                    )
                )

        worker.reset_device()
        worker.finalize()

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _discover_valid_platforms() -> list[str]:
    """Discover valid platforms from src/ directory structure (mirrors ci.sh logic)."""
    platforms = []
    src_dir = PROJECT_ROOT / "src"
    if not src_dir.is_dir():
        return platforms
    for arch_dir in sorted(src_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name
        platform_dir = arch_dir / "platform"
        if (platform_dir / "onboard").is_dir():
            platforms.append(arch)
        if (platform_dir / "sim").is_dir():
            platforms.append(f"{arch}sim")
    return platforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch CI test runner with ChipWorker reuse")
    parser.add_argument("-p", "--platform", default=None)
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
    parser.add_argument("--all", dest="run_all_cases", action="store_true", help="Run all cases, not just DEFAULT_CASE")
    parser.add_argument("--device-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-list-json", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_device_range(device_range: str) -> list[int]:
    if "-" in device_range:
        start, end = device_range.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(device_range)]


def _run_with_timeout(
    phase_name: str,
    timeout_s: int,
    runner: Callable[[], list[TaskResult]],
) -> list[TaskResult]:
    def _watchdog_handler(signum, frame):
        print(f"\n{'=' * 40}", flush=True)
        print(
            f"[CI] TIMEOUT: {phase_name} exceeded {timeout_s}s ({timeout_s // 60}min) limit, aborting",
            flush=True,
        )
        print(f"{'=' * 40}", flush=True)
        os._exit(1)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _watchdog_handler)
    signal.alarm(timeout_s)
    try:
        return runner()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run_single_platform(platform: str, args: argparse.Namespace) -> list[TaskResult]:
    """Run all tasks for a single platform. Returns list of TaskResults."""
    is_sim = platform.endswith("sim")

    tasks = discover_tasks(platform, runtime_filter=args.runtime)
    if not tasks:
        logger.info(f"[{platform}] No tasks found")
        return []
    logger.info(f"[{platform}] Discovered {len(tasks)} tasks")

    # Compile and run via subprocess isolation.
    # Sim: single subprocess with all tasks (ChipWorker reuse + parallel within).
    # HW: one subprocess per task with device-level quarantine.
    sub_args = argparse.Namespace(**vars(args))
    sub_args.platform = platform
    if is_sim:
        all_results = _run_device_worker_subprocess(tasks, 0, sub_args, tag="sim", timeout=args.timeout, quiet=False)
    else:
        all_results = _run_with_timeout(
            f"{platform} initial pass",
            args.timeout,
            lambda: run_hw_tasks_subprocess(tasks, args.devices, sub_args),
        )

    # Pin retry — re-run failed tasks with pinned PTO-ISA commit.
    final: dict[str, TaskResult] = {}
    for r in all_results:
        final[r.name] = r
    failures = [r for r in final.values() if not r.passed]

    if failures and args.pto_isa_commit:
        failed_names = {r.name for r in failures}
        failed_tasks = [t for t in tasks if t.name in failed_names]
        logger.info(f"[{platform}] {len(failed_tasks)} failure(s), retrying with pinned PTO-ISA {args.pto_isa_commit}")
        if is_sim:
            pin_results = _run_device_worker_subprocess(
                failed_tasks,
                0,
                sub_args,
                tag="sim",
                pto_isa_commit=args.pto_isa_commit,
                print_log_on_fail=True,
                quiet=False,
                timeout=args.timeout,
            )
        else:
            pin_results = _run_with_timeout(
                f"{platform} pin retry",
                args.timeout,
                lambda: run_hw_tasks_subprocess(
                    failed_tasks,
                    args.devices,
                    sub_args,
                    pto_isa_commit=args.pto_isa_commit,
                ),
            )
        all_results.extend(pin_results)

    return all_results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)

    args = parse_args()
    args.devices = parse_device_range(args.device_range)

    valid_platforms = _discover_valid_platforms()

    # Device-worker sub-command (always needs explicit -p)
    if args.device_worker:
        if not args.platform:
            print("--device-worker requires -p/--platform")
            return 1
        return device_worker_main(args)

    # Determine which platforms to run
    if args.platform:
        if args.platform not in valid_platforms:
            print(f"Unknown platform: {args.platform}")
            print(f"Valid platforms: {' '.join(valid_platforms)}")
            return 1
        platforms = [args.platform]
    else:
        # No -p: run all sim platforms
        platforms = [p for p in valid_platforms if p.endswith("sim")]
        if not platforms:
            print("No sim platforms found")
            return 1
        logger.info(f"No platform specified, running all sim platforms: {', '.join(platforms)}")

    all_results: list[TaskResult] = []
    for platform in platforms:
        all_results.extend(_run_single_platform(platform, args))

    if not all_results:
        logger.info("No tasks found")
        return 0

    return print_summary(all_results)


if __name__ == "__main__":
    sys.exit(main())
