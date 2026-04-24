# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SceneTestCase framework — unified scene test infrastructure.

``@scene_test`` decorator + ``SceneTestCase`` base class.
pytest: ``pytest --platform a2a3sim``
standalone: ``python test_xxx.py -p a2a3sim``

A scene test class declares three things:
  CALLABLE: what to compile/register
    L2: orchestration (C++ source) + incores (C++ kernels)
    L3: orchestration (Python DAG fn) + callables (ChipCallable + SubCallable)
  CASES: how to run (per-case platform, config, params)
  generate_args / compute_golden: data + golden comparison
"""

from __future__ import annotations

import gc
import inspect
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from .log_config import DEFAULT_LOG_LEVEL, LOG_LEVEL_CHOICES, configure_logging
from .pto_isa import ensure_pto_isa_root

logger = logging.getLogger(__name__)

_compile_cache: dict[tuple[str, str, str], object] = {}


def clear_compile_cache() -> None:
    """Drop every cached ``ChipCallable`` and force a GC pass.

    The cache keeps nanobind-owned ``ChipCallable`` instances alive for the
    whole pytest session. Module-level dicts are cleared by Python in an
    order that can outlive the nanobind module destructor, which then
    prints ``leaked N instances of type _task_interface.ChipCallable`` to
    stderr at interpreter shutdown. Call this from ``pytest_sessionfinish``
    (and other session-end paths) so the instances die while the nanobind
    module is still wired up.
    """
    _compile_cache.clear()
    gc.collect()


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


class Tensor(NamedTuple):
    """Tensor argument spec."""

    name: str
    value: Any  # torch.Tensor


class Scalar(NamedTuple):
    """Scalar argument spec (ctypes scalar)."""

    name: str
    value: Any  # ctypes.c_float, ctypes.c_int64, etc.


# ---------------------------------------------------------------------------
# TaskArgsBuilder — ordered container with named access
# ---------------------------------------------------------------------------


class TaskArgsBuilder:
    """Test-side task arguments container.

    Maintains insertion order (tensors before scalars) and provides
    attribute access by name for use in compute_golden.

    Usage::

        args = TaskArgsBuilder(
            Tensor("a", torch.full((N,), 2.0)),
            Tensor("b", torch.full((N,), 3.0)),
            Tensor("f", torch.zeros(N)),
            Scalar("scale", ctypes.c_float(1.5)),
        )
        args.a  # → tensor
        args.f[:] = args.a + args.b  # in compute_golden
    """

    def __init__(self, *specs):
        self._specs: list = []
        self._data: dict[str, Any] = {}
        self._has_scalar = False
        for spec in specs:
            if isinstance(spec, Tensor):
                self._add_tensor(spec)
            elif isinstance(spec, Scalar):
                self._add_scalar(spec)

    def add_tensor(self, name: str, value: Any) -> None:
        """Add a tensor. Must be called before any add_scalar."""
        self._add_tensor(Tensor(name, value))

    def add_scalar(self, name: str, value: Any) -> None:
        """Add a scalar. After this, add_tensor is not allowed."""
        self._add_scalar(Scalar(name, value))

    def _add_tensor(self, spec: Tensor) -> None:
        if self._has_scalar:
            raise ValueError("Cannot add tensor after scalar (tensor-before-scalar ordering required)")
        self._specs.append(spec)
        self._data[spec.name] = spec.value

    def _add_scalar(self, spec: Scalar) -> None:
        self._has_scalar = True
        self._specs.append(spec)
        self._data[spec.name] = spec.value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"TaskArgsBuilder has no argument '{name}'") from None

    def clone(self) -> TaskArgsBuilder:
        """Deep clone: all tensors are cloned, scalars copied."""
        import torch  # noqa: PLC0415

        new = TaskArgsBuilder.__new__(TaskArgsBuilder)
        new._specs = []
        new._data = {}
        new._has_scalar = False
        for spec in self._specs:
            if isinstance(spec, Tensor):
                cloned = spec.value.clone() if isinstance(spec.value, torch.Tensor) else spec.value
                new_spec = Tensor(spec.name, cloned)
                new._specs.append(new_spec)
                new._data[spec.name] = cloned
            elif isinstance(spec, Scalar):
                import copy  # noqa: PLC0415

                new._has_scalar = True
                cloned_val = copy.copy(spec.value)
                new_spec = Scalar(spec.name, cloned_val)
                new._specs.append(new_spec)
                new._data[spec.name] = cloned_val
        return new

    @property
    def specs(self) -> list:
        """Ordered list of Tensor/Scalar specs."""
        return self._specs

    def tensor_names(self) -> list[str]:
        """Names of all tensor arguments, in order."""
        return [s.name for s in self._specs if isinstance(s, Tensor)]


# ---------------------------------------------------------------------------
# CallableNamespace — dot-access container for L3 callables
# ---------------------------------------------------------------------------


class CallableNamespace:
    """Dot-access container for compiled/registered callables.

    Used by L3 orch functions to access callables by name::

        callables.vector_kernel       # → ChipCallable object
        callables.vector_kernel_sig   # → signature list
        callables.verify              # → callable_id (int)

    Also provides ``keep()`` for lifetime management: L3 orch functions
    that build transient Python objects (e.g. ChipStorageTaskArgs) whose
    raw pointers are submitted to the C++ scheduler must register them
    via ``keep()`` so they outlive the scheduler drain::

        def run_dag(w, callables, task_args, config):
            chip_args, _ = _build_chip_task_args(task_args, callables.vector_kernel_sig)
            callables.keep(chip_args)  # survive until drain finishes
            ...
    """

    def __init__(self, entries: dict):
        self._entries = dict(entries)
        self._keepalive: list = []

    def __getattr__(self, name: str):
        try:
            return self._entries[name]
        except KeyError:
            raise AttributeError(f"CallableNamespace has no entry '{name}'") from None

    def keep(self, *objs):
        """Register objects to keep alive until this namespace is destroyed."""
        if not objs:
            return None
        self._keepalive.extend(objs)
        return objs[0] if len(objs) == 1 else objs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_chip_task_args(test_args: TaskArgsBuilder, orch_signature: list):
    """Build `ChipStorageTaskArgs` (POD) from `TaskArgsBuilder`.

    Used by the L2 path (`ChipWorker.run(callable, chip_args, config)`): the
    chip worker expects the runtime.so ABI-shaped POD directly (no tags).

    Returns:
        chip_args: ChipStorageTaskArgs (POD)
        output_names: list of tensor names that are OUTPUT or INOUT
    """
    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        ChipStorageTaskArgs,
        scalar_to_uint64,
    )

    from simpler_setup.torch_interop import make_tensor_arg  # noqa: PLC0415

    chip_args = ChipStorageTaskArgs()
    output_names: list[str] = []

    tensor_idx = 0
    for spec in test_args.specs:
        if isinstance(spec, Tensor):
            if tensor_idx >= len(orch_signature):
                raise ValueError(
                    f"Tensor '{spec.name}' at index {tensor_idx} has no matching entry in "
                    f"orchestration signature (length {len(orch_signature)}). "
                    f"Update CALLABLE['orchestration']['signature'] to match generate_args()."
                )
            direction = orch_signature[tensor_idx]
            chip_args.add_tensor(make_tensor_arg(spec.value))
            if direction in (ArgDirection.OUT, ArgDirection.INOUT):
                output_names.append(spec.name)
            tensor_idx += 1
        elif isinstance(spec, Scalar):
            chip_args.add_scalar(scalar_to_uint64(spec.value))

    return chip_args, output_names


def _build_l3_task_args(test_args: TaskArgsBuilder, orch_signature: list):
    """Build a tagged `TaskArgs` (vector-backed, with `TensorArgType` tags) from
    `TaskArgsBuilder`.

    Used by the L3 path (`orch.submit_next_level(callable, args, config)`):
    the orchestrator reads the tags to drive dependency inference.

    Returns:
        chip_args: TaskArgs (tagged)
        output_names: list of tensor names that are OUTPUT or INOUT
    """
    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        TaskArgs,
        TensorArgType,
        scalar_to_uint64,
    )

    from simpler_setup.torch_interop import make_tensor_arg  # noqa: PLC0415

    _DIR_TO_TAG = {
        ArgDirection.IN: TensorArgType.INPUT,
        ArgDirection.OUT: TensorArgType.OUTPUT_EXISTING,
        ArgDirection.INOUT: TensorArgType.INOUT,
    }

    chip_args = TaskArgs()
    output_names: list[str] = []

    tensor_idx = 0
    for spec in test_args.specs:
        if isinstance(spec, Tensor):
            if tensor_idx >= len(orch_signature):
                raise ValueError(
                    f"Tensor '{spec.name}' at index {tensor_idx} has no matching entry in "
                    f"orchestration signature (length {len(orch_signature)}). "
                    f"Update CALLABLE['orchestration']['signature'] to match generate_args()."
                )
            direction = orch_signature[tensor_idx]
            tag = _DIR_TO_TAG.get(direction, TensorArgType.INPUT)
            chip_args.add_tensor(make_tensor_arg(spec.value), tag)
            if direction in (ArgDirection.OUT, ArgDirection.INOUT):
                output_names.append(spec.name)
            tensor_idx += 1
        elif isinstance(spec, Scalar):
            chip_args.add_scalar(scalar_to_uint64(spec.value))

    return chip_args, output_names


@contextmanager
def _temporary_env(env_updates):
    """Temporarily set environment variables."""
    if not env_updates:
        yield
        return
    old = {k: os.environ.get(k) for k in env_updates}
    for k, v in env_updates.items():
        os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _resolve_callable_paths(cls, cls_dir):
    """Resolve relative source paths in CALLABLE against cls_dir."""
    callable_spec = cls.CALLABLE
    if "callables" in callable_spec:
        # L3: resolve inside each ChipCallable entry
        resolved = []
        for entry in callable_spec["callables"]:
            if "orchestration" in entry:
                entry = dict(entry)
                _resolve_chip_entry_paths(entry, cls_dir)
            resolved.append(entry)
        callable_spec["callables"] = resolved
    else:
        # L2: resolve orchestration + incores directly
        _resolve_chip_entry_paths(callable_spec, cls_dir)


def _resolve_chip_entry_paths(entry, cls_dir):
    """Resolve relative source paths in a chip entry (orchestration + incores)."""
    if "orchestration" in entry:
        orch = entry["orchestration"]
        if isinstance(orch, dict) and "source" in orch and not os.path.isabs(orch["source"]):
            entry["orchestration"] = dict(orch)
            entry["orchestration"]["source"] = str(cls_dir / orch["source"])
    if "incores" in entry:
        resolved = []
        for k in entry["incores"]:
            k = dict(k)
            if "source" in k and not os.path.isabs(k["source"]):
                k["source"] = str(cls_dir / k["source"])
            resolved.append(k)
        entry["incores"] = resolved


def _extract_name_map(callable_spec: dict) -> dict:
    """Extract name mapping from a CALLABLE spec.

    Each level exports only its own ``callable_id_to_name`` — the mapping
    from next-level-down IDs to human-readable names.  No cross-level
    nesting: the perf data declares its level, the mapping declares its
    level, and the tool matches them.

    * **L2** — ``callable_id`` = incore ``func_id``::

        {"level": 2, "orchestrator_name": "PagedAttn",
         "callable_id_to_name": {"0": "QK", "1": "SF"}}

    * **L3** — ``callable_id`` = index in ``callables`` list::

        {"level": 3, "orchestrator_name": "run_dag",
         "callable_id_to_name": {"0": "vec_kernel", "1": "verify"}}
    """
    if "callables" not in callable_spec:
        # L2: orchestration + incores
        callable_id_to_name: dict[str, str] = {}
        orch = callable_spec.get("orchestration", {})
        orchestrator_name = orch.get("name") if isinstance(orch, dict) else None
        for k in callable_spec.get("incores", []):
            if "name" in k and "func_id" in k:
                callable_id_to_name[str(k["func_id"])] = k["name"]
        result: dict = {"level": 2, "orchestrator_name": orchestrator_name}
        if callable_id_to_name:
            result["callable_id_to_name"] = callable_id_to_name
        return result

    # L3: Python orch function + callables list
    orch = callable_spec.get("orchestration")
    orchestrator_name = getattr(orch, "__name__", None) if callable(orch) else None

    callable_id_to_name = {}
    for idx, entry in enumerate(callable_spec["callables"]):
        callable_id_to_name[str(idx)] = entry.get("name", f"callable_{idx}")

    result = {"level": 3, "orchestrator_name": orchestrator_name}
    if callable_id_to_name:
        result["callable_id_to_name"] = callable_id_to_name
    return result


def _dump_name_map(mapping: dict, output_path: Path) -> Path | None:
    """Write name mapping to JSON if it contains any names. Returns path or None."""
    import json as _json  # noqa: PLC0415

    if not mapping.get("callable_id_to_name") and not mapping.get("orchestrator_name"):
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        _json.dump(mapping, f, indent=2)
    return output_path


def _parse_case_selector(value: str) -> tuple[str | None, str | None]:
    """Parse one ``--case`` value into ``(class_name, case_name)``.

    ``Foo`` -> ``(None, "Foo")`` (any class)
    ``ClassA::Foo`` -> ``("ClassA", "Foo")``
    ``ClassA::`` -> ``("ClassA", None)`` (all cases in ClassA)
    ``::Foo`` -> ``(None, "Foo")``
    """
    if "::" in value:
        cls_part, case_part = value.split("::", 1)
        return (cls_part or None, case_part or None)
    return (None, value)


def _match_selectors(cls_name: str, case_name: str, selectors: list[tuple]) -> bool:
    """True if ``(cls_name, case_name)`` matches any selector (empty list means no selector filter)."""
    if not selectors:
        return True
    for sel_cls, sel_case in selectors:
        if (sel_cls is None or sel_cls == cls_name) and (sel_case is None or sel_case == case_name):
            return True
    return False


def _select_cases(test_classes, platform: str, selectors: list[tuple], manual_mode: str):
    """Resolve (class, case) pairs to run. Validates selectors strictly.

    Filters: platform match -> selector match -> manual_mode (exclude/include/only).
    Raises ``ValueError`` on unknown selector class/case or empty selection.
    """
    class_index = {c.__name__: c for c in test_classes}
    for sel_cls, _ in selectors:
        if sel_cls is not None and sel_cls not in class_index:
            available = ", ".join(sorted(class_index)) or "(none)"
            raise ValueError(f"--case: unknown class '{sel_cls}'. Available: {available}")
    for sel_cls, sel_case in selectors:
        if sel_case is None:
            continue
        scoped = [class_index[sel_cls]] if sel_cls else test_classes
        if not any(case["name"] == sel_case for c in scoped for case in c.CASES):
            scope = sel_cls or "any class"
            raise ValueError(f"--case: case '{sel_case}' not found in {scope}")

    selected: list[tuple] = []
    for cls in test_classes:
        for case in cls.CASES:
            if platform not in case["platforms"]:
                continue
            if not _match_selectors(cls.__name__, case["name"], selectors):
                continue
            is_manual = bool(case.get("manual"))
            if manual_mode == "exclude" and is_manual:
                continue
            if manual_mode == "only" and not is_manual:
                continue
            selected.append((cls, case))

    if not selected:
        if selectors:
            sel_str = ", ".join(f"{c or '*'}::{n or '*'}" for c, n in selectors)
            hint = " (matches are manual; pass --manual include or only)" if manual_mode == "exclude" else ""
            raise ValueError(f"--case: no cases matched [{sel_str}] for platform={platform}{hint}")
        raise ValueError(f"No cases matched platform={platform} (manual={manual_mode})")
    return selected


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _outputs_dir() -> Path:
    """Return the directory where l2_perf_records_*.json lands.

    Honors ``SIMPLER_L2_PERF_RECORDS_OUTPUT_DIR`` so the parallel test dispatcher can give each
    subprocess its own isolated output directory. Absolute paths pass through;
    relative paths are interpreted against the project root. Empty/unset env
    var falls back to ``<project>/outputs`` (the historical default).
    """
    env = os.environ.get("SIMPLER_L2_PERF_RECORDS_OUTPUT_DIR")
    if env:
        p = Path(env)
        return p if p.is_absolute() else _project_root() / p
    return _project_root() / "outputs"


def _snapshot_l2_perf_records_files() -> set[Path]:
    d = _outputs_dir()
    return set(d.glob("l2_perf_records_*.json")) if d.exists() else set()


def _wait_new_l2_perf_records_file(before: set[Path], timeout: float = 2.0) -> Path | None:
    """Wait briefly for a new ``l2_perf_records_*.json`` to appear in outputs/."""
    import time  # noqa: PLC0415

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        new = _snapshot_l2_perf_records_files() - before
        if new:
            return max(new, key=lambda p: p.stat().st_mtime)
        time.sleep(0.1)
    return None


def _get_device_log_dir(device_id) -> Path:
    """Return CANN device log directory (matches device_log_resolver)."""
    ascend_work_path = os.environ.get("ASCEND_WORK_PATH")
    if ascend_work_path:
        root = Path(ascend_work_path).expanduser() / "log" / "debug"
        if root.exists():
            return root / f"device-{device_id}"
    return Path.home() / "ascend" / "log" / "debug" / f"device-{device_id}"


def _snapshot_device_logs(device_id) -> set[Path]:
    log_dir = _get_device_log_dir(device_id)
    return set(log_dir.glob("*.log")) if log_dir.exists() else set()


def _wait_new_device_log(device_id, before: set[Path], timeout: float = 15.0) -> Path | None:
    """Wait for a new CANN device log; returns the newest new file or None."""
    import time  # noqa: PLC0415

    log_dir = _get_device_log_dir(device_id)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_dir.exists():
            new = set(log_dir.glob("*.log")) - before
            if new:
                return max(new, key=lambda p: p.stat().st_mtime)
        time.sleep(0.5)
    return None


def _run_swimlane_converter(
    input_path: Path | None = None,
    device_id=None,
    device_log: Path | None = None,
    func_names_path: Path | None = None,
) -> None:
    """Invoke ``tools/swimlane_converter.py``.

    When ``input_path`` is given, the converter derives its output filename from
    the input's timestamp (see ``tools/swimlane_converter.py::_resolve_output_path``).
    Without it, the converter auto-selects the latest ``l2_perf_records_*.json``.
    """
    import logging  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    script = _project_root() / "tools" / "swimlane_converter.py"
    if not script.exists():
        logger.warning(f"Swimlane converter script not found: {script}")
        return
    cmd = [sys.executable, str(script)]
    if input_path is not None:
        cmd.append(str(input_path))
    if func_names_path is not None:
        cmd += ["--func-names", str(func_names_path)]
    if device_log is not None:
        cmd += ["--device-log", str(device_log)]
    elif device_id is not None:
        cmd += ["-d", str(device_id)]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        logger.info("Swimlane JSON generation completed")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to generate swimlane JSON: {e}")
        if e.stdout:
            logger.debug(f"stdout: {e.stdout}")
        if e.stderr:
            logger.debug(f"stderr: {e.stderr}")


def _sanitize_for_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def _convert_case_swimlane(
    case_label: str,
    device_id,
    before_l2_perf_records: set[Path],
    before_device: set[Path] | None,
    callable_spec: dict | None = None,
) -> None:
    """Post-case: rename the new perf file to include ``case_label`` (guarding against
    the runtime's second-precision filename collisions), then invoke the converter.

    The ``l2_perf_records_`` prefix is preserved so the converter's stem-based output
    naming still strips it and produces ``merged_swimlane_<ts>_<case_label>.json``.

    When ``callable_spec`` is provided and contains ``"name"`` entries on incores,
    a ``func_id_names_<label>.json`` sidecar is written and passed to the converter
    so that swimlane visualizations display human-readable kernel names.
    """
    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    l2_perf_records_file = _wait_new_l2_perf_records_file(before_l2_perf_records)
    if l2_perf_records_file is None:
        logger.warning(f"[{case_label}] No new l2_perf_records_*.json produced; skipping conversion")
        return
    safe_label = _sanitize_for_filename(case_label)
    suffix = (
        l2_perf_records_file.stem[len("l2_perf_records_") :]
        if l2_perf_records_file.stem.startswith("l2_perf_records_")
        else l2_perf_records_file.stem
    )
    renamed = l2_perf_records_file.with_name(f"l2_perf_records_{suffix}_{safe_label}.json")
    if renamed.exists():
        logger.warning(f"[{case_label}] target {renamed.name} already exists; overwriting")
        renamed.unlink()
    l2_perf_records_file.rename(renamed)

    # Dump callable name mapping if the CALLABLE spec provides names
    func_names_path = None
    if callable_spec:
        mapping = _extract_name_map(callable_spec)
        func_names_path = _dump_name_map(mapping, renamed.parent / f"name_map_{safe_label}.json")

    device_log = None
    if before_device is not None:
        device_log = _wait_new_device_log(device_id, before_device)
        if device_log is None:
            logger.warning(f"[{case_label}] no new device log found; scheduler deep-dive may use stale log")
    _run_swimlane_converter(
        input_path=renamed, device_id=device_id, device_log=device_log, func_names_path=func_names_path
    )


def run_class_cases(  # noqa: PLR0913 -- shared layer-5 entry; kwargs mirror CLI surface
    worker,
    cls_inst,
    cases,
    *,
    callable_obj,
    sub_ids,
    primary_device_id,
    is_hardware,
    rounds,
    skip_golden,
    enable_l2_swimlane,
    enable_dump_tensor,
    enable_pmu,
):
    """Execute a pre-filtered list of cases for one class (layers 5-6).

    Caller is responsible for platform/selector/manual filtering. Profiling
    snapshots wrap each case. Validation failures propagate; caller decides
    fail-fast vs collect semantics.
    """
    cls_name = type(cls_inst).__name__
    callable_spec = getattr(type(cls_inst), "CALLABLE", None)
    for case in cases:
        before_l2_perf_records = _snapshot_l2_perf_records_files() if enable_l2_swimlane else set()
        before_device = _snapshot_device_logs(primary_device_id) if enable_l2_swimlane and is_hardware else None
        try:
            cls_inst._run_and_validate(
                worker,
                callable_obj,
                case,
                sub_ids=sub_ids,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
            )
        finally:
            if enable_l2_swimlane:
                _convert_case_swimlane(
                    f"{cls_name}_{case['name']}",
                    primary_device_id,
                    before_l2_perf_records,
                    before_device,
                    callable_spec=callable_spec,
                )


def _compare_outputs(test_args, golden_args, output_names, rtol, atol):
    """Compare output tensors against golden values."""
    import torch  # noqa: PLC0415

    for name in output_names:
        actual = getattr(test_args, name)
        expected = getattr(golden_args, name)
        if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            diff = (actual - expected).abs().max().item()
            raise AssertionError(f"Golden mismatch on '{name}': max_diff={diff}, rtol={rtol}, atol={atol}")


def _compile_chip_callable_from_spec(spec, platform, runtime, cache_key):
    """Compile a chip entry spec (orchestration + incores) -> ChipCallable. Session-cached."""
    if cache_key in _compile_cache:
        return _compile_cache[cache_key]

    from simpler.task_interface import ChipCallable, CoreCallable  # noqa: PLC0415

    from .elf_parser import extract_text_section  # noqa: PLC0415
    from .kernel_compiler import KernelCompiler  # noqa: PLC0415
    from .pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    orch = spec["orchestration"]
    incores = spec["incores"]

    pto_isa_root = ensure_pto_isa_root()
    kc = KernelCompiler(platform=platform)
    is_sim = platform.endswith("sim")

    orch_binary = kc.compile_orchestration(runtime, orch["source"])
    inc_dirs = kc.get_orchestration_include_dirs(runtime)

    kernel_binaries = []
    for k in incores:
        incore = kc.compile_incore(
            k["source"], core_type=k["core_type"], pto_isa_root=pto_isa_root, extra_include_dirs=inc_dirs
        )
        if not is_sim:
            incore = extract_text_section(incore)
        kernel_binaries.append((k["func_id"], CoreCallable.build(signature=k.get("signature", []), binary=incore)))

    chip_callable = ChipCallable.build(
        signature=orch.get("signature", []),
        func_name=orch["function_name"],
        binary=orch_binary,
        children=kernel_binaries,
        config_name=orch.get("config_name", ""),
    )
    _compile_cache[cache_key] = chip_callable
    return chip_callable


# ---------------------------------------------------------------------------
# @scene_test decorator
# ---------------------------------------------------------------------------


def scene_test(level: int, runtime: str):
    """Decorator marking a SceneTestCase with level and runtime.

    Platforms are declared per-case in CASES, not here.
    """

    def decorator(cls):
        cls._st_level = level
        cls._st_runtime = runtime
        cls_dir = Path(inspect.getfile(cls)).parent
        if hasattr(cls, "CALLABLE"):
            _resolve_callable_paths(cls, cls_dir)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# SceneTestCase base class
# ---------------------------------------------------------------------------


class SceneTestCase:
    """Base class for scene tests at any hierarchy level.

    Subclasses declare CALLABLE, CASES, generate_args(), compute_golden().
    """

    CALLABLE: dict = {}
    CASES: list[dict] = []
    RTOL: float = 1e-5
    ATOL: float = 1e-5
    RUNTIME_ENV: dict = {}

    def generate_args(self, params) -> TaskArgsBuilder:
        """Return TaskArgsBuilder with ordered Tensor/Scalar specs."""
        raise NotImplementedError

    def compute_golden(self, args: TaskArgsBuilder, params) -> None:
        """Compute expected outputs in-place on a cloned TaskArgsBuilder."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Callable compilation
    # ------------------------------------------------------------------

    @classmethod
    def compile_chip_callable(cls, platform):
        """Compile CALLABLE -> ChipCallable (L2). Session-cached."""
        cache_key = (cls.__qualname__, platform, cls._st_runtime)
        return _compile_chip_callable_from_spec(cls.CALLABLE, platform, cls._st_runtime, cache_key)

    @classmethod
    def _compile_l3_callables(cls, platform):
        """Compile all ChipCallable entries in CALLABLE['callables'] (L3)."""
        compiled = {}
        for entry in cls.CALLABLE["callables"]:
            if "orchestration" in entry:
                name = entry["name"]
                cache_key = (cls.__qualname__, name, platform, cls._st_runtime)
                chip = _compile_chip_callable_from_spec(entry, platform, cls._st_runtime, cache_key)
                compiled[name] = chip
                compiled[f"{name}_sig"] = entry["orchestration"].get("signature", [])
        return compiled

    # ------------------------------------------------------------------
    # Worker creation
    # ------------------------------------------------------------------

    @classmethod
    def _get_binaries(cls, platform, build=False):
        from .runtime_builder import RuntimeBuilder  # noqa: PLC0415

        return RuntimeBuilder(platform=platform).get_binaries(cls._st_runtime, build=build)

    @classmethod
    def _create_worker(cls, platform, device_id=0, build=False):
        from simpler.task_interface import ChipWorker  # noqa: PLC0415

        bins = cls._get_binaries(platform, build=build)
        w = ChipWorker()
        w.init(
            str(bins.host_path),
            str(bins.aicpu_path),
            str(bins.aicore_path),
            str(bins.sim_context_path) if bins.sim_context_path else "",
        )
        w.set_device(device_id)
        return w

    # ------------------------------------------------------------------
    # Default build methods
    # ------------------------------------------------------------------

    def build_callable(self, platform):
        """Build callable for the current level.

        L2: returns ChipCallable.
        L3: returns dict of {name: ChipCallable, name_sig: signature}.
        """
        if self._st_level == 2:
            return self.compile_chip_callable(platform)
        elif self._st_level == 3:
            return self._compile_l3_callables(platform)
        raise ValueError(f"Unsupported level: {self._st_level}")

    def _build_config(self, config_dict, enable_l2_swimlane=False, enable_dump_tensor=False, enable_pmu=0):
        from simpler.task_interface import ChipCallConfig  # noqa: PLC0415

        config = ChipCallConfig()
        config.block_dim = config_dict.get("block_dim", 1)
        config.aicpu_thread_num = config_dict.get("aicpu_thread_num", 3)
        config.enable_l2_swimlane = enable_l2_swimlane
        config.enable_dump_tensor = enable_dump_tensor
        config.enable_pmu = enable_pmu  # 0=disabled, >0=enabled with event type
        return config

    def _resolve_env(self):
        env = self.RUNTIME_ENV
        if not env:
            return {}
        cls_dir = Path(inspect.getfile(type(self))).parent
        out = {}
        for k, v in env.items():
            s = str(v)
            if (k.endswith("_DIR") or k.endswith("_PATH")) and not Path(s).is_absolute():
                s = str((cls_dir / s).resolve())
            out[k] = s
        return out

    # ------------------------------------------------------------------
    # Run + validate
    # ------------------------------------------------------------------

    def _run_and_validate(
        self,
        worker,
        callable_obj,
        case,
        sub_ids=None,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=False,
        enable_dump_tensor=False,
        enable_pmu=0,
    ):
        if self._st_level == 2:
            self._run_and_validate_l2(
                worker,
                callable_obj,
                case,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
            )
        elif self._st_level == 3:
            self._run_and_validate_l3(
                worker,
                callable_obj,
                sub_ids or {},
                case,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
            )

    def _run_and_validate_l2(
        self,
        worker,
        callable_obj,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=False,
        enable_dump_tensor=False,
        enable_pmu=0,
    ):
        params = case.get("params", {})
        config_dict = case.get("config", {})
        orch_sig = self.CALLABLE.get("orchestration", {}).get("signature", [])

        # Build args
        test_args = self.generate_args(params)
        chip_args, output_names = _build_chip_task_args(test_args, orch_sig)

        # Compute golden (unless skip_golden)
        golden_args = None
        if not skip_golden:
            golden_args = test_args.clone()
            self.compute_golden(golden_args, params)

        # Save initial output tensor values for reset between rounds
        initial_outputs = {}
        if rounds > 1:
            for name in output_names:
                initial_outputs[name] = getattr(test_args, name).clone()

        # Execute rounds
        for round_idx in range(rounds):
            if round_idx > 0:
                for name, initial in initial_outputs.items():
                    getattr(test_args, name).copy_(initial)

            config = self._build_config(
                config_dict,
                enable_l2_swimlane=(enable_l2_swimlane and round_idx == 0),
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
            )

            with _temporary_env(self._resolve_env()):
                worker.run(callable_obj, chip_args, config=config)

            if not skip_golden:
                _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

    def _run_and_validate_l3(
        self,
        worker,
        compiled_callables,
        sub_ids,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=False,
        enable_dump_tensor=False,
        enable_pmu=0,
    ):
        # Defensive belt-and-braces: the pytest dispatcher and run_module both
        # block --enable-l2-swimlane for L3 at the CLI boundary. Catch any code
        # path that reaches here with the flag on anyway (direct API use,
        # future refactors) so we fail loud rather than produce garbage perf
        # files. Lift once the runtime embeds device_id in the perf filename.
        if enable_l2_swimlane:
            raise NotImplementedError(
                "L3 profiling is not supported yet (multi-chip-process perf "
                "filename collision). Gate at the CLI level in "
                "conftest.pytest_collection_modifyitems / scene_test.run_module."
            )

        params = case.get("params", {})
        config_dict = case.get("config", {})

        # Build args
        test_args = self.generate_args(params)

        # Compute golden (unless skip_golden)
        golden_args = None
        if not skip_golden:
            golden_args = test_args.clone()
            self.compute_golden(golden_args, params)

        # Save initial tensor values for reset between rounds
        all_tensor_names = test_args.tensor_names()
        initial_tensors = {}
        if rounds > 1:
            for name in all_tensor_names:
                initial_tensors[name] = getattr(test_args, name).clone()

        # Build CallableNamespace: compiled ChipCallables + sub callable IDs
        ns = CallableNamespace({**compiled_callables, **sub_ids})

        # Get orch function (plain function from CALLABLE)
        orch_fn = self.CALLABLE["orchestration"]

        # Execute rounds
        for round_idx in range(rounds):
            if round_idx > 0:
                for name, initial in initial_tensors.items():
                    getattr(test_args, name).copy_(initial)

            config = self._build_config(
                config_dict,
                enable_l2_swimlane=(enable_l2_swimlane and round_idx == 0),
                enable_dump_tensor=enable_dump_tensor,
                enable_pmu=enable_pmu,
            )

            # Orch fn signature: (orch, args, cfg) — inner fn forwards to
            # the user's scene orch which takes (orch, callables, task_args, config).
            def task_orch(orch, _args, _cfg, _ns=ns, _test_args=test_args, _config=config):
                orch_fn(orch, _ns, _test_args, _config)

            with _temporary_env(self._resolve_env()):
                worker.run(task_orch)

            if not skip_golden:
                _compare_outputs(test_args, golden_args, all_tensor_names, self.RTOL, self.ATOL)

    # ------------------------------------------------------------------
    # pytest auto test method
    # ------------------------------------------------------------------

    def test_run(self, st_platform, st_worker, request):
        """Auto test method — runs matching cases for the current platform."""
        raw_selectors = request.config.getoption("--case", default=None) or []
        selectors = [_parse_case_selector(v) for v in raw_selectors]
        manual_mode = request.config.getoption("--manual", default="exclude")
        rounds = request.config.getoption("--rounds", default=1)
        skip_golden = request.config.getoption("--skip-golden", default=False)
        enable_l2_swimlane = request.config.getoption("--enable-l2-swimlane", default=False)
        enable_dump_tensor = request.config.getoption("--dump-tensor", default=False)
        enable_pmu = request.config.getoption("--enable-pmu", default=0)
        if rounds > 1:
            if enable_l2_swimlane:
                logger.warning("Profiling disabled: --rounds > 1")
                enable_l2_swimlane = False
            if enable_dump_tensor:
                logger.warning("Dump tensor disabled: --rounds > 1")
                enable_dump_tensor = False
            if enable_pmu:
                logger.warning("PMU disabled: --rounds > 1")
                enable_pmu = 0

        cls_name = type(self).__name__
        callable_obj = self.build_callable(st_platform)
        sub_ids = getattr(type(self), "_st_sub_ids", {})

        # Primary device id: prefer the one actually allocated by st_worker
        # (each test class can hold a different slot from DevicePool); fall back
        # to the first id in --device if the fixture didn't stash it.
        primary_device_id = getattr(st_worker, "_st_device_id", None)
        if primary_device_id is None:
            raw_device = request.config.getoption("--device", default="0")
            primary_device_id = raw_device.split("-", 1)[0] if "-" in raw_device else raw_device
        is_hardware = not st_platform.endswith("sim")

        matched = []
        for case in self.CASES:
            if st_platform not in case["platforms"]:
                continue
            if not _match_selectors(cls_name, case["name"], selectors):
                continue
            is_manual = bool(case.get("manual"))
            if manual_mode == "exclude" and is_manual:
                continue
            if manual_mode == "only" and not is_manual:
                continue
            matched.append(case)

        if not matched:
            import pytest  # noqa: PLC0415

            pytest.skip(f"No cases matched {cls_name} (platform={st_platform}, manual={manual_mode})")

        run_class_cases(
            st_worker,
            self,
            matched,
            callable_obj=callable_obj,
            sub_ids=sub_ids,
            primary_device_id=primary_device_id,
            is_hardware=is_hardware,
            rounds=rounds,
            skip_golden=skip_golden,
            enable_l2_swimlane=enable_l2_swimlane,
            enable_dump_tensor=enable_dump_tensor,
            enable_pmu=enable_pmu,
        )

    # ------------------------------------------------------------------
    # Standalone entry point
    # ------------------------------------------------------------------

    @staticmethod
    def run_module(module_name):  # noqa: PLR0912, PLR0915 -- CLI parsing + dispatch; branches map to user-facing flags
        """Standalone entry: ``if __name__ == "__main__": SceneTestCase.run_module(__name__)``.

        Supports -d as either a single id or a range ("0-7"). When more than
        one device is provided (or any L3 case needs more than its single
        device), the outer invocation becomes a test dispatcher that spawns
        per-case subprocesses via ``parallel_scheduler``; each child re-enters
        this function in single-group mode via ``--runtime`` + ``--level``.
        """
        import argparse  # noqa: PLC0415

        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--platform", required=True)
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="0",
            help="Device id or range ('0', '4-7', '0,2,5')",
        )
        parser.add_argument(
            "--case",
            action="append",
            default=None,
            help="Case selector; repeatable. Forms: 'Foo' (any class), 'ClassA::Foo', 'ClassA::'",
        )
        parser.add_argument(
            "--manual",
            choices=["exclude", "include", "only"],
            default="exclude",
            help="Manual case handling: exclude (default), include, only",
        )
        parser.add_argument("--rounds", type=int, default=1, help="Run each case N times (default: 1)")
        parser.add_argument("--skip-golden", action="store_true", help="Skip golden comparison (benchmark mode)")
        parser.add_argument(
            "--enable-l2-swimlane", action="store_true", help="Enable perf swimlane collection (first round only)"
        )
        parser.add_argument("--dump-tensor", action="store_true", help="Dump per-task tensor I/O at runtime")
        parser.add_argument(
            "--enable-pmu",
            nargs="?",
            const=2,
            default=0,
            type=int,
            metavar="EVENT_TYPE",
            help="Enable PMU collection. Bare flag = PIPE_UTILIZATION(2). "
            "Pass event type to override (e.g. --enable-pmu 4)",
        )
        parser.add_argument("--build", action="store_true", help="Compile runtime from source")
        parser.add_argument(
            "--runtime",
            default=None,
            help="Only run classes with this _st_runtime (child-mode marker when combined with --level)",
        )
        parser.add_argument(
            "--level",
            type=int,
            choices=[2, 3],
            default=None,
            help="Only run classes with this _st_level (child-mode marker when combined with --runtime)",
        )
        parser.add_argument(
            "-x", "--exitfirst", action="store_true", help="Stop on first failing case (matches pytest -x)"
        )
        parser.add_argument(
            "--max-parallel",
            default="auto",
            help=(
                "Max in-flight subprocesses (make-style); decouples -d pool size from "
                "parallelism. 'auto' = min(nproc, len(-d)) on sim, len(-d) on hardware. "
                "Use e.g. '--max-parallel 2' to throttle sim on a CPU-constrained CI "
                "runner without shrinking -d. No short form — pytest reserves lowercase "
                "shorts; standalone mirrors that restriction for consistency."
            ),
        )
        parser.add_argument(
            "-c",
            "--pto-isa-commit",
            default=None,
            help="Checkout PTO-ISA at this git commit before running.",
        )
        parser.add_argument(
            "--clone-protocol",
            choices=["ssh", "https"],
            default="ssh",
            help="Git protocol for auto-cloning PTO-ISA (used with --pto-isa-commit). Default: ssh.",
        )
        parser.add_argument(
            "--log-level",
            choices=LOG_LEVEL_CHOICES,
            default=DEFAULT_LOG_LEVEL,
            help=f"Root logger level (default: {DEFAULT_LOG_LEVEL})",
        )
        args = parser.parse_args()
        configure_logging(args.log_level)

        os.environ["PTO_ISA_ROOT"] = ensure_pto_isa_root(
            commit=args.pto_isa_commit,
            clone_protocol=args.clone_protocol,
            update_if_exists=True,
            verbose=True,
        )

        if args.rounds > 1 and args.enable_l2_swimlane:
            logger.warning("Profiling disabled: --rounds > 1")
            args.enable_l2_swimlane = False

        from .parallel_scheduler import default_max_parallel, device_range_to_list  # noqa: PLC0415

        device_ids = device_range_to_list(args.device)
        if not device_ids:
            print("ERROR: --device must be a non-empty id or range", file=sys.stderr)
            sys.exit(2)
        args.device_ids = device_ids
        # Keep ``args.device`` as an int for paths that expect a single id
        # (profiling snapshots, device-id binding inside one worker). In child
        # mode this is the single allocated id; in parent mode we use the first
        # slot but the dispatcher doesn't actually run tests here.
        args.device = device_ids[0]

        # Resolve -j (max parallel) — 'auto' is CPU-aware on sim, device-count on hardware.
        if args.max_parallel in (None, "", "auto"):
            args.max_parallel = default_max_parallel(args.platform, device_ids)
        else:
            try:
                args.max_parallel = int(args.max_parallel)
            except (TypeError, ValueError):
                print(f"ERROR: -j must be 'auto' or an integer, got {args.max_parallel!r}", file=sys.stderr)
                sys.exit(2)
            if args.max_parallel < 1:
                print(f"ERROR: -j must be >= 1, got {args.max_parallel}", file=sys.stderr)
                sys.exit(2)
        # Note: profiling + parallelism used to be blocked here because perf
        # files shared a process-global directory. The test dispatcher now scopes
        # each subprocess to its own SIMPLER_L2_PERF_RECORDS_OUTPUT_DIR so the combination
        # is safe.

        module = sys.modules[module_name]
        test_classes = [
            v
            for v in vars(module).values()
            if isinstance(v, type) and issubclass(v, SceneTestCase) and v is not SceneTestCase and hasattr(v, "CASES")
        ]

        # Apply --runtime/--level filters (child mode sets both; parent may also
        # use them when the user wants a narrow run).
        if args.runtime is not None:
            test_classes = [c for c in test_classes if getattr(c, "_st_runtime", None) == args.runtime]
        if args.level is not None:
            test_classes = [c for c in test_classes if getattr(c, "_st_level", None) == args.level]
        if not test_classes:
            print(
                f"No matching classes (runtime={args.runtime}, level={args.level})",
                file=sys.stderr,
            )
            sys.exit(0)

        selectors = [_parse_case_selector(v) for v in (args.case or [])]
        try:
            selected = _select_cases(test_classes, args.platform, selectors, args.manual)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)

        selected_by_cls: dict[type, list[dict]] = {}
        for cls, case in selected:
            selected_by_cls.setdefault(cls, []).append(case)

        # L3 profiling not supported yet (multi-chip-process filename collision).
        # Mirror the pytest-side guard so standalone users get the same early-fail.
        if args.enable_l2_swimlane:
            l3_classes = sorted(cls.__name__ for cls in selected_by_cls if cls._st_level == 3)
            if l3_classes:
                print(
                    f"ERROR: --enable-l2-swimlane is not supported for L3 tests yet — "
                    f"multi-chip-process filename collision unresolved. "
                    f"L3 classes selected: {', '.join(l3_classes)}. "
                    f"Either drop --enable-l2-swimlane or scope to L2 with --level 2.",
                    file=sys.stderr,
                )
                sys.exit(2)

        # Child mode: both --runtime and --level set. Run inline without
        # spawning further subprocesses; this is the path dispatcher
        # children take after we re-enter run_module.
        child_mode = args.runtime is not None and args.level is not None

        if not child_mode:
            has_multi_dev_case = any(
                int(case.get("config", {}).get("device_count", 1)) > 1
                for cases in selected_by_cls.values()
                for case in cases
            )
            has_multiple_groups = len({(cls._st_runtime, cls._st_level) for cls in selected_by_cls}) > 1
            needs_orchestration = len(device_ids) > 1 or has_multi_dev_case or has_multiple_groups
            if needs_orchestration:
                ok = _dispatch_test_phases_standalone(module_name, selected_by_cls, args)
                sys.exit(0 if ok else 1)

        # ----- Inline execution (single group or child mode) -----
        by_rt_level: dict[tuple[str, int], list[type]] = {}
        for cls in selected_by_cls:
            by_rt_level.setdefault((cls._st_runtime, cls._st_level), []).append(cls)

        is_hardware = not args.platform.endswith("sim")
        if args.enable_l2_swimlane and is_hardware and "-d" not in sys.argv and "--device" not in sys.argv:
            print(
                f"WARNING: --enable-l2-swimlane on hardware platform '{args.platform}' without explicit "
                "-d/--device; defaulting to device 0 may point scheduler deep-dive at the wrong log.",
                file=sys.stderr,
            )

        ok = True
        for (runtime, level), group in by_rt_level.items():
            print(f"\n=== Runtime: {runtime}  Level: {level} ===")
            worker, per_class_sub_ids = _create_standalone_worker(group, level, args, selected_by_cls)
            try:
                for cls in group:
                    inst = cls()
                    callable_obj = inst.build_callable(args.platform)
                    sub_ids = per_class_sub_ids.get(cls, {})
                    for case in selected_by_cls[cls]:
                        label = f"{cls.__name__}::{case['name']}"
                        print(f"  {label} ... ", end="", flush=True)
                        try:
                            run_class_cases(
                                worker,
                                inst,
                                [case],
                                callable_obj=callable_obj,
                                sub_ids=sub_ids,
                                primary_device_id=args.device,
                                is_hardware=is_hardware,
                                rounds=args.rounds,
                                skip_golden=args.skip_golden,
                                enable_l2_swimlane=args.enable_l2_swimlane,
                                enable_dump_tensor=args.dump_tensor,
                                enable_pmu=args.enable_pmu,
                            )
                            print("PASSED")
                        except Exception as e:  # noqa: BLE001
                            print(f"FAILED: {e}")
                            ok = False
                            if args.exitfirst:
                                raise SystemExit(1) from None
            finally:
                if level == 2:
                    worker.finalize()
                else:
                    worker.close()

        sys.exit(0 if ok else 1)


def _dispatch_test_phases_standalone(module_name, selected_by_cls, args):  # noqa: PLR0912 -- L3 + L2 phases + chunking + fail-fast
    """Parent-mode test dispatcher for run_module.

    L3 phase: one subprocess per class, scheduled by device count.
    L2 phase: per-runtime fanout — up to max_parallel concurrent subprocesses,
    each owning one device and running its round-robin chunk of classes.

    Returns True on full success, False if any child failed.
    """
    from .parallel_scheduler import Job, format_device_range, run_jobs  # noqa: PLC0415

    module = sys.modules[module_name]
    # Path to the user's test script — sys.argv[0] is the script they invoked.
    script = os.path.abspath(getattr(module, "__file__", sys.argv[0]))

    common = ["-p", args.platform, "--manual", args.manual, "--log-level", args.log_level]
    if args.rounds != 1:
        common += ["--rounds", str(args.rounds)]
    if args.skip_golden:
        common.append("--skip-golden")
    if args.enable_l2_swimlane:
        common.append("--enable-l2-swimlane")
    if args.dump_tensor:
        common.append("--dump-tensor")
    if args.build:
        common.append("--build")

    # ----- L3 phase: one subprocess per class (not per case).
    # The child's _create_standalone_worker allocates max(cls.CASES.device_count)
    # for the whole class, so the scheduler must grant the class-level max,
    # otherwise a class with a 4-device case can't run any of its 1-device
    # cases when we dispatch them individually with --device <1>. Cases inside
    # a class still run serially in the child, reusing the L3 Worker.
    l3_jobs = []
    for cls, cases in selected_by_cls.items():
        if cls._st_level != 3:
            continue
        if not cases:
            continue
        class_dev_count = max(int(c.get("config", {}).get("device_count", 1)) for c in cases)
        label = f"L3 {cls.__name__} (rt={cls._st_runtime}, dev={class_dev_count})"

        def _build(ids, _cls=cls.__name__, _rt=cls._st_runtime):
            return [
                sys.executable,
                script,
                *common,
                "-d",
                format_device_range(ids),
                "--case",
                f"{_cls}::",
                "--runtime",
                _rt,
                "--level",
                "3",
            ]

        # Each L3 class gets its own perf output subdir so two concurrent L3
        # cases can't collide on a per_swimlane_<second-precision-ts>.json.
        # Anchor to _project_root() so the C++ runtime (resolved against its
        # own CWD) and Python post-processing agree on the location.
        child_env = {
            **os.environ,
            "SIMPLER_L2_PERF_RECORDS_OUTPUT_DIR": str(
                _project_root() / "outputs" / f"l2_perf_records_l3_{cls.__name__}"
            ),
        }
        l3_jobs.append(Job(label=label, device_count=class_dev_count, build_cmd=_build, env=child_env))

    l3_failed = False
    if l3_jobs:
        print(
            f"\n{'=' * 60}\n  L3 phase: {len(l3_jobs)} case(s), pool={args.device_ids}, "
            f"max_parallel={args.max_parallel}\n{'=' * 60}\n"
        )

        def _on_done(res):
            tag = "PASSED" if res.returncode == 0 else f"FAILED (rc={res.returncode})"
            print(f"  {res.label}: {tag} on devices {res.device_ids}", flush=True)

        try:
            results = run_jobs(
                l3_jobs,
                args.device_ids,
                max_parallel=args.max_parallel,
                fail_fast=args.exitfirst,
                on_job_done=_on_done,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return False
        l3_failed = any(r.returncode != 0 for r in results)
        if l3_failed and args.exitfirst:
            return False

    # ----- L2 phase: runtimes serial (CANN isolation); within a runtime, fan
    # out classes across device_ids as one subprocess per device. Each child
    # owns one ChipWorker and runs its chunk of classes back-to-back (layer-4
    # reuse). Single-device case reduces to one subprocess per runtime.
    l2_by_runtime: dict[str, list[type]] = {}
    for cls in selected_by_cls:
        if cls._st_level == 2:
            l2_by_runtime.setdefault(cls._st_runtime, []).append(cls)

    l2_failed = False
    for rt in sorted(l2_by_runtime):
        classes = l2_by_runtime[rt]
        # Chunk count = min(-j, number of classes). We intentionally do NOT
        # include len(device_ids) here: each chunk uses 1 device and at most
        # max_parallel chunks run concurrently, so a pool bigger than -j just
        # leaves unused ids. Fewer, larger chunks also amortize ChipWorker
        # init (layer-4 reuse) over more cases.
        n = min(args.max_parallel, len(classes))
        if n == 0:
            continue
        # Round-robin distribute classes to N children.
        chunks: list[list[type]] = [[] for _ in range(n)]
        for i, cls in enumerate(classes):
            chunks[i % n].append(cls)

        header = f"  L2 Runtime: {rt}" + (f"  [fanout n={n}]" if n > 1 else "")
        print(f"\n{'=' * 60}\n{header}\n{'=' * 60}\n")

        l2_jobs = []
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            dev = args.device_ids[i]
            case_filters: list[str] = []
            for cls in chunk:
                # User-supplied selectors still filter; we scope to this chunk's
                # classes using "ClassName::<case>" or "ClassName::" (whole class).
                for sel in args.case or []:
                    if "::" in sel:
                        sel_cls, sel_case = sel.split("::", 1)
                        if not sel_cls or sel_cls == cls.__name__:
                            case_filters.append(f"{cls.__name__}::{sel_case}" if sel_case else f"{cls.__name__}::")
                    else:
                        # Bare selector "Foo" = case name in any class — forward as-is,
                        # but still scope to this chunk's classes.
                        case_filters.append(f"{cls.__name__}::{sel}")
                if not args.case:
                    case_filters.append(f"{cls.__name__}::")
            label = f"L2 {rt} dev={dev} ({len(chunk)} class(es))"

            def _build(ids, _rt=rt, _dev=dev, _filters=tuple(case_filters)):
                cmd = [
                    sys.executable,
                    script,
                    *common,
                    "-d",
                    str(_dev),
                    "--runtime",
                    _rt,
                    "--level",
                    "2",
                ]
                for f in _filters:
                    cmd += ["--case", f]
                return cmd

            # device_count=1 for L2 fanout children (each child uses one slot).
            # Scope perf output to this fanout child's own subdir so concurrent
            # L2 children don't collide on the second-precision perf filename.
            # Absolute path anchored to _project_root() so the C++ runtime and
            # Python post-processing agree on the filesystem location.
            child_env = {
                **os.environ,
                "SIMPLER_L2_PERF_RECORDS_OUTPUT_DIR": str(
                    _project_root() / "outputs" / f"l2_perf_records_l2_{rt}_dev{dev}"
                ),
            }
            l2_jobs.append(Job(label=label, device_count=1, build_cmd=_build, env=child_env))

        # Use the same scheduler: pool=device_ids, fail_fast=exitfirst. This
        # gives us automatic parallelism + SIGTERM on fail-fast.
        def _on_l2_done(res):
            tag = "PASSED" if res.returncode == 0 else f"FAILED (rc={res.returncode})"
            print(f"  {res.label}: {tag}", flush=True)

        try:
            results = run_jobs(
                l2_jobs,
                args.device_ids,
                max_parallel=args.max_parallel,
                fail_fast=args.exitfirst,
                on_job_done=_on_l2_done,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            l2_failed = True
            if args.exitfirst:
                break
            continue

        if any(r.returncode != 0 for r in results):
            l2_failed = True
            if args.exitfirst:
                break

    # Flatten per-subprocess outputs/l2_perf_records_*/ subdirs back to outputs/.
    # Anchor to _project_root() so flushing works when the user runs a
    # standalone test from a subdirectory.
    from .parallel_scheduler import flatten_l2_perf_records_subdirs  # noqa: PLC0415

    flatten_l2_perf_records_subdirs(_project_root() / "outputs")

    return not (l3_failed or l2_failed)


def _create_standalone_worker(group, level, args, selected_by_cls):
    """Create a Worker for a (runtime, level) group in run_module.

    ``level`` is passed explicitly by the caller; do not read it from
    ``group[0]._st_level`` because groups are now keyed on (runtime, level)
    and mixed-level files are allowed.

    ``selected_by_cls`` is the dict of the cases that will actually run (after
    ``--case`` / ``--manual`` / platform filtering). L3 ``max_devices`` /
    ``max_sub_workers`` must be computed from these, not from ``cls.CASES``:
    otherwise a manual case with a larger ``device_count`` inflates the
    allocation even when it isn't scheduled.
    """
    first_cls = group[0]
    build = getattr(args, "build", False)
    if level == 2:
        return first_cls._create_worker(args.platform, args.device, build=build), {}

    from simpler.worker import Worker  # noqa: PLC0415

    max_devices = max(
        (c.get("config", {}).get("device_count", 1) for cls in group for c in selected_by_cls.get(cls, [])),
        default=1,
    )
    max_subs = max(
        (c.get("config", {}).get("num_sub_workers", 0) for cls in group for c in selected_by_cls.get(cls, [])),
        default=0,
    )
    # Prefer the allocated list (dispatcher child mode), fall back to
    # contiguous range starting at args.device (legacy inline path).
    allocated = getattr(args, "device_ids", None)
    if allocated and len(allocated) >= max_devices:
        device_ids = allocated[:max_devices]
    else:
        device_ids = list(range(args.device, args.device + max_devices))
    worker = Worker(
        level=3,
        device_ids=device_ids,
        num_sub_workers=max_subs,
        platform=args.platform,
        runtime=first_cls._st_runtime,
        build=build,
    )
    # Register sub callables per-class to avoid name collisions
    per_class_sub_ids: dict[type, dict] = {}
    for cls in group:
        cls_sub_ids = {}
        for entry in cls.CALLABLE.get("callables", []):
            if "callable" in entry:
                cid = worker.register(entry["callable"])
                cls_sub_ids[entry["name"]] = cid
        per_class_sub_ids[cls] = cls_sub_ids
    worker.init()
    return worker, per_class_sub_ids
