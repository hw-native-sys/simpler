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

import inspect
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from .log_config import DEFAULT_LOG_LEVEL, LOG_LEVEL_CHOICES, configure_logging

_compile_cache: dict[tuple[str, str, str], object] = {}


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
        make_tensor_arg,
        scalar_to_uint64,
    )

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
        make_tensor_arg,
        scalar_to_uint64,
    )

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
    return _project_root() / "outputs"


def _snapshot_perf_files() -> set[Path]:
    d = _outputs_dir()
    return set(d.glob("perf_swimlane_*.json")) if d.exists() else set()


def _wait_new_perf_file(before: set[Path], timeout: float = 2.0) -> Path | None:
    """Wait briefly for a new ``perf_swimlane_*.json`` to appear in outputs/."""
    import time  # noqa: PLC0415

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        new = _snapshot_perf_files() - before
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
) -> None:
    """Invoke ``tools/swimlane_converter.py``.

    When ``input_path`` is given, the converter derives its output filename from
    the input's timestamp (see ``tools/swimlane_converter.py::_resolve_output_path``).
    Without it, the converter auto-selects the latest ``perf_swimlane_*.json``.
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
    before_perf: set[Path],
    before_device: set[Path] | None,
) -> None:
    """Post-case: rename the new perf file to include ``case_label`` (guarding against
    the runtime's second-precision filename collisions), then invoke the converter.

    The ``perf_swimlane_`` prefix is preserved so the converter's stem-based output
    naming still strips it and produces ``merged_swimlane_<ts>_<case_label>.json``.
    """
    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    perf_file = _wait_new_perf_file(before_perf)
    if perf_file is None:
        logger.warning(f"[{case_label}] No new perf_swimlane_*.json produced; skipping conversion")
        return
    safe_label = _sanitize_for_filename(case_label)
    suffix = perf_file.stem[len("perf_swimlane_") :] if perf_file.stem.startswith("perf_swimlane_") else perf_file.stem
    renamed = perf_file.with_name(f"perf_swimlane_{suffix}_{safe_label}.json")
    if renamed.exists():
        logger.warning(f"[{case_label}] target {renamed.name} already exists; overwriting")
        renamed.unlink()
    perf_file.rename(renamed)
    device_log = None
    if before_device is not None:
        device_log = _wait_new_device_log(device_id, before_device)
        if device_log is None:
            logger.warning(f"[{case_label}] no new device log found; scheduler deep-dive may use stale log")
    _run_swimlane_converter(input_path=renamed, device_id=device_id, device_log=device_log)


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

    def _build_config(self, config_dict, enable_profiling=False, enable_dump_tensor=False):
        from simpler.task_interface import ChipCallConfig  # noqa: PLC0415

        config = ChipCallConfig()
        config.block_dim = config_dict.get("block_dim", 1)
        config.aicpu_thread_num = config_dict.get("aicpu_thread_num", 3)
        config.enable_profiling = enable_profiling
        config.enable_dump_tensor = enable_dump_tensor
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
        enable_profiling=False,
        enable_dump_tensor=False,
    ):
        if self._st_level == 2:
            self._run_and_validate_l2(
                worker,
                callable_obj,
                case,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_profiling=enable_profiling,
                enable_dump_tensor=enable_dump_tensor,
            )
        elif self._st_level == 3:
            self._run_and_validate_l3(
                worker,
                callable_obj,
                sub_ids or {},
                case,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_profiling=enable_profiling,
                enable_dump_tensor=enable_dump_tensor,
            )

    def _run_and_validate_l2(
        self, worker, callable_obj, case, rounds=1, skip_golden=False, enable_profiling=False, enable_dump_tensor=False
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
                enable_profiling=(enable_profiling and round_idx == 0),
                enable_dump_tensor=enable_dump_tensor,
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
        enable_profiling=False,
        enable_dump_tensor=False,
    ):
        from simpler.worker import Task  # noqa: PLC0415

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
                enable_profiling=(enable_profiling and round_idx == 0),
                enable_dump_tensor=enable_dump_tensor,
            )

            # Wrap in Task — user orch signature: (orch, callables, task_args, config)
            def task_orch(orch, _unused, _ns=ns, _test_args=test_args, _config=config):
                orch_fn(orch, _ns, _test_args, _config)

            with _temporary_env(self._resolve_env()):
                worker.run(Task(orch=task_orch))

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
        enable_profiling = request.config.getoption("--enable-profiling", default=False)
        enable_dump_tensor = request.config.getoption("--dump-tensor", default=False)

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

        ran_any = False
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
            before_perf = _snapshot_perf_files() if enable_profiling else set()
            before_device = _snapshot_device_logs(primary_device_id) if enable_profiling and is_hardware else None
            try:
                self._run_and_validate(
                    st_worker,
                    callable_obj,
                    case,
                    sub_ids=sub_ids,
                    rounds=rounds,
                    skip_golden=skip_golden,
                    enable_profiling=enable_profiling,
                    enable_dump_tensor=enable_dump_tensor,
                )
            finally:
                if enable_profiling:
                    _convert_case_swimlane(f"{cls_name}_{case['name']}", primary_device_id, before_perf, before_device)
            ran_any = True

        if not ran_any:
            import pytest  # noqa: PLC0415

            pytest.skip(f"No cases matched {cls_name} (platform={st_platform}, manual={manual_mode})")

    # ------------------------------------------------------------------
    # Standalone entry point
    # ------------------------------------------------------------------

    @staticmethod
    def run_module(module_name):
        """Standalone entry: ``if __name__ == "__main__": SceneTestCase.run_module(__name__)``."""
        import argparse  # noqa: PLC0415

        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--platform", required=True)
        parser.add_argument("-d", "--device", type=int, default=0)
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
        parser.add_argument("-n", "--rounds", type=int, default=1, help="Run each case N times (default: 1)")
        parser.add_argument("--skip-golden", action="store_true", help="Skip golden comparison (benchmark mode)")
        parser.add_argument("--enable-profiling", action="store_true", help="Enable profiling (first round only)")
        parser.add_argument("--dump-tensor", action="store_true", help="Dump per-task tensor I/O at runtime")
        parser.add_argument("--build", action="store_true", help="Compile runtime from source")
        parser.add_argument(
            "--log-level",
            choices=LOG_LEVEL_CHOICES,
            default=DEFAULT_LOG_LEVEL,
            help=f"Root logger level (default: {DEFAULT_LOG_LEVEL})",
        )
        args = parser.parse_args()
        configure_logging(args.log_level)

        module = sys.modules[module_name]
        test_classes = [
            v
            for v in vars(module).values()
            if isinstance(v, type) and issubclass(v, SceneTestCase) and v is not SceneTestCase and hasattr(v, "CASES")
        ]

        selectors = [_parse_case_selector(v) for v in (args.case or [])]
        try:
            selected = _select_cases(test_classes, args.platform, selectors, args.manual)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)

        selected_by_cls: dict[type, list[dict]] = {}
        for cls, case in selected:
            selected_by_cls.setdefault(cls, []).append(case)

        by_runtime: dict[str, list[type]] = {}
        for cls in selected_by_cls:
            by_runtime.setdefault(cls._st_runtime, []).append(cls)

        is_hardware = not args.platform.endswith("sim")
        if args.enable_profiling and is_hardware and "-d" not in sys.argv and "--device" not in sys.argv:
            print(
                f"WARNING: --enable-profiling on hardware platform '{args.platform}' without explicit "
                "-d/--device; defaulting to device 0 may point scheduler deep-dive at the wrong log.",
                file=sys.stderr,
            )

        ok = True
        for runtime, group in by_runtime.items():
            print(f"\n=== Runtime: {runtime} ===")
            worker, per_class_sub_ids = _create_standalone_worker(group, args)
            try:
                for cls in group:
                    inst = cls()
                    callable_obj = inst.build_callable(args.platform)
                    sub_ids = per_class_sub_ids.get(cls, {})
                    for case in selected_by_cls[cls]:
                        label = f"{cls.__name__}::{case['name']}"
                        print(f"  {label} ... ", end="", flush=True)
                        before_perf = _snapshot_perf_files() if args.enable_profiling else set()
                        before_device = (
                            _snapshot_device_logs(args.device) if args.enable_profiling and is_hardware else None
                        )
                        try:
                            inst._run_and_validate(
                                worker,
                                callable_obj,
                                case,
                                sub_ids=sub_ids,
                                rounds=args.rounds,
                                skip_golden=args.skip_golden,
                                enable_profiling=args.enable_profiling,
                                enable_dump_tensor=args.dump_tensor,
                            )
                            print("PASSED")
                        except Exception as e:
                            print(f"FAILED: {e}")
                            ok = False
                        finally:
                            if args.enable_profiling:
                                _convert_case_swimlane(
                                    f"{cls.__name__}_{case['name']}",
                                    args.device,
                                    before_perf,
                                    before_device,
                                )
            finally:
                if group[0]._st_level == 2:
                    worker.finalize()
                else:
                    worker.close()

        sys.exit(0 if ok else 1)


def _create_standalone_worker(group, args):
    """Create a Worker for standalone run_module entry point."""
    first_cls = group[0]
    level = first_cls._st_level
    build = getattr(args, "build", False)
    if level == 2:
        return first_cls._create_worker(args.platform, args.device, build=build), {}

    from simpler.worker import Worker  # noqa: PLC0415

    max_devices = max((c.get("config", {}).get("device_count", 1) for cls in group for c in cls.CASES), default=1)
    max_subs = max((c.get("config", {}).get("num_sub_workers", 0) for cls in group for c in cls.CASES), default=0)
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
