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
  CALLABLE: what to compile (orchestration + incores)
  CASES: how to run (per-case platform, config, params)
  generate_args / compute_golden: data + golden comparison
"""

import inspect
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from .environment import ensure_python_path

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

    def clone(self) -> "TaskArgsBuilder":
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
# Internal helpers
# ---------------------------------------------------------------------------


def _build_chip_task_args(test_args: TaskArgsBuilder, orch_signature: list):
    """Build ChipStorageTaskArgs from TaskArgsBuilder + identify outputs via signature.

    Returns:
        chip_args: ChipStorageTaskArgs (for worker.run)
        output_names: list of tensor names that are OUTPUT or INOUT
    """
    ensure_python_path()
    from task_interface import ArgDirection, ChipStorageTaskArgs, make_tensor_arg, scalar_to_uint64  # noqa: PLC0415

    chip_args = ChipStorageTaskArgs()
    output_names: list[str] = []

    tensor_idx = 0
    for spec in test_args.specs:
        if isinstance(spec, Tensor):
            chip_args.add_tensor(make_tensor_arg(spec.value))
            if tensor_idx >= len(orch_signature):
                raise ValueError(
                    f"Tensor '{spec.name}' at index {tensor_idx} has no matching entry in "
                    f"orchestration signature (length {len(orch_signature)}). "
                    f"Update CALLABLE['orchestration']['signature'] to match generate_args()."
                )
            direction = orch_signature[tensor_idx]
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
    if "orchestration" in callable_spec:
        orch = callable_spec["orchestration"]
        if "source" in orch and not os.path.isabs(orch["source"]):
            callable_spec["orchestration"] = dict(orch)
            callable_spec["orchestration"]["source"] = str(cls_dir / orch["source"])
    if "incores" in callable_spec:
        resolved = []
        for k in callable_spec["incores"]:
            k = dict(k)
            if "source" in k and not os.path.isabs(k["source"]):
                k["source"] = str(cls_dir / k["source"])
            resolved.append(k)
        callable_spec["incores"] = resolved


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
        """Compile CALLABLE -> ChipCallable. Session-cached."""
        callable_spec = cls.CALLABLE
        orch = callable_spec["orchestration"]
        incores = callable_spec["incores"]

        cache_key = (cls.__qualname__, platform, cls._st_runtime)
        if cache_key in _compile_cache:
            return _compile_cache[cache_key]

        from .elf_parser import extract_text_section  # noqa: PLC0415
        from .kernel_compiler import KernelCompiler  # noqa: PLC0415
        from .pto_isa import ensure_pto_isa_root  # noqa: PLC0415

        ensure_python_path()
        from task_interface import ChipCallable, CoreCallable  # noqa: PLC0415

        pto_isa_root = ensure_pto_isa_root()
        kc = KernelCompiler(platform=platform)
        is_sim = platform.endswith("sim")

        orch_binary = kc.compile_orchestration(cls._st_runtime, orch["source"])
        inc_dirs = kc.get_orchestration_include_dirs(cls._st_runtime)

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
        )
        _compile_cache[cache_key] = chip_callable
        return chip_callable

    # ------------------------------------------------------------------
    # Worker creation
    # ------------------------------------------------------------------

    @classmethod
    def _get_binaries(cls, platform):
        from .runtime_builder import RuntimeBuilder  # noqa: PLC0415

        return RuntimeBuilder(platform=platform).get_binaries(cls._st_runtime, build=False)

    @classmethod
    def _create_worker(cls, platform, device_id=0):
        ensure_python_path()
        from task_interface import ChipWorker  # noqa: PLC0415

        bins = cls._get_binaries(platform)
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
        """Default: compile from CALLABLE. Override for L3+."""
        return self.compile_chip_callable(platform)

    def _build_config(self, config_dict):
        ensure_python_path()
        from task_interface import CallConfig  # noqa: PLC0415

        config = CallConfig()
        config.block_dim = config_dict.get("block_dim", 1)
        config.aicpu_thread_num = config_dict.get("aicpu_thread_num", 3)
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

    def _run_and_validate(self, worker, callable_obj, case):
        import torch  # noqa: PLC0415

        params = case.get("params", {})
        config = self._build_config(case.get("config", {}))
        orch_sig = self.CALLABLE.get("orchestration", {}).get("signature", [])

        # Build args
        test_args = self.generate_args(params)
        chip_args, output_names = _build_chip_task_args(test_args, orch_sig)

        # Clone for golden
        golden_args = test_args.clone()
        self.compute_golden(golden_args, params)

        # Execute
        with _temporary_env(self._resolve_env()):
            worker.run(callable_obj, chip_args, block_dim=config.block_dim, aicpu_thread_num=config.aicpu_thread_num)

        # Compare outputs
        for name in output_names:
            actual = getattr(test_args, name)
            expected = getattr(golden_args, name)
            if not torch.allclose(actual, expected, rtol=self.RTOL, atol=self.ATOL):
                diff = (actual - expected).abs().max().item()
                raise AssertionError(
                    f"Golden mismatch on '{name}': max_diff={diff}, rtol={self.RTOL}, atol={self.ATOL}"
                )

    # ------------------------------------------------------------------
    # pytest auto test method
    # ------------------------------------------------------------------

    def test_run(self, st_platform, st_worker, request):
        """Auto test method — runs matching cases for the current platform."""
        case_filter = request.config.getoption("--case", default=None)
        all_cases = request.config.getoption("--all-cases", default=False)

        callable_obj = self.build_callable(st_platform)
        ran_any = False
        for case in self.CASES:
            if st_platform not in case["platforms"]:
                continue
            if case_filter and case["name"] != case_filter:
                continue
            if case.get("manual") and not case_filter and not all_cases:
                continue
            self._run_and_validate(st_worker, callable_obj, case)
            ran_any = True

        if not ran_any:
            import pytest  # noqa: PLC0415

            pytest.skip(f"No cases matched platform={st_platform}")

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
        parser.add_argument("--case", help="Run specific case name")
        parser.add_argument("--all-cases", action="store_true", help="Include manual cases")
        args = parser.parse_args()

        module = sys.modules[module_name]
        test_classes = [
            v
            for v in vars(module).values()
            if isinstance(v, type) and issubclass(v, SceneTestCase) and v is not SceneTestCase and hasattr(v, "CASES")
        ]

        by_runtime: dict[str, list[type]] = {}
        for cls in test_classes:
            by_runtime.setdefault(cls._st_runtime, []).append(cls)

        ok = True
        for runtime, group in by_runtime.items():
            print(f"\n=== Runtime: {runtime} ===")
            worker = group[0]._create_worker(args.platform, args.device)
            try:
                for cls in group:
                    inst = cls()
                    callable_obj = inst.build_callable(args.platform)
                    for case in inst.CASES:
                        if args.platform not in case["platforms"]:
                            continue
                        if args.case and case["name"] != args.case:
                            continue
                        if case.get("manual") and not args.case and not args.all_cases:
                            continue
                        label = f"{cls.__name__}::{case['name']}"
                        print(f"  {label} ... ", end="", flush=True)
                        try:
                            inst._run_and_validate(worker, callable_obj, case)
                            print("PASSED")
                        except Exception as e:
                            print(f"FAILED: {e}")
                            ok = False
            finally:
                worker.finalize()
        sys.exit(0 if ok else 1)
