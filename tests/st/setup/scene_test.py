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

Supports multiple levels:
  level=2: single ChipWorker (L2 device execution)
  level=3: Worker with orch function + submit (L3 distributed, future)
"""

import inspect
import os
import sys
from pathlib import Path

from .environment import ensure_python_path

_compile_cache: dict[tuple[str, str, str], object] = {}


def _build_args(inputs_list, output_names=None):
    """Convert [(name, value), ...] -> (ChipStorageTaskArgs, all_tensors, output_tensors)."""
    import torch  # noqa: PLC0415

    ensure_python_path()
    from task_interface import ChipStorageTaskArgs, make_tensor_arg, scalar_to_uint64  # noqa: PLC0415, E402

    orch_args = ChipStorageTaskArgs()
    tensors: dict = {}
    outputs: dict = {}
    for name, val in inputs_list:
        if isinstance(val, torch.Tensor):
            orch_args.add_tensor(make_tensor_arg(val))
            tensors[name] = val
            if output_names and name in output_names:
                outputs[name] = val
        elif hasattr(val, "value"):
            orch_args.add_scalar(scalar_to_uint64(val))
        else:
            import numpy as np  # noqa: PLC0415

            if isinstance(val, np.ndarray):
                t = torch.from_numpy(val)
                orch_args.add_tensor(make_tensor_arg(t))
                tensors[name] = t
                if output_names and name in output_names:
                    outputs[name] = t
            else:
                orch_args.add_scalar(scalar_to_uint64(val))
    if not outputs:
        outputs = dict(tensors)
    return orch_args, tensors, outputs


def scene_test(level: int, platforms: list[str], runtime: str):
    """Decorator marking a SceneTestCase with level, platform, and runtime metadata.

    Args:
        level: Hierarchy level. 2 = single ChipWorker, 3 = distributed Worker.
        platforms: List of supported platforms (e.g., ["a2a3sim", "a2a3"]).
        runtime: Runtime name (e.g., "tensormap_and_ringbuffer").
    """

    def decorator(cls):
        cls._st_level = level
        cls._st_platforms = list(platforms)
        cls._st_runtime = runtime
        cls_dir = Path(inspect.getfile(cls)).parent
        if hasattr(cls, "ORCHESTRATION") and "source" in cls.ORCHESTRATION:
            src = cls.ORCHESTRATION["source"]
            if not os.path.isabs(src):
                cls.ORCHESTRATION = dict(cls.ORCHESTRATION)
                cls.ORCHESTRATION["source"] = str(cls_dir / src)
        if hasattr(cls, "KERNELS"):
            resolved = []
            for k in cls.KERNELS:
                k = dict(k)
                if "source" in k and not os.path.isabs(k["source"]):
                    k["source"] = str(cls_dir / k["source"])
                resolved.append(k)
            cls.KERNELS = resolved
        return cls

    return decorator


class SceneTestCase:
    """Base class for scene tests at any hierarchy level.

    Subclasses define ORCHESTRATION, KERNELS, RUNTIME_CONFIG, generate_inputs(),
    and compute_golden().  The decorator ``@scene_test(level=N, ...)`` sets the
    execution model.
    """

    ORCHESTRATION: dict = {}
    KERNELS: list[dict] = []
    RUNTIME_CONFIG: dict = {}
    RTOL: float = 1e-5
    ATOL: float = 1e-5
    ALL_CASES: dict = {"default": {}}
    __outputs__: list[str] = []

    def generate_inputs(self, params):
        raise NotImplementedError

    def compute_golden(self, tensors, params):
        raise NotImplementedError

    @classmethod
    def _compile(cls, platform: str):
        """Compile orchestration + kernels -> ChipCallable. Session-cached."""
        source_dir = str(Path(cls.ORCHESTRATION["source"]).parent.parent)
        cache_key = (source_dir, platform, cls._st_runtime)
        if cache_key in _compile_cache:
            return _compile_cache[cache_key]
        from .elf_parser import extract_text_section  # noqa: PLC0415
        from .kernel_compiler import KernelCompiler  # noqa: PLC0415
        from .pto_isa import ensure_pto_isa_root  # noqa: PLC0415

        ensure_python_path()
        from task_interface import ChipCallable, CoreCallable  # noqa: PLC0415, E402

        pto_isa_root = ensure_pto_isa_root()
        kc = KernelCompiler(platform=platform)
        is_sim = platform.endswith("sim")
        orch_binary = kc.compile_orchestration(cls._st_runtime, cls.ORCHESTRATION["source"])
        inc_dirs = kc.get_orchestration_include_dirs(cls._st_runtime)
        kernel_binaries = []
        for k in cls.KERNELS:
            incore = kc.compile_incore(
                k["source"], core_type=k["core_type"], pto_isa_root=pto_isa_root, extra_include_dirs=inc_dirs
            )
            if not is_sim:
                incore = extract_text_section(incore)
            kernel_binaries.append((k["func_id"], CoreCallable.build(signature=k.get("signature", []), binary=incore)))
        chip_callable = ChipCallable.build(
            signature=cls.ORCHESTRATION.get("signature", []),
            func_name=cls.ORCHESTRATION["function_name"],
            binary=orch_binary,
            children=kernel_binaries,
        )
        _compile_cache[cache_key] = chip_callable
        return chip_callable

    @classmethod
    def _get_binaries(cls, platform: str):
        from .runtime_builder import RuntimeBuilder  # noqa: PLC0415

        return RuntimeBuilder(platform=platform).get_binaries(cls._st_runtime, build=False)

    @classmethod
    def _create_worker(cls, platform: str, device_id: int = 0):
        ensure_python_path()
        from task_interface import ChipWorker  # noqa: PLC0415, E402

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

    def _run_and_validate(self, worker, chip_callable, params=None):
        import torch  # noqa: PLC0415

        ensure_python_path()
        from task_interface import CallConfig  # noqa: PLC0415, E402

        if params is None:
            params = {}
        inputs_list = self.generate_inputs(params)
        orch_args, tensors, outputs = _build_args(inputs_list, self.__outputs__ or None)
        golden_tensors = {k: v.clone() for k, v in tensors.items()}
        self.compute_golden(golden_tensors, params)
        config = CallConfig()
        config.block_dim = self.RUNTIME_CONFIG.get("block_dim", 1)
        config.aicpu_thread_num = self.RUNTIME_CONFIG.get("aicpu_thread_num", 3)
        worker.run(chip_callable, orch_args, config)
        for name in self.__outputs__ or list(outputs.keys()):
            if name not in outputs or name not in golden_tensors:
                continue
            if not torch.allclose(outputs[name], golden_tensors[name], rtol=self.RTOL, atol=self.ATOL):
                diff = (outputs[name] - golden_tensors[name]).abs().max().item()
                raise AssertionError(f"Golden mismatch on '{name}': max_diff={diff}")

    def test_run(self, st_platform, st_chip_worker):
        """Auto test method — called by pytest with session fixtures."""
        chip_callable = self._compile(st_platform)
        for _case_name, params in self.ALL_CASES.items():
            self._run_and_validate(st_chip_worker, chip_callable, params)

    @staticmethod
    def run_module(module_name: str):
        """Standalone entry: ``if __name__ == "__main__": SceneTestCase.run_module(__name__)``."""
        import argparse  # noqa: PLC0415

        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--platform", required=True)
        parser.add_argument("-d", "--device", type=int, default=0)
        parser.add_argument("--case", help="Run specific class name")
        args = parser.parse_args()
        module = sys.modules[module_name]
        cases = [
            v
            for v in vars(module).values()
            if isinstance(v, type)
            and issubclass(v, SceneTestCase)
            and v is not SceneTestCase
            and hasattr(v, "_st_platforms")
            and args.platform in v._st_platforms
        ]
        if args.case:
            cases = [c for c in cases if c.__name__ == args.case]
        if not cases:
            print(f"No cases for platform {args.platform}")
            sys.exit(1)
        by_runtime: dict[str, list[type]] = {}
        for c in cases:
            by_runtime.setdefault(c._st_runtime, []).append(c)
        ok = True
        for runtime, group in by_runtime.items():
            print(f"\n=== Runtime: {runtime} ===")
            worker = group[0]._create_worker(args.platform, args.device)
            try:
                for cls in group:
                    inst = cls()
                    cc = cls._compile(args.platform)
                    for name, params in inst.ALL_CASES.items():
                        print(f"  {cls.__name__}::{name} ... ", end="", flush=True)
                        try:
                            inst._run_and_validate(worker, cc, params)
                            print("PASSED")
                        except Exception as e:
                            print(f"FAILED: {e}")
                            ok = False
            finally:
                worker.finalize()
        sys.exit(0 if ok else 1)
