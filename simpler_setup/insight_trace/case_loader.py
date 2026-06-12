# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

from .models import SceneCaseContext


def load_module(path: Path) -> ModuleType:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_scene_test_class(module: ModuleType) -> type:
    candidates = []
    for obj in module.__dict__.values():
        if inspect.isclass(obj) and hasattr(obj, "CALLABLE") and hasattr(obj, "CASES"):
            if getattr(obj, "_st_level", None) == 2:
                candidates.append(obj)
    if not candidates:
        raise ValueError("No level-2 SceneTestCase class found")
    if len(candidates) > 1:
        names = ", ".join(cls.__name__ for cls in candidates)
        raise ValueError(f"Multiple SceneTestCase classes found: {names}")
    return candidates[0]


def load_scene_case(test_module: Path, case_name: str) -> SceneCaseContext:
    module = load_module(test_module)
    test_class = find_scene_test_class(module)
    case = next((case for case in test_class.CASES if case.get("name") == case_name), None)
    if case is None:
        available = ", ".join(case.get("name", "<unnamed>") for case in test_class.CASES)
        raise ValueError(f"Unknown case {case_name!r}; available cases: {available}")
    return SceneCaseContext(
        test_class=test_class,
        case=case,
        callable_spec=test_class.CALLABLE,
        test_module=test_module.resolve(),
        module_dir=test_module.resolve().parent,
        runtime=getattr(test_class, "_st_runtime", ""),
    )
