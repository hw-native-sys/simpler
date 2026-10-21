# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generic node preparation descriptors, local context, and startup stage gates."""

from __future__ import annotations

import importlib
import json
import keyword
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .startup_artifacts import StartupArtifactSpec, activate_startup_artifacts, install_startup_artifacts

if TYPE_CHECKING:
    from .callable_identity import CallableHandle
    from .worker import Worker

_DEFAULT_PARAMETERS = object()


def _valid_import_component(value: str) -> bool:
    return all(part.isidentifier() and not keyword.iskeyword(part) for part in value.split("."))


def _validate_json(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError("startup hook parameters must contain only finite JSON values")


@dataclass(frozen=True, init=False)
class NodeStartupHook:
    """An importable preparation function with a defensive snapshot of JSON parameters."""

    import_path: str
    _parameters_json: str = field(repr=False)

    def __init__(self, import_path: str, parameters: Any = _DEFAULT_PARAMETERS) -> None:
        if not isinstance(import_path, str) or import_path.count(":") != 1:
            raise ValueError("startup hook import_path must use module:qualname syntax")
        module, qualname = import_path.split(":")
        if not _valid_import_component(module) or not _valid_import_component(qualname):
            raise ValueError("startup hook import_path must use valid module:qualname identifiers")
        parameters = {} if parameters is _DEFAULT_PARAMETERS else parameters
        if type(parameters) is not dict:
            raise ValueError("startup hook parameters must be a JSON object")
        try:
            _validate_json(parameters)
            encoded = json.dumps(parameters, allow_nan=False)
        except (RecursionError, ValueError, TypeError) as exc:
            raise ValueError("startup hook parameters must be a finite JSON object") from exc
        object.__setattr__(self, "import_path", import_path)
        object.__setattr__(self, "_parameters_json", encoded)

    @property
    def parameters(self) -> dict[str, Any]:
        return json.loads(self._parameters_json)

    def to_manifest(self) -> dict[str, Any]:
        return {"import_path": self.import_path, "parameters": self.parameters}


@dataclass(frozen=True)
class NodeStartupContext:
    """Process-local resources available before the inner worker forks its children."""

    rank: int
    artifact_roots: dict[str, Path]
    inner_worker: Worker
    chip_handles: dict[str, CallableHandle]
    parameters: dict[str, Any]


def normalize_startup_artifacts(values) -> tuple[StartupArtifactSpec, ...]:
    result = tuple(values)
    if any(not isinstance(value, StartupArtifactSpec) for value in result):
        raise TypeError("MpiL3GroupSpec.startup_artifacts entries must be StartupArtifactSpec instances")
    return result


def normalize_startup_hooks(values, world_size: int) -> tuple[NodeStartupHook | None, ...]:
    result = tuple(values) or (None,) * world_size
    if len(result) != world_size:
        raise ValueError("MpiL3GroupSpec.startup_hooks_by_rank must match hosts length")
    if any(value is not None and not isinstance(value, NodeStartupHook) for value in result):
        raise TypeError("MpiL3GroupSpec.startup_hooks_by_rank entries must be NodeStartupHook or None")
    return result


def _install_node_artifacts(descriptions) -> dict[str, Path]:
    cache = Path(os.environ.get("SIMPLER_STARTUP_ARTIFACT_CACHE", "~/.cache/simpler/startup_artifacts")).expanduser()
    installed = []
    for description in descriptions:
        digest = description.get("sha256", "unknown") if isinstance(description, dict) else "unknown"
        try:
            artifacts = install_startup_artifacts([description], cache)
            activate_startup_artifacts(artifacts)
            installed.extend(artifacts)
        except BaseException as exc:
            raise RuntimeError(f"artifact {digest}: {type(exc).__name__}: {exc}") from exc
    return {artifact.digest: artifact.root for artifact in installed}


def _invoke_node_hook(description: dict[str, Any] | None, context: NodeStartupContext) -> Any:
    if description is None:
        return None
    if not isinstance(description, dict) or set(description) != {"import_path", "parameters"}:
        raise ValueError("startup_hook manifest must contain import_path and parameters")
    hook = NodeStartupHook(description["import_path"], description["parameters"])
    module, qualname = hook.import_path.split(":")
    target = importlib.import_module(module)
    for name in qualname.split("."):
        target = getattr(target, name)
    if not callable(target):
        raise TypeError(f"startup hook {hook.import_path!r} is not callable")
    state = target(replace(context, parameters=hook.parameters))
    if state is not None and not callable(getattr(state, "close", None)):
        raise TypeError(f"startup hook {hook.import_path!r} must return None or an object with close()")
    return state


class _StartupStages:
    """One absolute startup budget, including blocking hooks and MPI collectives."""

    def __init__(self, comm: Any, rank: int, deadline: float) -> None:
        self.comm = comm
        self.rank = rank
        self.deadline = deadline
        self.stage = "startup"
        self._timer = threading.Timer(max(0.0, deadline - time.monotonic()), self._abort)
        self._timer.daemon = True
        self._timer.start()

    def _abort(self) -> None:
        print(
            f"MPI rank {self.rank} {self.stage}: startup deadline exceeded; exiting rank", file=sys.stderr, flush=True
        )
        os._exit(124)

    def run(self, stage: str, action, *, error_context: str = ""):
        self.stage = stage + error_context
        result = None
        error = ""
        try:
            self._check_deadline()
            result = action()
            self._check_deadline()
        except BaseException as exc:  # noqa: BLE001 -- all peers must reach the same stage status exchange.
            error = f"{type(exc).__name__}: {exc}"
        statuses = self.comm.allgather((self.rank, not error, error))
        self._check_deadline()
        failures = [f"rank {rank} {self.stage}: {message}" for rank, ok, message in statuses if not ok]
        if failures:
            raise RuntimeError("; ".join(failures))
        return result

    def _check_deadline(self) -> None:
        if time.monotonic() >= self.deadline:
            raise TimeoutError(f"MPI rank {self.rank} {self.stage}: startup deadline exceeded")

    def close(self) -> None:
        self._timer.cancel()
