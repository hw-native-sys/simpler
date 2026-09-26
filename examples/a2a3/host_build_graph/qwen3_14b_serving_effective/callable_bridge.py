# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile a verified generated Qwen HBG child with the selected Simpler tree.

The generated kernel_config.py is trusted executable Python. Distributed host
parameters and chip parameters are distinct ABIs; callers retain the generated
host wrapper when binding a request to the returned child callable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build_hbg_artifact import _validate_param_names, verify_hbg_artifact


@dataclass
class DecodeArtifact:
    root: Path
    parameter_names: tuple[str, ...]
    callable_spec: dict[str, Any]
    source_hashes: dict[str, str]
    runtime_config: dict[str, Any]

    def compile(self):
        """Recompile generated sources using current runtime headers and tools."""
        from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key  # noqa: PLC0415

        for filename, digest in self.source_hashes.items():
            if _sha256(Path(filename)) != digest:
                raise ValueError(f"artifact source changed after inspection: {filename}")
        identity = hashlib.sha256(json.dumps(self.source_hashes, sort_keys=True).encode()).hexdigest()
        key = l3_compile_cache_key("qwen_hbg_artifact", identity, "decode_fwd", "a2a3", "host_build_graph")
        return compile_chip_callable_spec(self.callable_spec, "a2a3", "host_build_graph", key)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(value: str, child: Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else child / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"expected source file: {path}")
    return path


def inspect_artifact(root: Path) -> DecodeArtifact:
    """Validate frozen binaries and preserve the generated child configuration.

    Binary checksums establish artifact identity, not current-runtime ABI
    compatibility. Compilation consumes generated sources rather than loading
    a historical orchestration shared library into the current process.
    """
    root = root.resolve(strict=True)
    manifest = verify_hbg_artifact(root)
    if manifest.get("platform") != "a2a3":
        raise ValueError("Qwen bridge requires an a2a3 artifact")
    metadata_path = root / "distributed_meta.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("platform") != "a2a3":
        raise ValueError("distributed metadata platform must be a2a3")
    if metadata.get("distributed_config", {}).get("runtime") != "host_build_graph":
        raise ValueError("distributed metadata runtime must be host_build_graph")
    names = [parameter["name"].split("__ssa_", 1)[0] for parameter in metadata["params"]]
    _validate_param_names(names)
    if len(names) != len(set(names)):
        raise ValueError("duplicate normalized distributed parameter name")
    if manifest.get("external_argument_count") != len(names):
        raise ValueError("manifest and distributed argument count differ")

    child = root / "next_levels" / "decode_fwd"
    config_path = child / "kernel_config.py"
    config_hash = _sha256(config_path)
    config = runpy.run_path(str(config_path))
    if _sha256(config_path) != config_hash:
        raise ValueError("kernel configuration changed during inspection")
    if config.get("RUNTIME_CONFIG", {}).get("runtime") != "host_build_graph":
        raise ValueError("chip runtime must be host_build_graph")
    orchestration = copy.deepcopy(config["ORCHESTRATION"])
    orchestration_source = _source_path(orchestration["source"], child)
    if orchestration_source != (child / "orchestration" / "decode_fwd.cpp").resolve():
        raise ValueError("chip orchestration differs from the verified source")
    if not orchestration.get("function_name"):
        raise ValueError("missing orchestration entry symbol")
    orchestration["source"] = str(orchestration_source)
    kernels = copy.deepcopy(config["KERNELS"])
    if len(kernels) != len(manifest["source_incore_bins"]):
        raise ValueError("kernel configuration and verified binary counts differ")
    hashes = {
        str(root / "hbg_artifact_manifest.json"): _sha256(root / "hbg_artifact_manifest.json"),
        str(metadata_path): _sha256(metadata_path),
        str(config_path): config_hash,
        str(orchestration_source): _sha256(orchestration_source),
    }
    identifiers = set()
    for kernel in kernels:
        identifier = kernel["func_id"]
        if type(identifier) is not int or identifier < 0 or identifier in identifiers:
            raise ValueError("kernel function IDs must be unique non-negative integers")
        identifiers.add(identifier)
        if not any(
            name == f"incore_{identifier}.bin" or name.startswith(f"incore_{identifier}_")
            for name in manifest["source_incore_bins"]
        ):
            raise ValueError("kernel function ID has no verified in-core binary")
        if kernel.get("core_type") not in ("aic", "aiv"):
            raise ValueError("unsupported kernel core type")
        source = _source_path(kernel["source"], child)
        kernel["source"] = str(source)
        hashes[str(source)] = _sha256(source)
    _validate_signatures(orchestration, kernels)

    return DecodeArtifact(
        root,
        tuple(names),
        {"name": "decode_fwd", "orchestration": orchestration, "incores": kernels},
        hashes,
        copy.deepcopy(config["RUNTIME_CONFIG"]),
    )


def _validate_signatures(orchestration, kernels):
    from simpler.task_interface import ArgDirection  # noqa: PLC0415

    for entry in (orchestration, *kernels):
        signature = entry.get("signature")
        if not isinstance(signature, list) or (entry is orchestration and not signature):
            raise ValueError("explicit tensor directions are required for every callable")
        if any(direction not in (ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT) for direction in signature):
            raise ValueError("invalid tensor direction")
