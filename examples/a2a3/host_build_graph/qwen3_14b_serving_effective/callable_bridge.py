# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile a verified declarative Qwen HBG callable with the selected runtime."""

from __future__ import annotations

import hashlib
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from build_hbg_artifact import _sha256, _validate_param_names, verify_hbg_artifact


@dataclass
class DecodeArtifact:
    root: Path
    parameter_names: tuple[str, ...]
    callable_spec: dict[str, Any]
    source_hashes: dict[str, str]
    source_paths: dict[str, Path]
    runtime_config: dict[str, Any]

    def compile(self, *, seed_outputs=()):
        """Recompile generated sources using current runtime headers and tools."""
        from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key  # noqa: PLC0415

        for filename, digest in self.source_hashes.items():
            if _sha256(self.source_paths[filename]) != digest:
                raise ValueError(f"artifact source changed after inspection: {filename}")
        identity = hashlib.sha256(json.dumps(self.source_hashes, sort_keys=True).encode()).hexdigest()
        spec = self.callable_spec
        if seed_outputs:
            from simpler.task_interface import ArgDirection  # noqa: PLC0415

            signature = list(spec["orchestration"]["signature"])
            if len(signature) != len(self.parameter_names):
                raise ValueError("seeded outputs require matching host and child tensor counts")
            for name in seed_outputs:
                index = self.parameter_names.index(name)
                if index >= len(signature) or signature[index] != ArgDirection.OUT:
                    raise ValueError(f"seeded output {name} must have an OUT declaration")
                signature[index] = ArgDirection.INOUT
            spec = {**spec, "orchestration": {**spec["orchestration"], "signature": signature}}
            identity = hashlib.sha256(json.dumps([identity, list(seed_outputs)]).encode()).hexdigest()
        key = l3_compile_cache_key("qwen_hbg_artifact", identity, "decode_fwd", "a2a3", "host_build_graph")
        return compile_chip_callable_spec(spec, "a2a3", "host_build_graph", key)


def _decode_callable_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__type__") == "ArgDirection":
        from simpler.task_interface import ArgDirection  # noqa: PLC0415

        try:
            return ArgDirection[value["name"]]
        except (KeyError, TypeError) as error:
            raise ValueError("invalid callable tensor direction") from error
    if isinstance(value, dict):
        return {key: _decode_callable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_callable_value(item) for item in value]
    return value


def _source_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("callable source paths must be relative to the artifact")
    path = root / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("callable source escapes artifact root")
    if not path.is_file():
        raise ValueError(f"expected callable source file: {path.name}")
    return path


def _release_path(value: str | Path, root: Path) -> Path:
    """Resolve a generated release path while keeping it inside the artifact."""
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("release artifact path escapes or is missing")
    return path


def _release_source(value: str | Path, root: Path) -> tuple[str, Path]:
    path = _release_path(value, root)
    return str(path.relative_to(root)).replace("\\", "/"), path


def _release_binary(binary: dict[str, Any], root: Path, child: Path) -> tuple[str, Path]:
    name, path = _release_source(child / binary["path"], root)
    if _sha256(path) != binary.get("sha256") or path.stat().st_size != binary.get("size"):
        raise ValueError(f"release binary checksum mismatch: {name}")
    return name, path


def _validate_release_configuration(metadata, names, orchestration, kernels, registrations):
    from simpler.task_interface import ArgDirection  # noqa: PLC0415

    if len(kernels) != 32 or len(registrations) != 32:
        raise ValueError("stage-0 release must contain exactly 32 kernel registrations")
    if [entry.get("func_id") for entry in registrations] != list(range(32)):
        raise ValueError("stage-0 registration IDs are not contiguous")
    if len(orchestration.get("signature", [])) != len(names):
        raise ValueError("orchestration signature does not match the 25-argument ABI")
    directions = {"In": ArgDirection.IN, "Out": ArgDirection.OUT, "InOut": ArgDirection.INOUT}
    try:
        declared = [directions[param["direction"]] for param in metadata["params"]]
    except (KeyError, TypeError) as error:
        raise ValueError("invalid distributed tensor direction") from error
    if orchestration["signature"] != declared:
        raise ValueError("distributed and callable tensor directions differ")
    for name in ("k_cache", "v_cache"):
        if name not in names or declared[names.index(name)] != ArgDirection.INOUT:
            raise ValueError("KV cache direction must be INOUT")


def _inspect_stage0_release(root: Path) -> DecodeArtifact:
    """Read the stage-0 emitted release without assuming the old pure-artifact schema."""
    metadata_path = root / "distributed_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("platform") != "a2a3":
        raise ValueError("Qwen bridge requires an a2a3 artifact")
    names = [param["name"].split("__ssa_", 1)[0] for param in metadata.get("params", [])]
    _validate_param_names(names)
    if len(names) != len(set(names)):
        raise ValueError("duplicate normalized distributed parameter name")
    if metadata.get("schema") != 3 or len(names) != 25:
        raise ValueError("unsupported stage-0 distributed metadata")

    child = root / "next_levels" / "decode_fwd"
    binary_path = child / "binary_manifest.json"
    context_path = child / "cache" / "binary_context.json"
    config_path = _release_path(child / "kernel_config.py", root)
    binary = json.loads(binary_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if binary.get("schema") != 1 or binary.get("platform") != "a2a3":
        raise ValueError("unsupported stage-0 binary manifest")
    if binary.get("runtime_name") != "host_build_graph":
        raise ValueError("stage-0 binary runtime must be host_build_graph")
    if context.get("context", {}).get("runtime_name") != "host_build_graph":
        raise ValueError("stage-0 binary context must be host_build_graph")

    config = runpy.run_path(str(config_path))
    runtime_config = dict(config.get("RUNTIME_CONFIG", {}))
    if runtime_config.get("runtime") != "host_build_graph":
        raise ValueError("stage-0 kernel config runtime must be host_build_graph")
    orchestration = dict(config["ORCHESTRATION"])
    kernels = [dict(kernel) for kernel in config["KERNELS"]]
    registrations = binary.get("kernels", [])
    _validate_release_configuration(metadata, names, orchestration, kernels, registrations)

    paths: dict[str, Path] = {}
    for rel, path in (
        ("distributed_meta.json", metadata_path),
        ("next_levels/decode_fwd/binary_manifest.json", binary_path),
        ("next_levels/decode_fwd/cache/binary_context.json", context_path),
        ("next_levels/decode_fwd/kernel_config.py", config_path),
    ):
        paths[rel] = path
    orchestration_name, orchestration_source = _release_source(orchestration["source"], root)
    expected_orchestration = (child / "orchestration" / "decode_fwd.cpp").resolve()
    if orchestration_source != expected_orchestration:
        raise ValueError("stage-0 orchestration source differs from the release path")
    orchestration["source"] = str(orchestration_source)
    paths[orchestration_name] = orchestration_source

    for kernel, registration in zip(kernels, registrations):
        if kernel.get("func_id") != registration.get("func_id") or kernel.get("name") != registration.get("name"):
            raise ValueError("kernel config and registration mapping differ")
        signature = [direction.name for direction in kernel.get("signature", [])]
        if signature != registration.get("signature"):
            raise ValueError(f"kernel signature differs for {kernel.get('name')}")
        name, source = _release_source(kernel["source"], root)
        kernel["source"] = str(source)
        paths[name] = source
        binary_info = registration.get("binary", {})
        binary_name, binary_file = _release_binary(binary_info, root, child)
        paths[binary_name] = binary_file

    orchestration_bin = binary.get("orchestration", {}).get("binary", {})
    orch_bin_name, orch_bin = _release_binary(orchestration_bin, root, child)
    paths[orch_bin_name] = orch_bin
    orchestration_so = child / "orchestration" / "decode_fwd.so"
    paths["next_levels/decode_fwd/orchestration/decode_fwd.so"] = orchestration_so
    context_hash = context.get("files", {}).get("orchestration/decode_fwd.so")
    if context_hash is not None and _sha256(orchestration_so) != context_hash:
        raise ValueError("stage-0 orchestration shared-library checksum mismatch")

    _validate_signatures(orchestration, kernels)
    return DecodeArtifact(
        root,
        tuple(names),
        {"name": "decode_fwd", "orchestration": orchestration, "incores": kernels},
        {name: _sha256(path) for name, path in paths.items()},
        paths,
        runtime_config,
    )


def inspect_artifact(root: Path) -> DecodeArtifact:
    """Validate frozen binaries and preserve the generated child configuration.

    Binary checksums establish artifact identity, not current-runtime ABI
    compatibility. Compilation consumes generated sources rather than loading
    a historical orchestration shared library into the current process.
    """
    root = root.resolve(strict=True)
    if not (root / "hbg_artifact_manifest.json").exists():
        return _inspect_stage0_release(root)
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
    spec_path = root / "callable_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != "simpler-qwen-callable-spec-v1":
        raise ValueError("unsupported Qwen callable spec")
    config = cast(dict[str, Any], _decode_callable_value(spec))
    if config.get("runtime_config", {}).get("runtime") != "host_build_graph":
        raise ValueError("chip runtime must be host_build_graph")
    orchestration = config["orchestration"]
    orchestration_source = _source_path(orchestration["source"], root)
    if orchestration_source != (child / "orchestration" / "decode_fwd.cpp").resolve():
        raise ValueError("chip orchestration differs from the verified source")
    if not orchestration.get("function_name"):
        raise ValueError("missing orchestration entry symbol")
    orchestration["source"] = str(orchestration_source)
    kernels = config["incores"]
    if len(kernels) != len(manifest["source_incore_bins"]):
        raise ValueError("kernel configuration and verified binary counts differ")
    if not kernels:
        raise ValueError("Qwen artifact requires in-core kernels")
    paths = {
        "hbg_artifact_manifest.json": root / "hbg_artifact_manifest.json",
        "distributed_meta.json": metadata_path,
        "callable_spec.json": spec_path,
        "next_levels/decode_fwd/orchestration/decode_fwd.cpp": orchestration_source,
    }
    paths.update(_kernel_sources(kernels, manifest, root))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    _validate_signatures(orchestration, kernels)

    return DecodeArtifact(
        root,
        tuple(names),
        {"name": "decode_fwd", "orchestration": orchestration, "incores": kernels},
        hashes,
        paths,
        config["runtime_config"],
    )


def _kernel_sources(kernels, manifest, root):
    identifiers = set()
    paths = {}
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
        source = _source_path(kernel["source"], root)
        kernel["source"] = str(source)
        name = str(source.relative_to(root)).replace("\\", "/")
        paths[name] = source
    return paths


def _validate_signatures(orchestration, kernels):
    from simpler.task_interface import ArgDirection  # noqa: PLC0415

    for entry in (orchestration, *kernels):
        signature = entry.get("signature")
        if not isinstance(signature, list) or (entry is orchestration and not signature):
            raise ValueError("explicit tensor directions are required for every callable")
        if any(direction not in (ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT) for direction in signature):
            raise ValueError("invalid tensor direction")
