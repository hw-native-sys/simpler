# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU validation of generated Qwen child loading and compilation boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.syspath_prepend(str(CASE))
    spec = importlib.util.spec_from_file_location("_qwen_artifact_bridge_tests", CASE / "callable_bridge.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def artifact(tmp_path):
    child = tmp_path / "next_levels/decode_fwd"
    (child / "orchestration").mkdir(parents=True)
    (child / "cache").mkdir()
    (child / "kernels").mkdir()
    (child / "orchestration/decode_fwd.cpp").write_text("generated source")
    (child / "orchestration/decode_fwd.so").write_bytes(b"historical orchestration")
    (child / "kernels/vector.cpp").write_text("generated vector source")
    for index in range(39):
        (child / f"cache/incore_{index}.bin").write_bytes(bytes([index]))
    names = [f"input_{index}" for index in range(20)]
    names += ["out", "embed_weight", "sampled_ids_in", "sampled_ids", "next_hidden"]
    metadata = {
        "platform": "a2a3",
        "distributed_config": {"runtime": "host_build_graph"},
        "params": [{"name": name + "__ssa_v0"} for name in names],
    }
    (tmp_path / "distributed_meta.json").write_text(json.dumps(metadata))
    (child / "kernel_config.py").write_text(
        "from pathlib import Path\nfrom simpler.task_interface import ArgDirection as D\n"
        "R = Path(__file__).parent\n"
        'RUNTIME_CONFIG = {"runtime": "host_build_graph"}\n'
        'ORCHESTRATION = {"source": str(R / "orchestration/decode_fwd.cpp"), '
        '"function_name": "entry", "signature": [D.IN, D.INOUT, D.OUT]}\n'
        'KERNELS = [{"func_id": i, "name": f"kernel_{i}", '
        '"source": str(R / "kernels/vector.cpp"), "core_type": "aiv", '
        '"signature": [D.IN, D.OUT]} for i in range(39)]\n'
    )
    callable_spec = {
        "schema": "simpler-qwen-callable-spec-v1",
        "orchestration": {
            "source": "next_levels/decode_fwd/orchestration/decode_fwd.cpp",
            "function_name": "entry",
            "signature": [
                {"__type__": "ArgDirection", "name": "IN"},
                {"__type__": "ArgDirection", "name": "INOUT"},
                {"__type__": "ArgDirection", "name": "OUT"},
            ],
        },
        "incores": [
            {
                "func_id": i,
                "name": f"kernel_{i}",
                "source": "next_levels/decode_fwd/kernels/vector.cpp",
                "core_type": "aiv",
                "signature": [
                    {"__type__": "ArgDirection", "name": "IN"},
                    {"__type__": "ArgDirection", "name": "OUT"},
                ],
            }
            for i in range(39)
        ],
        "runtime_config": {"runtime": "host_build_graph"},
    }
    callable_spec_path = tmp_path / "callable_spec.json"
    callable_spec_path.write_text(json.dumps(callable_spec))
    manifest = {
        "schema": "simpler-hbg-pure-artifact-v1",
        "runtime": "host_build_graph",
        "platform": "a2a3",
        "external_argument_count": 25,
        "graph_definition_task_count_per_layer": 277,
        "distributed_meta_sha256": _digest(tmp_path / "distributed_meta.json"),
        "callable_spec_sha256": _digest(callable_spec_path),
        "orchestration_cpp_sha256": _digest(child / "orchestration/decode_fwd.cpp"),
        "orchestration_so_sha256": _digest(child / "orchestration/decode_fwd.so"),
        "source_incore_bins": {f"incore_{i}.bin": _digest(child / f"cache/incore_{i}.bin") for i in range(39)},
    }
    (tmp_path / "hbg_artifact_manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_compile_preserves_child_abi_separately_from_host_params(bridge, artifact, monkeypatch):
    inspected = bridge.inspect_artifact(artifact)
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        importlib.import_module("simpler_setup.scene_test"),
        "compile_chip_callable_spec",
        lambda *args: calls.append(args) or sentinel,
    )
    assert inspected.compile() is sentinel
    spec, platform, runtime, _key = calls[0]
    assert len(inspected.parameter_names) == 25
    assert spec["orchestration"]["signature"] == [ArgDirection.IN, ArgDirection.INOUT, ArgDirection.OUT]
    assert spec["orchestration"]["source"].endswith("decode_fwd.cpp")
    assert spec["orchestration"]["function_name"] == "entry"
    assert [kernel["func_id"] for kernel in spec["incores"]] == list(range(39))
    assert (platform, runtime) == ("a2a3", "host_build_graph")
    assert all(not Path(name).is_absolute() for name in inspected.source_hashes)
    assert set(inspected.source_hashes) == set(inspected.source_paths)


def test_seeded_host_output_has_distinct_signature_and_cache_identity(bridge, artifact, monkeypatch):
    inspected = bridge.inspect_artifact(artifact)
    with pytest.raises(ValueError, match="matching host and child tensor counts"):
        inspected.compile(seed_outputs=("out",))
    inspected.parameter_names = ("input", "state", "out")
    calls = []
    monkeypatch.setattr(
        importlib.import_module("simpler_setup.scene_test"),
        "compile_chip_callable_spec",
        lambda *args: calls.append(args),
    )
    inspected.compile()
    inspected.compile(seed_outputs=("out",))
    assert calls[0][0]["orchestration"]["signature"][-1] == ArgDirection.OUT
    assert calls[1][0]["orchestration"]["signature"][-1] == ArgDirection.INOUT
    assert calls[0][-1] != calls[1][-1]
    assert inspected.callable_spec["orchestration"]["signature"][-1] == ArgDirection.OUT
    with pytest.raises(ValueError, match="OUT declaration"):
        inspected.compile(seed_outputs=("input",))


@pytest.mark.parametrize("relative", ["distributed_meta.json", "next_levels/decode_fwd/cache/incore_0.bin"])
def test_tampered_artifact_rejected_before_compilation(bridge, artifact, relative):
    (artifact / relative).write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        bridge.inspect_artifact(artifact)


def test_source_change_after_inspection_rejected(bridge, artifact):
    inspected = bridge.inspect_artifact(artifact)
    (artifact / "next_levels/decode_fwd/kernels/vector.cpp").write_text("changed source")
    with pytest.raises(ValueError, match="source changed"):
        inspected.compile()


def test_runtime_does_not_execute_kernel_config(bridge, artifact):
    (artifact / "next_levels/decode_fwd/kernel_config.py").write_text("raise RuntimeError('must not execute')\n")
    assert bridge.inspect_artifact(artifact).parameter_names[-1] == "next_hidden"


def test_absolute_callable_source_rejected(bridge, artifact):
    path = artifact / "callable_spec.json"
    spec = json.loads(path.read_text())
    spec["orchestration"]["source"] = str((artifact / "next_levels/decode_fwd/orchestration/decode_fwd.cpp").resolve())
    path.write_text(json.dumps(spec))
    manifest_path = artifact / "hbg_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["callable_spec_sha256"] = _digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="relative to the artifact"):
        bridge.inspect_artifact(artifact)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("incores", 1, "func_id", 0), "unique"),
        (("incores", 1, "func_id", 100), "no verified"),
        (("orchestration", None, "signature", []), "explicit tensor directions"),
        (("runtime_config", None, "runtime", "tensormap_and_ringbuffer"), "chip runtime"),
    ],
)
def test_invalid_generated_config_is_rejected(bridge, artifact, extra, message):
    path = artifact / "callable_spec.json"
    spec = json.loads(path.read_text())
    section, index, key, value = extra
    target = spec[section] if index is None else spec[section][index]
    target[key] = value
    path.write_text(json.dumps(spec))
    manifest_path = artifact / "hbg_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["callable_spec_sha256"] = _digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        bridge.inspect_artifact(artifact)


def test_host_output_abi_is_preserved(bridge, artifact):
    metadata_path = artifact / "distributed_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["params"].append({"name": "sampled_ids_host__ssa_v0"})
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = artifact / "hbg_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(external_argument_count=26, distributed_meta_sha256=_digest(metadata_path))
    manifest_path.write_text(json.dumps(manifest))
    assert bridge.inspect_artifact(artifact).parameter_names[-1] == "sampled_ids_host"


def test_explicit_empty_kernel_signature_is_preserved(bridge, artifact):
    path = artifact / "callable_spec.json"
    spec = json.loads(path.read_text())
    spec["incores"][0]["signature"] = []
    path.write_text(json.dumps(spec))
    manifest_path = artifact / "hbg_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["callable_spec_sha256"] = _digest(path)
    manifest_path.write_text(json.dumps(manifest))
    assert bridge.inspect_artifact(artifact).callable_spec["incores"][0]["signature"] == []


def test_runtime_configuration_survives_bridge(bridge, artifact):
    path = artifact / "callable_spec.json"
    spec = json.loads(path.read_text())
    spec["runtime_config"]["aicpu_thread_num"] = 6
    path.write_text(json.dumps(spec))
    manifest_path = artifact / "hbg_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["callable_spec_sha256"] = _digest(path)
    manifest_path.write_text(json.dumps(manifest))
    assert bridge.inspect_artifact(artifact).runtime_config == {"runtime": "host_build_graph", "aicpu_thread_num": 6}


@pytest.fixture
def stage0_release(artifact):
    (artifact / "hbg_artifact_manifest.json").unlink()
    child = artifact / "next_levels/decode_fwd"
    (child / "prebuilt").mkdir()
    for index in range(32):
        (child / f"prebuilt/kernel_{index}.bin").write_bytes((child / f"cache/incore_{index}.bin").read_bytes())
    (child / "prebuilt/orchestration.bin").write_bytes((child / "orchestration/decode_fwd.so").read_bytes())
    metadata_path = artifact / "distributed_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema"] = 3
    for index, param in enumerate(metadata["params"]):
        param["direction"] = "Out" if index in (20, 23, 24) else "In"
    for index, name in ((11, "k_cache"), (12, "v_cache")):
        metadata["params"][index].update(name=name + "__ssa_v0", direction="InOut")
    metadata_path.write_text(json.dumps(metadata))
    signature = "[D.IN] * 11 + [D.INOUT, D.INOUT] + [D.IN] * 7 + [D.OUT, D.IN, D.IN, D.OUT, D.OUT]"
    (child / "kernel_config.py").write_text(
        "from pathlib import Path\nfrom simpler.task_interface import ArgDirection as D\n"
        "R = Path(__file__).parent\n"
        'RUNTIME_CONFIG = {"runtime": "host_build_graph"}\n'
        'ORCHESTRATION = {"source": str(R / "orchestration/decode_fwd.cpp"), '
        f'"function_name": "entry", "signature": {signature}}}\n'
        'KERNELS = [{"func_id": i, "name": f"kernel_{i}", '
        '"source": str(R / "kernels/vector.cpp"), "core_type": "aiv", '
        '"signature": [D.IN, D.OUT]} for i in range(32)]\n'
    )
    registrations = [
        {
            "func_id": index,
            "name": f"kernel_{index}",
            "signature": ["IN", "OUT"],
            "binary": {
                "path": f"prebuilt/kernel_{index}.bin",
                "sha256": _digest(child / f"cache/incore_{index}.bin"),
                "size": 1,
            },
        }
        for index in range(32)
    ]
    (child / "binary_manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "platform": "a2a3",
                "runtime_name": "host_build_graph",
                "kernels": registrations,
                "orchestration": {
                    "binary": {
                        "path": "prebuilt/orchestration.bin",
                        "sha256": _digest(child / "orchestration/decode_fwd.so"),
                        "size": 24,
                    }
                },
            }
        )
    )
    (child / "cache/binary_context.json").write_text(
        json.dumps(
            {
                "context": {"runtime_name": "host_build_graph"},
                "files": {"orchestration/decode_fwd.so": _digest(child / "orchestration/decode_fwd.so")},
            }
        )
    )
    return artifact


def test_stage0_release_preserves_32_registrations_and_kv_inout(bridge, stage0_release):
    inspected = bridge.inspect_artifact(stage0_release)
    assert len(inspected.callable_spec["incores"]) == 32
    assert inspected.callable_spec["orchestration"]["signature"][11:13] == [ArgDirection.INOUT] * 2


@pytest.mark.parametrize("count", [31, 33])
def test_stage0_release_rejects_incorrect_registration_count(bridge, stage0_release, count):
    path = stage0_release / "next_levels/decode_fwd/binary_manifest.json"
    binary = json.loads(path.read_text())
    binary["kernels"] = (binary["kernels"] + binary["kernels"][:1])[:count]
    path.write_text(json.dumps(binary))
    with pytest.raises(ValueError, match="exactly 32"):
        bridge.inspect_artifact(stage0_release)


def test_stage0_release_rejects_metadata_direction_mismatch(bridge, stage0_release):
    path = stage0_release / "distributed_meta.json"
    metadata = json.loads(path.read_text())
    metadata["params"][11]["direction"] = "In"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="direction"):
        bridge.inspect_artifact(stage0_release)


def test_release_checks_manifest_named_binary(bridge, stage0_release):
    (stage0_release / "next_levels/decode_fwd/prebuilt/kernel_0.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        bridge.inspect_artifact(stage0_release)


@pytest.mark.parametrize("name", ["kernel_config.py", "kernels/vector.cpp"])
def test_stage0_compile_rejects_changed_sources(bridge, stage0_release, name):
    inspected = bridge.inspect_artifact(stage0_release)
    path = stage0_release / "next_levels/decode_fwd" / name
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="source changed"):
        inspected.compile()


def test_stage0_config_cannot_escape_the_release(bridge, stage0_release, tmp_path):
    config = stage0_release / "next_levels/decode_fwd/kernel_config.py"
    external = tmp_path.parent / f"{tmp_path.name}-external-config.py"
    external.write_text("raise RuntimeError('must not execute')")
    config.unlink()
    config.symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        bridge.inspect_artifact(stage0_release)
