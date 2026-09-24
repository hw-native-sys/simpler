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
    manifest = {
        "schema": "simpler-hbg-pure-artifact-v1",
        "runtime": "host_build_graph",
        "platform": "a2a3",
        "external_argument_count": 25,
        "graph_definition_task_count_per_layer": 277,
        "distributed_meta_sha256": _digest(tmp_path / "distributed_meta.json"),
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


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ('KERNELS[1]["func_id"] = 0', "unique"),
        ('KERNELS[1]["func_id"] = 100', "no verified"),
        ('ORCHESTRATION["signature"] = []', "explicit tensor directions"),
        ('RUNTIME_CONFIG["runtime"] = "tensormap_and_ringbuffer"', "chip runtime"),
    ],
)
def test_invalid_generated_config_is_rejected(bridge, artifact, extra, message):
    path = artifact / "next_levels/decode_fwd/kernel_config.py"
    path.write_text(path.read_text() + extra + "\n")
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
    path = artifact / "next_levels/decode_fwd/kernel_config.py"
    path.write_text(path.read_text() + 'KERNELS[0]["signature"] = []\n')
    assert bridge.inspect_artifact(artifact).callable_spec["incores"][0]["signature"] == []


def test_runtime_configuration_survives_bridge(bridge, artifact):
    path = artifact / "next_levels/decode_fwd/kernel_config.py"
    path.write_text(path.read_text() + 'RUNTIME_CONFIG["aicpu_thread_num"] = 6\n')
    assert bridge.inspect_artifact(artifact).runtime_config == {"runtime": "host_build_graph", "aicpu_thread_num": 6}
