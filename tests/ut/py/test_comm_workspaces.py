# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Communication domain configuration and workspace selection."""

import subprocess
import sys
from pathlib import Path

import pytest
from simpler import CcuConfig, RdmaConfig, SdmaConfig, UrmaConfig
from simpler.buffer import AddressSpace
from simpler.comm_domain import DomainConfig

ROOT = Path(__file__).resolve().parents[3]


def test_ccu_workspace_mapping_uses_same_configuration_as_name_sequence():
    named = DomainConfig.capture([0, 1], 4096, [], True, (), ("ccu",))
    mapped = DomainConfig.capture([0, 1], 4096, [], True, (), {"ccu": CcuConfig()})
    assert named == mapped
    assert mapped.workspace_mask() == 8
    assert mapped.ccu_config().mode == "sched"


def test_ccu_mode_is_immutable_and_part_of_domain_reuse_identity():
    engines = {"ccu": CcuConfig(mode="ms")}
    saved = DomainConfig.capture([0, 1], 4096, [], True, (), engines)
    engines["ccu"] = CcuConfig()
    replacement = DomainConfig.capture([0, 1], 4096, [], True, (), engines)
    assert saved.ccu_config().mode == "ms"
    assert saved.difference(replacement).startswith("workspace_configs:")
    with pytest.raises(AttributeError):
        saved.ccu_config().mode = "sched"


@pytest.mark.parametrize("mode", ["", "MS", "CCU_SCHED", "auto"])
def test_invalid_ccu_mode_is_rejected_before_allocation(mode):
    with pytest.raises(ValueError, match="CCU mode"):
        CcuConfig(mode=mode)


@pytest.mark.parametrize("mode", [None, 5, 6, True, []])
def test_ccu_mode_requires_a_string(mode):
    with pytest.raises(TypeError, match="CCU mode"):
        CcuConfig(mode=mode)


@pytest.mark.parametrize(
    "engines,mask", [((), 0), ([], 0), (["sdma"], 1), (["urma"], 2), (["ccu", "sdma"], 9), (["rdma"], 4), (["ccu"], 8)]
)
def test_domain_workspace_selection_is_part_of_configuration(engines, mask):
    config = DomainConfig.capture([0, 1], 4096, [], True, (), engines)
    assert config.workspace_mask() == mask
    if engines:
        reordered = DomainConfig.capture([0, 1], 4096, [], True, (), list(reversed(engines)))
        assert not config.difference(reordered)
    synchronous = DomainConfig.capture([0, 1], 4096, [], True, ())
    assert synchronous.workspace_engines == () and synchronous.workspace_mask() == 0
    if engines:
        assert config.difference(synchronous).startswith("workspace_engines:")
    else:
        assert not config.difference(synchronous)


@pytest.mark.parametrize(
    "engines",
    [
        ["unknown"],
        ["sdma", "sdma"],
        ["rdma", "sdma"],
        ["rdma", "urma"],
        ["sdma", "urma"],
        ["urma", "ccu"],
        ["mte", "urma"],
    ],
)
def test_invalid_workspace_requests_are_rejected_before_allocation(engines):
    with pytest.raises(ValueError):
        DomainConfig.capture([0], 4096, [], False, (), engines)


@pytest.mark.parametrize("engines", ["sdma", None])
def test_workspace_names_require_a_sequence(engines):
    with pytest.raises(TypeError, match="sequence"):
        DomainConfig.capture([0], 4096, [], False, (), engines)


@pytest.mark.parametrize("kind", ["CommWorkspaceKind::COUNT", "static_cast<CommWorkspaceKind>(UINT32_MAX)"])
def test_invalid_static_workspace_kind_is_rejected_by_compiler(tmp_path, kind):
    source = tmp_path / "invalid_workspace.cpp"
    source.write_text(
        '#include "platform_comm/comm_context.h"\n'
        f"int main() {{ CommContext ctx{{}}; return comm_workspace<{kind}>(ctx).addr; }}\n"
    )
    result = subprocess.run(
        ["c++", "-std=c++17", f"-I{ROOT / 'src/common'}", "-c", str(source), "-o", str(tmp_path / "invalid.o")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid communication workspace kind" in result.stderr


@pytest.mark.parametrize(
    "platform,runtime,message",
    [
        ("a2a3sim", "tensormap_and_ringbuffer", "requires A5"),
        ("a5sim", "tensormap_and_ringbuffer", "requires A5"),
        ("a5sim", "host_build_graph", "requires A5"),
    ],
)
def test_unsupported_urma_domains_are_rejected_before_control(platform, runtime, message):
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, platform=platform, runtime=runtime, device_ids=[0, 1], num_sub_workers=0)
    with pytest.raises(ValueError, match=message):
        worker._domain_config([0, 1], 4096, [], False, ("urma",))


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize("members", [(1, 2), (2, 1)])
def test_device_urma_configuration_preserves_ordered_members(members, runtime):
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, platform="a5", runtime=runtime, device_ids=[1, 2, 3], num_sub_workers=0)
    config = worker._domain_config(members, 4096, [], False, ("urma",))
    assert config.members == members
    assert config.workspace_engines == ("urma",)


def test_host_urma_configuration_keeps_dense_prefix_requirement():
    from simpler.task_interface import AddressSpace  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, platform="a5", runtime="tensormap_and_ringbuffer", device_ids=[0, 1], num_sub_workers=0)
    with pytest.raises(ValueError, match="dense prefix"):
        worker._domain_config([1, 0], 4096, [], False, ("urma",), AddressSpace.HOST)


def test_single_rank_urma_configuration_is_delegated_to_provider():
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, platform="a5", runtime="tensormap_and_ringbuffer", device_ids=[0], num_sub_workers=0)
    config = worker._domain_config([0], 4096, [], False, ("urma",))
    assert config.members == (0,)
    assert config.workspace_engines == ("urma",)


@pytest.mark.parametrize(
    "names,configs",
    [
        (("sdma",), {"sdma": SdmaConfig()}),
        (("urma",), {"urma": UrmaConfig()}),
        (("rdma",), {"rdma": RdmaConfig()}),
        (("ccu", "sdma"), {"sdma": SdmaConfig(), "ccu": CcuConfig()}),
    ],
)
def test_default_engine_configurations_equal_legacy_names(names, configs):
    legacy = DomainConfig.capture([0, 1], 4096, [], True, (), names)
    configured = DomainConfig.capture([0, 1], 4096, [], True, (), configs)
    assert configured == legacy
    assert configured.difference(legacy) == ""


@pytest.mark.parametrize("memory_source", [AddressSpace.DEVICE, AddressSpace.HOST])
def test_engine_configuration_is_snapshotted_and_changes_reuse_identity(memory_source):
    engines = {"urma": UrmaConfig(jetties_per_aiv=2, sq_depth=512)}
    saved = DomainConfig.capture([0, 1], 4096, [], True, (), engines, memory_source)
    engines["urma"] = UrmaConfig()
    replacement = DomainConfig.capture([0, 1], 4096, [], True, (), engines, memory_source)
    assert saved.urma_config() == UrmaConfig(jetties_per_aiv=2, sq_depth=512)
    assert saved.difference(replacement).startswith("workspace_configs:")
    with pytest.raises(AttributeError):
        saved.urma_config().sq_depth = 1024


@pytest.mark.parametrize(
    "engines",
    [
        {"urma": SdmaConfig()},
        {"sdma": UrmaConfig()},
        {"rdma": None},
        {"urma": {"jetties_per_aiv": 2}},
    ],
)
def test_engine_configuration_type_must_match_name(engines):
    with pytest.raises(TypeError, match="requires"):
        DomainConfig.capture([0, 1], 4096, [], True, (), engines)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"jetties_per_aiv": 0},
        {"jetties_per_aiv": 9},
        {"jetties_per_aiv": -1},
        {"sq_depth": -1},
        {"sq_depth": 3},
        {"sq_depth": 2**31},
    ],
)
def test_invalid_urma_queue_sizes_fail_before_domain_allocation(kwargs):
    with pytest.raises(ValueError):
        UrmaConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"jetties_per_aiv": True}, {"jetties_per_aiv": 2.5}, {"sq_depth": "512"}])
def test_urma_queue_sizes_require_integers(kwargs):
    with pytest.raises(TypeError):
        UrmaConfig(**kwargs)


def test_engine_configs_are_available_when_native_bindings_are_missing():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['_task_interface'] = None; "
            "from simpler import UrmaConfig, SdmaConfig, RdmaConfig; "
            "assert UrmaConfig().jetties_per_aiv == 1; SdmaConfig(); RdmaConfig(); "
            "assert sys.modules['_task_interface'] is None",
        ],
        check=True,
    )
