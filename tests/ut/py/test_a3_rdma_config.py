# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""RDMA resource configuration is immutable and independent of task payloads."""

from dataclasses import FrozenInstanceError

import pytest
from simpler.comm_domain import DomainConfig
from simpler.worker import Worker
from simpler.workspace_config import RdmaConfig, normalize_workspace_configs


def test_defaults_match_name_only_workspace_selection():
    assert normalize_workspace_configs(("rdma",)) == (("rdma", RdmaConfig()),)
    assert RdmaConfig().qp_num == 1
    assert RdmaConfig().base_port is None
    assert RdmaConfig().roce_ips == ()


def test_resource_configuration_is_canonical_and_immutable():
    config = RdmaConfig(qp_num=2, base_port=61000, roce_ips=("192.0.2.1", "2001:0db8::1"))
    assert config.roce_ips == ("192.0.2.1", "2001:db8::1")
    assert config != RdmaConfig(qp_num=1, base_port=61000, roce_ips=config.roce_ips)
    with pytest.raises(FrozenInstanceError):
        config.qp_num = 3


@pytest.mark.parametrize("qp_num", [0, -1, 9])
def test_qp_budget_is_validated(qp_num):
    with pytest.raises(ValueError):
        RdmaConfig(qp_num=qp_num)


@pytest.mark.parametrize(
    "field,value", [("qp_num", True), ("qp_num", 1.5), ("base_port", False), ("base_port", "61000")]
)
def test_integer_fields_do_not_coerce_bool_float_or_text(field, value):
    with pytest.raises(TypeError):
        RdmaConfig(**{field: value})


@pytest.mark.parametrize("base_port", [0, -1, 65536])
def test_port_is_a_nonzero_uint16(base_port):
    with pytest.raises(ValueError):
        RdmaConfig(base_port=base_port)


@pytest.mark.parametrize("roce_ips", [("",), ("hostname",), ("192.0.2.1", "192.0.2.1")])
def test_addresses_must_be_distinct_ip_literals(roce_ips):
    with pytest.raises(ValueError):
        RdmaConfig(roce_ips=roce_ips)


def test_roce_addresses_are_a_sequence_not_a_single_string():
    with pytest.raises(TypeError):
        RdmaConfig(roce_ips="192.0.2.1")  # pyright: ignore[reportArgumentType] -- exercise runtime validation


def test_ordered_device_configuration_is_part_of_domain_identity():
    rdma = RdmaConfig(qp_num=2, roce_ips=("192.0.2.2", "192.0.2.1"))
    worker = Worker(level=3, platform="a2a3", runtime="tensormap_and_ringbuffer", device_ids=[0, 1], num_sub_workers=0)
    config = worker._domain_config([1, 0], 4096, [], True, {"rdma": rdma})
    assert config.members == (1, 0)
    assert config.rdma_config() == rdma
    assert config.rdma_config().base_port is None
    other = DomainConfig.capture([1, 0], 4096, [], True, config.topology, {"rdma": RdmaConfig()})
    assert config.difference(other).startswith("workspace_configs:")


@pytest.mark.parametrize(
    "platform,runtime,message",
    [
        ("a2a3sim", "tensormap_and_ringbuffer", "DEVICE domains"),
        ("a2a3", "host_build_graph", "tensormap_and_ringbuffer"),
        ("a5", "tensormap_and_ringbuffer", "non-default RdmaConfig"),
    ],
)
def test_in_die_configuration_does_not_leak_into_other_backends(platform, runtime, message):
    worker = Worker(level=3, platform=platform, runtime=runtime, device_ids=[0, 1], num_sub_workers=0)
    with pytest.raises(ValueError, match=message):
        worker._domain_config([0, 1], 4096, [], True, {"rdma": RdmaConfig(qp_num=2)})


def test_partial_roce_address_list_is_rejected_before_control():
    worker = Worker(level=3, platform="a2a3", runtime="tensormap_and_ringbuffer", device_ids=[0, 1], num_sub_workers=0)
    with pytest.raises(ValueError, match="one device address per domain member"):
        worker._domain_config([0, 1], 4096, [], True, {"rdma": RdmaConfig(roce_ips=("192.0.2.1",))})


@pytest.mark.parametrize("address", ["fe80::1%" + "x" * 56, "fe80::1%\0", "fe80::1%\u7f51\u5361"])
def test_invalid_wire_address_does_not_poison_worker_configuration(address):
    worker = Worker(level=3, platform="a2a3", runtime="tensormap_and_ringbuffer", device_ids=[0, 1], num_sub_workers=0)
    with pytest.raises(ValueError, match="ASCII|64|NUL"):
        worker._domain_config([0, 1], 4096, [], True, {"rdma": RdmaConfig(roce_ips=(address, "192.0.2.2"))})
    valid = RdmaConfig(roce_ips=("192.0.2.1", "192.0.2.2"))
    assert worker._domain_config([0, 1], 4096, [], True, {"rdma": valid}).rdma_config() == valid
