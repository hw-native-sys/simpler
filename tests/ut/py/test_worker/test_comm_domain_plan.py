# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest


def _buffer(name: str = "scratch"):
    from simpler.task_interface import ChipBufferSpec

    return ChipBufferSpec(name=name, dtype="float32", count=4, nbytes=16)


class TestCommDomainPlan:
    def test_bootstrap_for_worker_uses_worker_index_order_as_domain_rank(self):
        from simpler.task_interface import CommDomain, CommDomainPlan

        plan = CommDomainPlan(
            domains=[
                CommDomain(name="tp", worker_indices=[3, 1], window_size=4096, buffers=[_buffer()]),
                CommDomain(name="ep", worker_indices=[1, 2], window_size=8192, buffers=[_buffer()]),
            ]
        )
        plan.validate(worker_count=4)

        cfgs = plan.bootstrap_for_worker(1)

        assert [c.name for c in cfgs] == ["ep", "tp"]
        assert {c.name: c.domain_rank for c in cfgs} == {"tp": 1, "ep": 0}
        assert {c.name: c.domain_size for c in cfgs} == {"tp": 2, "ep": 2}
        assert {c.name: c.rank_ids for c in cfgs} == {"tp": [3, 1], "ep": [1, 2]}

    def test_bootstrap_for_worker_assigns_shared_base_window_layout(self):
        from simpler.task_interface import CommDomain, CommDomainPlan

        plan = CommDomainPlan(
            domains=[
                CommDomain(name="tp", worker_indices=[0, 1], window_size=4096, buffers=[_buffer()]),
                CommDomain(name="pp", worker_indices=[1, 2], window_size=8192, buffers=[_buffer()]),
                CommDomain(name="dp", worker_indices=[2], window_size=2048, buffers=[_buffer()]),
            ]
        )
        plan.validate(worker_count=3)

        cfgs = {c.name: c for c in plan.bootstrap_for_worker(1)}

        assert cfgs["pp"].window_offset == 2048
        assert cfgs["tp"].window_offset == 10240
        assert cfgs["pp"].base_window_size == 14336
        assert cfgs["tp"].base_window_size == 14336

    def test_non_member_worker_gets_no_domain_configs(self):
        from simpler.task_interface import CommDomain, CommDomainPlan

        plan = CommDomainPlan(domains=[CommDomain(name="tp", worker_indices=[0, 1], window_size=4096)])
        plan.validate(worker_count=3)

        assert plan.bootstrap_for_worker(2) == []

    def test_duplicate_domain_names_fail_validation(self):
        from simpler.task_interface import CommDomain, CommDomainPlan

        plan = CommDomainPlan(
            domains=[
                CommDomain(name="tp", worker_indices=[0], window_size=4096),
                CommDomain(name="tp", worker_indices=[1], window_size=4096),
            ]
        )

        with pytest.raises(ValueError, match="duplicate communication domain name"):
            plan.validate(worker_count=2)

    def test_duplicate_buffer_names_fail_within_one_domain(self):
        from simpler.task_interface import CommDomain, CommDomainPlan

        plan = CommDomainPlan(
            domains=[
                CommDomain(
                    name="tp",
                    worker_indices=[0, 1],
                    window_size=4096,
                    buffers=[_buffer("scratch"), _buffer("scratch")],
                )
            ]
        )

        with pytest.raises(ValueError, match="duplicate names"):
            plan.validate(worker_count=2)

    def test_out_of_range_worker_index_fails_validation(self):
        from simpler.task_interface import CommDomain, CommDomainPlan

        plan = CommDomainPlan(domains=[CommDomain(name="tp", worker_indices=[0, 3], window_size=4096)])

        with pytest.raises(ValueError, match="outside"):
            plan.validate(worker_count=2)

    def test_domain_plan_does_not_own_host_staging(self):
        from simpler.task_interface import CommDomain

        with pytest.raises(TypeError, match="host_inputs"):
            CommDomain(
                name="red",
                worker_indices=[0, 1],
                window_size=4096,
                buffers=[_buffer("scratch")],
                host_inputs=[],
            )

    def test_explicit_bootstrap_config_adds_per_chip_staging_to_plan_domains(self):
        from simpler.task_interface import ChipBootstrapConfig, CommDomain, CommDomainPlan, HostBufferStaging

        plan = CommDomainPlan(
            domains=[
                CommDomain(
                    name="red",
                    worker_indices=[0, 1],
                    window_size=4096,
                    buffers=[_buffer("scratch")],
                ),
                CommDomain(
                    name="blue",
                    worker_indices=[0, 1],
                    window_size=4096,
                    buffers=[_buffer("scratch")],
                ),
            ]
        )
        cfg = ChipBootstrapConfig(
            comm=plan.bootstrap_for_worker(0),
            host_inputs=[
                HostBufferStaging(domain_name="red", name="scratch", shm_name="psm_red", size=16),
                HostBufferStaging(domain_name="blue", name="scratch", shm_name="psm_blue", size=16),
            ],
        )

        domains = {c.name: c for c in cfg.domain_bootstrap_configs()}

        assert domains["red"].input_staging("scratch").shm_name == "psm_red"
        assert domains["blue"].input_staging("scratch").shm_name == "psm_blue"

    def test_overlapping_domains_have_independent_rank_spaces(self):
        from simpler.task_interface import ChipCommDomainContext, ChipContext, CommDomain, CommDomainPlan

        plan = CommDomainPlan(
            domains=[
                CommDomain(
                    name="tp",
                    worker_indices=[0, 1],
                    window_size=4096,
                    buffers=[_buffer("scratch")],
                ),
                CommDomain(
                    name="pp",
                    worker_indices=[1, 2],
                    window_size=4096,
                    buffers=[_buffer("scratch")],
                ),
            ]
        )
        plan.validate(worker_count=3)

        cfgs_by_worker = {
            worker_idx: {cfg.name: cfg for cfg in plan.bootstrap_for_worker(worker_idx)}
            for worker_idx in range(3)
        }

        assert set(cfgs_by_worker[0]) == {"tp"}
        assert set(cfgs_by_worker[1]) == {"tp", "pp"}
        assert set(cfgs_by_worker[2]) == {"pp"}
        assert cfgs_by_worker[1]["tp"].domain_rank == 1
        assert cfgs_by_worker[1]["pp"].domain_rank == 0
        assert cfgs_by_worker[1]["tp"].buffers[0].name == "scratch"
        assert cfgs_by_worker[1]["pp"].buffers[0].name == "scratch"

        ctx = ChipContext(
            device_id=1,
            worker_index=1,
            domains={
                "tp": ChipCommDomainContext(
                    name="tp",
                    domain_rank=1,
                    domain_size=2,
                    device_ctx=0x1000,
                    local_window_base=0x2000,
                    actual_window_size=4096,
                    buffer_ptrs={"scratch": 0x2000},
                ),
                "pp": ChipCommDomainContext(
                    name="pp",
                    domain_rank=0,
                    domain_size=2,
                    device_ctx=0x3000,
                    local_window_base=0x4000,
                    actual_window_size=4096,
                    buffer_ptrs={"scratch": 0x4000},
                ),
            },
        )

        assert ctx.domains["tp"].buffer_ptrs["scratch"] != ctx.domains["pp"].buffer_ptrs["scratch"]
        with pytest.raises(KeyError):
            ctx.domains["dp"]


class TestChipWorkerCommDomainBindings:
    def test_native_worker_exposes_multi_domain_comm_methods(self):
        from _task_interface import _ChipWorker

        worker = _ChipWorker()

        assert hasattr(worker, "comm_create_subcomm")
        assert hasattr(worker, "comm_create_domain")
        assert hasattr(worker, "comm_destroy_all")
