# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RDMA_BACKEND_KERNEL = (
    REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/rdma/rdma_completion_kernel.h"
)
ASYNC_WAIT = REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/async_wait.h"
MAILBOX_TYPES = REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/aicore_completion_mailbox_types.h"
HOST_CMAKE = REPO_ROOT / "src/a5/platform/onboard/host/CMakeLists.txt"
HOST_COMM = REPO_ROOT / "src/a5/platform/onboard/host/comm_hccl.cpp"
KERNEL_COMPILER = REPO_ROOT / "simpler_setup/kernel_compiler.py"
RDMA_SCHEDULER = REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/rdma/rdma_completion_scheduler.h"
SCHEDULER_COMPLETION = REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp"
RDMA_DEFERRED_DEMO = (
    REPO_ROOT
    / "examples/a5/tensormap_and_ringbuffer/rdma_deferred_completion_demo/test_rdma_deferred_completion_demo.py"
)
RDMA_DEFERRED_ORCH = (
    REPO_ROOT
    / "examples/a5/tensormap_and_ringbuffer/rdma_deferred_completion_demo/kernels/orchestration"
    / "rdma_deferred_completion_orch.cpp"
)


def test_rdma_kernel_backend_exposes_deferred_adapter_contract() -> None:
    backend = RDMA_BACKEND_KERNEL.read_text(encoding="utf-8")

    assert "enum class RdmaOp" in backend
    assert "struct RdmaRequestDescriptor" in backend
    assert "RdmaTget" in backend
    assert "RdmaTput" in backend
    assert "DmaEngine::RDMA" in backend
    assert "TGET_ASYNC<pto::comm::DmaEngine::RDMA>" in backend
    assert "TPUT_ASYNC<pto::comm::DmaEngine::RDMA>" in backend
    assert "register_rdma_async_event" in backend
    assert "CompletionRecordCount(session)" in backend
    assert "CompletionRecordAt(session, 0U)" in backend
    assert "CompletionKind::RDMA_HNS1825_CQE" in backend
    assert "COMPLETION_ENGINE_ROCE" in backend
    assert "COMPLETION_TYPE_RDMA_HNS1825_CQE" in backend
    assert "reinterpret_cast<uint64_t>(record.addr)" in backend
    assert "COMPLETION_ENGINE_ROCE," in backend
    assert "COMPLETION_TYPE_RDMA_HNS1825_CQE," in backend
    assert "reinterpret_cast<uint64_t>(workspace)" not in backend


def test_rdma_kernel_backend_uses_peer_mr_base_not_windows_in() -> None:
    backend = RDMA_BACKEND_KERNEL.read_text(encoding="utf-8")

    assert "peer_mr_base_addr" in backend
    assert "peer_mr_ptr" in backend
    assert "PeerMrBaseAddr(workspace, peer)" in backend
    assert "windowsIn" not in backend


def test_rdma_completion_type_is_registered_in_async_wait() -> None:
    wait_source = ASYNC_WAIT.read_text(encoding="utf-8")
    type_source = MAILBOX_TYPES.read_text(encoding="utf-8")

    assert "#define COMPLETION_TYPE_RDMA_HNS1825_CQE 3" in type_source
    assert "backend/rdma/rdma_completion_scheduler.h" in wait_source
    assert "rdma_hns1825_cqe_poll_op" in wait_source
    assert "rdma_hns1825_cqe_retire_op" in wait_source
    assert "COMPLETION_TYPE_RDMA_HNS1825_CQE = 3" in wait_source


def test_a5_host_cmake_gates_rdma_workspace_overlay() -> None:
    cmake = HOST_CMAKE.read_text(encoding="utf-8")

    assert "option(SIMPLER_ENABLE_PTO_RDMA_WORKSPACE" in cmake
    assert "set(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE OFF)" in cmake
    assert "set(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE ON)" in cmake
    assert "Only one PTO async workspace overlay may be enabled" in cmake
    assert "PTO RDMA workspace overlay requires pto-isa RDMA headers" in cmake
    assert "PTO_RDMA_SUPPORTED" in cmake
    assert "PTO_RDMA_BACKEND_HNS_1825_SUPPORTED" in cmake


def test_a5_comm_hccl_keeps_urma_and_rdma_workspace_paths_macro_gated() -> None:
    source = HOST_COMM.read_text(encoding="utf-8")

    assert "#ifdef SIMPLER_ENABLE_PTO_URMA_WORKSPACE" in source
    assert "#ifdef SIMPLER_ENABLE_PTO_RDMA_WORKSPACE" in source
    assert '#include "pto/comm/workspace.hpp"' in source
    assert "pto::comm::Workspace rdma_workspace" in source


def test_a5_comm_hccl_uses_mr1374_rdma_host_api_shape() -> None:
    source = HOST_COMM.read_text(encoding="utf-8")

    assert "#define __gm__" in source
    assert "aclrtGetPhyDevIdByUserDevId" in source
    assert "PTO_ROCE_PHYIDS" in source
    assert "PTO_ROCE_ROOTINFO" in source
    assert 'kDefaultRdmaRootinfoPath = "/etc/hccl_rootinfo.json"' in source
    assert 'kDefaultVirtualTopologyPath = "/var/run/ascend-topologyd/virtualTopology.xml"' in source
    assert "rdma_resolve_local_ip_from_virtual_topology(out.phy_id, out.local_ip)" in source
    assert "rdma_resolve_local_ip_from_env(base_rank, rank_count, out.local_ip)" in source
    assert "PTO_ROCE_LOCAL_IP" in source
    assert "PTO_ROCE_IPS" in source
    assert "rdma_resolve_local_ip_from_hccn_tool(device_id, out.local_ip)" in source
    assert "rdma_resolve_local_ip_from_rootinfo(out.phy_id, out.local_ip, rootinfo_path)" in source
    assert 's.find("\\"CLOS\\"", dev_key_pos)' in source
    assert "hccn_tool -g -dev_info -i %d" in source
    assert '\\"addr\\"' in source
    assert "pto::comm::RdmaTransportConfig" in source
    assert "pto::comm::CreateWorkspace(pto::comm::DmaEngine::RDMA" in source
    assert "WorkspaceStatus::Ok" in source
    assert "RdmaBackend::HNS_1825" not in source


def test_kernel_compiler_forwards_rdma_workspace_macros_to_incore_builds() -> None:
    source = KERNEL_COMPILER.read_text(encoding="utf-8")

    assert '"SIMPLER_ENABLE_PTO_RDMA_WORKSPACE"' in source
    assert '"-DPTO_RDMA_SUPPORTED"' in source
    assert '"-DPTO_RDMA_BACKEND_HNS_1825_SUPPORTED"' in source
    assert "cmd += self._incore_feature_defines()" in source
    assert "_pto_isa_include_dirs" in source
    assert '"pkg_inc"' in source


def test_kernel_compiler_adds_ascend_device_headers_for_rdma_backend_headers() -> None:
    source = KERNEL_COMPILER.read_text(encoding="utf-8")

    assert "get_ascend_incore_include_dirs" in source
    assert 'root / "asc" / "include" / "interface"' in source
    assert 'root / "asc" / "include" / "basic_api" / "interface"' in source
    assert 'root / "ascendc" / "include" / "basic_api" / "interface"' in source
    assert '"kernel_operator_sys_var_intf.h"' in source
    assert '"kernel_operator_sys_var_intf_impl.h"' in source
    assert ".rglob(header)" in source
    assert "for inc_dir in self.get_ascend_incore_include_dirs():" in source


def test_rdma_scheduler_consumes_the_exported_completion_record() -> None:
    source = RDMA_SCHEDULER.read_text(encoding="utf-8")

    assert "poll_rdma_hns1825_cqe_record(uint64_t cqe_addr, uint32_t expected_owner)" in source
    assert "kRdmaCqeOwnerBit" in source
    assert "retire_rdma_hns1825_cqe_record" in source

    # The record carries the CQE address and the owner value that marks it done, so the scheduler
    # holds no copy of the RDMA queue layout and no ABI-drift guard for one.
    assert "struct RdmaInfo" not in source
    assert "struct RdmaWqCtx" not in source
    assert "struct RdmaCqCtx" not in source
    assert "struct Hns1825Cqe" not in source
    assert "struct RdmaMemInfo" not in source
    assert "load_wq_ctx" not in source
    assert "load_cq_ctx" not in source
    assert "load_mem_info" not in source
    assert "poll_rdma_event_handle" not in source
    assert "ABI drift" not in source


def test_rdma_scheduler_stall_snapshot_reads_the_record() -> None:
    source = RDMA_SCHEDULER.read_text(encoding="utf-8")

    assert "log_rdma_hns1825_cqe_record_snapshot" in source
    assert "op_sr_wqebb=0x%x" in source
    assert "log_rdma_wq_snapshot" not in source
    assert "log_rdma_cq_snapshot" not in source
    assert "log_rdma_mem_snapshot" not in source
    assert "log_rdma_wqe_snapshot" not in source


def test_rdma_domain_bootstrap_resolves_env_arrays_by_domain_rank() -> None:
    source = HOST_COMM.read_text(encoding="utf-8")

    assert (
        "resolve_rdma_bootstrap(h, domain_rank, static_cast<uint32_t>(rank_count), myDevice, rdma_bootstrap)" in source
    )
    assert "resolve_rdma_bootstrap(\n            h, static_cast<uint32_t>(h->rank)" not in source


def test_rdma_kernel_uses_peer_independent_session_and_explicit_peer_ops() -> None:
    backend = RDMA_BACKEND_KERNEL.read_text(encoding="utf-8")
    common = (
        REPO_ROOT
        / "examples/a5/tensormap_and_ringbuffer/rdma_deferred_completion_demo/kernels/aiv"
        / "rdma_deferred_completion_common.h"
    ).read_text(encoding="utf-8")
    consumer = (
        REPO_ROOT
        / "examples/a5/tensormap_and_ringbuffer/rdma_deferred_completion_demo/kernels/aiv"
        / "kernel_rdma_deferred_completion_consumer.cpp"
    ).read_text(encoding="utf-8")

    assert "pto::comm::rdma::kRdmaScratchBytes" in common
    assert "using RdmaScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, kRdmaScratchBytes>" in common
    assert (
        "using RdmaScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>"
        not in common
    )
    assert "desc.scratch, desc.workspace, desc.local_rank, session, desc.sync_id" in backend
    assert "desc.scratch, desc.workspace, desc.peer_rank, desc.local_rank, session" not in backend
    assert "TGET_ASYNC<pto::comm::DmaEngine::RDMA>(desc.dst, desc.src, session, desc.peer_rank)" in backend
    assert "TPUT_ASYNC<pto::comm::DmaEngine::RDMA>(desc.dst, desc.src, session, desc.peer_rank)" in backend
    assert "comm_ctx->workSpace), comm_ctx->rankId," in consumer
    assert "readback_session, 2" in consumer
    assert "TGET_ASYNC<pto::comm::DmaEngine::RDMA>(local_scratch, remote_tput, readback_session, peer)" in consumer


def test_a5_scheduler_orders_deferred_completion_slab_reads_after_cond() -> None:
    source = SCHEDULER_COMPLETION.read_text(encoding="utf-8")
    assert "volatile DeferredCompletionSlab *deferred_slab" in source
    poll = source[source.index("void SchedulerContext::check_running_cores_for_completion(") :]
    assert poll.index("reg_load_acquire(core.cond_ptr)") < poll.index("rmb();")
    assert poll.index("rmb();") < poll.index("complete_slot_task(")
    assert "deferred_slab->error_code" in source
    assert "deferred_slab->count = 0" in source


def test_rdma_deferred_demo_serializes_hns1825_sq_posts() -> None:
    py_source = RDMA_DEFERRED_DEMO.read_text(encoding="utf-8")
    orch_source = RDMA_DEFERRED_ORCH.read_text(encoding="utf-8")

    assert py_source.count("[ArgDirection.IN, ArgDirection.OUT, ArgDirection.OUT") == 1
    assert py_source.count("[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT") == 1
    assert "serialize producer posts through marker dependencies" in orch_source
    assert "tget1_args.add_input(tget0_marker)" in orch_source
    assert "tput0_args.add_input(tget1_marker)" in orch_source
    assert "tput1_args.add_input(tput0_marker)" in orch_source
