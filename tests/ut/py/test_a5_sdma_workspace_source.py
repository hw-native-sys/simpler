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
HOST_COMM = REPO_ROOT / "src/a5/platform/onboard/host/comm_hccl.cpp"
HOST_CMAKE = REPO_ROOT / "src/a5/platform/onboard/host/CMakeLists.txt"


def test_a5_sdma_overlay_uses_unified_workspace_interface() -> None:
    source = HOST_COMM.read_text()

    assert '#include "pto/comm/workspace.hpp"' in source
    assert "pto/comm/async/sdma/sdma_workspace_manager.hpp" not in source
    assert "pto/comm/async/urma/urma_workspace_manager.hpp" not in source
    assert "std::unique_ptr<pto::comm::sdma::SdmaWorkspaceManager>" not in source
    assert "pto::comm::Workspace sdma_workspace{}" in source
    assert "pto::comm::WorkspaceRequest req{}" in source
    assert "pto::comm::CreateWorkspace(pto::comm::DmaEngine::SDMA, req, &h->sdma_workspace)" in source
    assert "h->host_ctx.workSpace = reinterpret_cast<uint64_t>(h->sdma_workspace.addr)" in source
    assert "h->host_ctx.workSpaceSize = h->sdma_workspace.bytes" in source
    assert "16 * 1024" not in source


def test_a5_sdma_release_and_domain_paths_use_workspace_object() -> None:
    source = HOST_COMM.read_text()
    start = source.index("static void ensure_sdma_workspace")
    ensure_sdma = source[start : source.index("#ifdef SIMPLER_ENABLE_PTO_URMA_WORKSPACE", start)]

    assert "pto::comm::DestroyWorkspace(&h->sdma_workspace)" in source
    assert "pto::comm::AbandonWorkspace(&h->sdma_workspace)" in source
    assert "domain_workspace_addr = reinterpret_cast<uint64_t>(h->sdma_workspace.addr)" in source
    assert "domain_workspace_size = h->sdma_workspace.bytes" in source
    assert "GetWorkspaceAddr()" not in ensure_sdma
    assert "h->sdma_workspace->" not in ensure_sdma


def test_a5_sdma_cmake_maps_to_unified_workspace_macros() -> None:
    cmake = HOST_CMAKE.read_text()

    assert "SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=1" in cmake
    assert "PTO_COMM_WORKSPACE_SDMA_SUPPORTED=1" in cmake
    assert "PTO_COMM_WORKSPACE_URMA_SUPPORTED=0" in cmake
    assert "PTO_COMM_WORKSPACE_RDMA_SUPPORTED=0" in cmake


def test_a5_sdma_derived_context_inherits_workspace_outside_urma_mode() -> None:
    source = HOST_COMM.read_text()
    start = source.index('extern "C" int comm_derive_context')
    derive_context = source[start : source.index("void *newDevMem", start)]

    assert "ctx.workSpace = h->host_ctx.workSpace" in derive_context
    assert "ctx.workSpaceSize = h->host_ctx.workSpaceSize" in derive_context
    assert "if (!rank_ids_are_dense_prefix(rank_ids, rank_count))" in derive_context
