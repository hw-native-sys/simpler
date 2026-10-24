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
HOST_COMM = REPO_ROOT / "src/a2a3/platform/onboard/host/comm_hccl.cpp"
HOST_CMAKE = REPO_ROOT / "src/a2a3/platform/onboard/host/CMakeLists.txt"


def test_a2a3_host_uses_unified_sdma_workspace_interface() -> None:
    source = HOST_COMM.read_text()

    assert '#include "pto/comm/workspace.hpp"' in source
    assert "sdma_workspace_manager.hpp" not in source
    assert "std::unique_ptr<pto::comm::sdma::SdmaWorkspaceManager>" not in source
    assert "auto workspace = std::make_unique<pto::comm::Workspace>()" in source
    assert "pto::comm::WorkspaceRequest req{}" in source
    assert "pto::comm::CreateWorkspace(pto::comm::DmaEngine::SDMA, req, workspace.get())" in source
    assert "addr_out[DMA_WORKSPACE_SDMA] = reinterpret_cast<uint64_t>(workspace->addr)" in source
    assert "16 * 1024" not in source


def test_a2a3_sdma_workspace_release_paths_are_explicit() -> None:
    source = HOST_COMM.read_text()

    assert "pto::comm::AbandonWorkspace(workspace.get())" in source
    assert "std::unique_ptr<pto::comm::Workspace> workspace(static_cast<pto::comm::Workspace *>(handle))" in source
    assert "pto::comm::DestroyWorkspace(workspace.get())" in source
    assert "destroy_sdma_workspace" not in source
    assert "abandon_sdma_workspace" not in source


def test_a2a3_host_cmake_maps_simpler_sdma_to_unified_workspace_macros() -> None:
    cmake = HOST_CMAKE.read_text()

    assert "SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=1" in cmake
    assert "PTO_COMM_WORKSPACE_SDMA_SUPPORTED=1" in cmake
    assert "PTO_COMM_WORKSPACE_URMA_SUPPORTED=0" in cmake
    assert "PTO_COMM_WORKSPACE_RDMA_SUPPORTED=0" in cmake
