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
URMA_KERNEL = REPO_ROOT / "src/a5/runtime/tensormap_and_ringbuffer/runtime/backend/urma/urma_completion_kernel.h"


def test_a5_urma_overlay_uses_unified_workspace_interface() -> None:
    source = HOST_COMM.read_text()

    assert '#include "pto/comm/workspace.hpp"' in source
    assert "UrmaWorkspaceManager" not in source
    assert "static uint64_t urma_workspace_bytes" not in source
    assert "sizeof(UrmaInfo)" not in source
    assert "sizeof(UrmaWQCtx)" not in source
    assert "sizeof(UrmaCqCtx)" not in source
    assert "sizeof(UrmaMemInfo)" not in source
    assert "pto::comm::Workspace urma_workspace{}" in source
    assert "pto::comm::WorkspaceRequest req{}" in source
    assert "pto::comm::CreateWorkspace(pto::comm::DmaEngine::URMA, req, &workspace)" in source
    assert "h->host_ctx.workSpace = reinterpret_cast<uint64_t>(h->urma_workspace.addr)" in source
    assert "h->host_ctx.workSpaceSize = h->urma_workspace.bytes" in source


def test_a5_urma_release_paths_use_workspace_object() -> None:
    source = HOST_COMM.read_text()

    assert "pto::comm::DestroyWorkspace(&alloc.urma_workspace)" in source
    assert "pto::comm::DestroyWorkspace(&h->urma_workspace)" in source
    assert "pto::comm::AbandonWorkspace(&workspace)" in source


def test_a5_urma_completion_records_support_multi_jetty_events() -> None:
    source = URMA_KERNEL.read_text()

    assert "CompletionRecordCount(session)" in source
    assert "record_count != 1U" not in source
    assert "record_count == 0U" in source
    assert "record_count > ctx.completion_capacity" in source
    assert "for (uint32_t record_index = 0U; record_index < record_count; ++record_index)" in source
    assert "CompletionRecordAt(session, record_index)" in source
    assert "CompletionKind::URMA_CQE_DW0" in source
