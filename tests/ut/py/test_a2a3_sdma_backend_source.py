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
SDMA_KERNEL = REPO_ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/backend/sdma/sdma_completion_kernel.h"
SDMA_WAIT = REPO_ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/async_wait.h"
SDMA_MAILBOX = REPO_ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/aicore_completion_mailbox.h"
SDMA_TYPES = REPO_ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/aicore_completion_mailbox_types.h"


def test_a2a3_sdma_backend_uses_post_done_completion_flow() -> None:
    kernel = SDMA_KERNEL.read_text()
    wait = SDMA_WAIT.read_text()

    assert "PrepareEventCheck" not in kernel
    assert "GetEventRecord" not in kernel
    assert "session.sdmaSession" not in kernel
    assert "COMPLETION_TYPE_SDMA_EVENT_RECORD" in kernel
    assert "LoadSdmaSession" not in kernel
    assert "runtimeCtx.postDoneBase" not in kernel
    assert "PTO2_ERROR" not in kernel
    assert "CompletionRecordCount(session)" in kernel
    assert "CompletionRecordAt(session, record_index)" in kernel
    assert "CompletionKind::SDMA_POST_DONE" in kernel
    assert "register_sdma_post_done_record(ctx, record.addr, record.expected)" in kernel

    assert "COMPLETION_TYPE_SDMA_EVENT_RECORD" in wait
    assert "COMPLETION_TYPE_SDMA_POST_DONE" not in wait
    assert "backend_cookie" in wait
    assert "poll_sdma_post_done_record(cond.addr, cond.backend_cookie)" in wait


def test_a2a3_sdma_mailbox_carries_backend_cookie() -> None:
    mailbox = SDMA_MAILBOX.read_text()
    types = SDMA_TYPES.read_text()

    assert "uint64_t backend_cookie" in mailbox
    assert "backend_cookie" in mailbox and "try_push_condition(" in mailbox
    assert "backend_cookie" in types
    assert "COMPLETION_TYPE_SDMA_EVENT_RECORD" in types
    assert "COMPLETION_TYPE_SDMA_POST_DONE" not in types
