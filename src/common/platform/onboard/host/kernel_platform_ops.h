/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * The CANN backing of the kernel-mode context vocabulary.
 *
 * Every platform constant the context lifecycle needs — chiefly the event
 * creation flag — is resolved here, so KernelExecutionState stays free of
 * CANN and its host-only tests keep observing the values this platform
 * actually asks for.
 */

#pragma once

#include "host/kernel_execution_state.h"

/**
 * The context-lifetime operation table backed by CANN.
 *
 * Every entry is a free CANN call, so the table carries no receiver and its
 * `context` stays null. Events are created with ACL_EVENT_SYNC: the launch
 * sequence waits on them from a stream and never queries their status from
 * the host.
 */
KernelContextOps make_onboard_kernel_context_ops();

/**
 * Select process-wide hardware capture events before creating kernel resources.
 * Reuses hardware mode, including CANN's fixed hardware-mode platforms. An
 * explicit software setting is retained with a diagnostic; other query/set
 * failures are returned.
 * The mode survives context teardown and is shared by every runtime in the process.
 */
int ensure_onboard_kernel_hardware_events() noexcept;
