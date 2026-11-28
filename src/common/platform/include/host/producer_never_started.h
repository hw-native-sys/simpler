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
#pragma once

#include "worker/native_run_execution.h"

class NativeExecutionGate;

// A submitted wrapper can return without ever entering a DFX producer. Only
// the native gate owner may attest that fact, after immutable CANCEL, matching
// entry outcomes and completion of both private streams and publication.
class ProducerNeverStartedProof {
public:
    const NativeRunIdentity &identity() const { return identity_; }

private:
    friend class NativeExecutionGate;
    explicit ProducerNeverStartedProof(const NativeRunIdentity &identity) :
        identity_(identity) {}
    NativeRunIdentity identity_;
};
