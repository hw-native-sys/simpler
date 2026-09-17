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
 * @file collected_record.h
 * @brief A host-collected diagnostic record together with the run that produced it.
 *
 * Every collector's device producer stamps the run's epoch onto a buffer when
 * it acquires it, and the host copies that stamp out alongside the records.
 * Identity has to be *copied*, not referenced: a buffer returned to the pool is
 * re-stamped by whichever run acquires it next, so a record already collected
 * here can no longer recover its run from the buffer it came from.
 *
 * The wrapper is deliberately thin and shared, because attribution is the same
 * question for every collector — only the record payload differs.
 */

#pragma once

#include <cstdint>

template <typename Record>
struct CollectedRecord {
    Record record;
    uint64_t run_epoch;  // 0 when the producer had no run identity to stamp
    // Buffer generation within that run. Restarts per run, so it identifies a
    // buffer only together with `run_epoch` — never a cross-run ordering key.
    uint32_t local_seq;
    uint32_t reserved;
};
