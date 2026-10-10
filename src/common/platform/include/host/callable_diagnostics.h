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

#include "callable.h"
#include "host_log.h"
#include "utils/fnv1a_64.h"

// image is the patched host source of a completed H2D, not the caller's
// unresolved ChipCallable. No device pointer is dereferenced here.
inline void log_callable_image(
    int device_id, const void *runner, uint64_t chip_hash, uint64_t chip_dev, size_t upload_bytes,
    const ChipCallable &image
) {
    auto &logger = HostLogger::get_instance();
    if (!logger.is_enabled(simpler::log::LogLevel::TIMING)) return;
    logger.log(
        simpler::log::LogLevel::TIMING, __func__,
        "Callable image: device=%d runner=%p chip_hash=0x%lx chip_dev=0x%lx upload_bytes=%zu children=%d", device_id,
        runner, chip_hash, chip_dev, upload_bytes, image.child_count()
    );
    for (int32_t i = 0; i < image.child_count(); ++i) {
        const CoreCallable &child = image.child(i);
        const uint64_t code_hash = simpler::common::utils::fnv1a_64(child.binary_data(), child.binary_size());
        logger.log(
            simpler::log::LogLevel::TIMING, __func__,
            "Callable code: device=%d runner=%p chip_hash=0x%lx chip_dev=0x%lx func_id=%d "
            "code_begin=0x%lx code_end_exclusive=0x%lx code_bytes=%u code_fnv1a64=0x%lx",
            device_id, runner, chip_hash, chip_dev, image.child_func_id(i), child.resolved_addr(),
            child.resolved_addr() + child.binary_size(), child.binary_size(), code_hash
        );
    }
}
