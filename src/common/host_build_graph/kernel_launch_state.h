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

#include <memory>
#include <unordered_map>
#include <vector>

#include "host_build_graph/kernel_callable_registration.h"
#include "host_build_graph/kernel_graph_template.h"
#include "host_build_graph/kernel_graph_prepared.h"
#include "host_build_graph/kernel_resource_requirements.h"
#include "task_interface/task_args.h"

namespace hbg {

struct KernelPreparedGraphImage {
    GraphPreparedDispatch dispatch{};
    void *pinned_image{nullptr};
    int (*release_image)(void *) noexcept {nullptr};

    ~KernelPreparedGraphImage() { (void)release(); }
    KernelPreparedGraphImage() = default;
    KernelPreparedGraphImage(const KernelPreparedGraphImage &) = delete;
    KernelPreparedGraphImage &operator=(const KernelPreparedGraphImage &) = delete;

    int release() noexcept {
        if (pinned_image == nullptr || release_image == nullptr) return 0;
        const int rc = release_image(pinned_image);
        if (rc == 0) pinned_image = nullptr;
        return rc;
    }

    void abandon() noexcept {
        pinned_image = nullptr;
        release_image = nullptr;
    }
};

struct KernelCallableLaunchState {
    GraphLaunchTemplate graph_template;
    std::shared_ptr<KernelPreparedGraphImage> prepared_image;
    std::unique_ptr<ChipStorageTaskArgs> argument_snapshot;
    uint64_t argument_hash{0};
};

struct KernelContextLaunchState {
    KernelResourcePlan resource_plan;
    GraphSlotRegistration slot_registration{};
    std::unordered_map<int32_t, HbgCallableRegistration> callable_registrations;
    // Captured DMA nodes may retain an older source after the callable's cache
    // changes. Sources remain immutable and owned until quiescent context close.
    std::vector<std::shared_ptr<KernelPreparedGraphImage>> prepared_images;
    bool resources_prepared{false};
    bool slot_registered{false};

    int release_prepared_images() noexcept {
        for (auto &image : prepared_images) {
            const int rc = image->release();
            if (rc != 0) return rc;
        }
        prepared_images.clear();
        return 0;
    }

    void abandon_prepared_images() noexcept {
        for (auto &image : prepared_images)
            image->abandon();
    }
};

}  // namespace hbg
