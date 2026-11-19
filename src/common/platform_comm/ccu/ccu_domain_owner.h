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
#include <vector>

#include "ccu_native_module.h"
#include "ccu_run_lifecycle.h"

namespace simpler::ccu {

class DomainResourceBackend {
public:
    virtual ~DomainResourceBackend() = default;
    virtual bool drained() const = 0;
    virtual bool close() = 0;
};

class DomainCcuOwner final {
public:
    explicit DomainCcuOwner(std::unique_ptr<DomainResourceBackend> backend);

    DomainCcuOwner(const DomainCcuOwner &) = delete;
    DomainCcuOwner &operator=(const DomainCcuOwner &) = delete;

    void retain_module(std::shared_ptr<NativeModule> module);
    void set_registered_kernels(std::vector<RegisteredKernel> kernels);
    std::unique_ptr<DomainRun>
    begin_run(uint64_t generation, const std::vector<MissionRunResources> &missions, RunLifecycleSdk &sdk);

    bool drained() const;
    bool close();
    bool closed() const noexcept { return closed_; }
    size_t retained_module_count() const noexcept { return modules_.size(); }
    const std::vector<RegisteredKernel> &registered_kernels() const noexcept { return registered_kernels_; }
    DomainRunLifecycle &lifecycle() noexcept { return lifecycle_; }

private:
    std::unique_ptr<DomainResourceBackend> backend_;
    std::vector<std::shared_ptr<NativeModule>> modules_;
    std::vector<RegisteredKernel> registered_kernels_;
    DomainRunLifecycle lifecycle_;
    bool closed_{false};
};

}  // namespace simpler::ccu
