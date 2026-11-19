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

#include "ccu_domain_owner.h"

#include <stdexcept>
#include <utility>

namespace simpler::ccu {

DomainCcuOwner::DomainCcuOwner(std::unique_ptr<DomainResourceBackend> backend) :
    backend_(std::move(backend)) {
    if (backend_ == nullptr) throw std::invalid_argument("CCU domain backend is required");
}

void DomainCcuOwner::retain_module(std::shared_ptr<NativeModule> module) {
    if (closed_ || module == nullptr) throw std::invalid_argument("invalid CCU module ownership transfer");
    modules_.push_back(std::move(module));
}

void DomainCcuOwner::set_registered_kernels(std::vector<RegisteredKernel> kernels) {
    if (closed_ || lifecycle_.has_active_run()) throw std::runtime_error("CCU domain registration is not mutable");
    registered_kernels_ = std::move(kernels);
}

std::unique_ptr<DomainRun>
DomainCcuOwner::begin_run(uint64_t generation, const std::vector<MissionRunResources> &missions, RunLifecycleSdk &sdk) {
    if (closed_) throw std::runtime_error("CCU domain is closed");
    return lifecycle_.begin_run(generation, registered_kernels_, missions, sdk);
}

bool DomainCcuOwner::drained() const {
    return !closed_ && !lifecycle_.has_active_run() && lifecycle_.admission_open() &&
           !lifecycle_.resources_retained() && backend_->drained();
}

bool DomainCcuOwner::close() {
    if (closed_) return true;
    if (!drained() || !backend_->close()) return false;
    registered_kernels_.clear();
    modules_.clear();
    backend_.reset();
    closed_ = true;
    return true;
}

}  // namespace simpler::ccu
