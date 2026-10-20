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

#include "runtime_module.h"

#include <dlfcn.h>

#include <stdexcept>
#include <string>

RuntimeModule::RuntimeModule(const char *path) {
    dlerror();
    handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
        const char *error = dlerror();
        throw std::runtime_error(std::string("dlopen failed: ") + (error != nullptr ? error : "unknown error"));
    }
}

RuntimeModule::~RuntimeModule() {
    if (unload_allowed_) dlclose(handle_);
}

void *RuntimeModule::required_symbol(const char *name) const {
    dlerror();
    void *symbol = dlsym(handle_, name);
    const char *error = dlerror();
    if (error != nullptr) {
        throw std::runtime_error(
            std::string("dlsym failed for '") + name + "': " + error +
            "; every compatible module built from this source tree exports it, so rebuild the module"
        );
    }
    return symbol;
}

void *RuntimeModule::optional_symbol(const char *name) const noexcept { return dlsym(handle_, name); }
