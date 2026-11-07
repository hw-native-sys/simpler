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

#include <cstdlib>

#if defined(SIMPLER_TEST_LINKER_WRAP)
extern "C" int __real_release_run_bindings_impl(void *, const void *);
extern "C" int __real_publish_run_image_impl(void *, const void *);
#endif

namespace {
int (*release_bindings)(void *, const void *) = nullptr;
int injected_result = 0;
int (*publish_image)(void *, const void *) = nullptr;
int publication_result = 0;
int publication_calls = 0, release_calls = 0;
int actual_publication_result = 0, actual_release_result = 0;
}  // namespace

extern "C" void set_binding_release_result(void *function, int result) {
    release_bindings = reinterpret_cast<int (*)(void *, const void *)>(function);
    injected_result = result;
    release_calls = 0;
}

extern "C" void set_publication_result(void *function, int result) {
    publish_image = reinterpret_cast<int (*)(void *, const void *)>(function);
    publication_result = result;
    publication_calls = 0;
}

extern "C" int publication_call_count() { return publication_calls; }
extern "C" int binding_release_call_count() { return release_calls; }
extern "C" int actual_publication_rc() { return actual_publication_result; }
extern "C" int actual_binding_release_rc() { return actual_release_result; }

#if defined(SIMPLER_TEST_LINKER_WRAP)
extern "C" int __wrap_release_run_bindings_impl(void *runtime, const void *api) {
    const auto original = release_bindings != nullptr ? release_bindings : __real_release_run_bindings_impl;
#else
extern "C" int release_run_bindings_impl(void *runtime, const void *api) {
    const auto original = release_bindings;
#endif
    if (original == nullptr) std::abort();
    ++release_calls;
    actual_release_result = original(runtime, api);
    return actual_release_result != 0 ? actual_release_result : injected_result;
}

#if defined(SIMPLER_TEST_LINKER_WRAP)
extern "C" int __wrap_publish_run_image_impl(void *runtime, const void *api) {
    const auto original = publish_image != nullptr ? publish_image : __real_publish_run_image_impl;
#else
extern "C" int publish_run_image_impl(void *runtime, const void *api) {
    const auto original = publish_image;
#endif
    if (original == nullptr) std::abort();
    ++publication_calls;
    actual_publication_result = original(runtime, api);
    return actual_publication_result != 0 ? actual_publication_result : publication_result;
}
