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

#include "host/auto_admission.h"
#include "host/host_dependency_capture_scope.h"
#include "host_build_graph/host/dep_gen_host_graph.h"
extern "C" void touch_capture_prepare() {
    AutoAdmissionState::ClaimScope prepare_claims;
    HostDependencyCaptureScope capture(true);
    AutoAdmissionState::record_claim({});
    AutoAdmissionState::commit_claims();
    dep_gen_host_graph_begin_capture();
    simpler::dfx::host_graph::HostGraphExport graph;
    dep_gen_host_graph_take(&graph);
}

#if defined(__linux__)
#include <cerrno>
#include <system_error>
#include <pthread.h>

namespace {
int fail_create_at = 0;
int create_calls = 0;
bool fail_bind = false;
int live_keys = 0;
}  // namespace
extern "C" int __real_pthread_key_create(pthread_key_t *, void (*)(void *));
extern "C" int __real_pthread_key_delete(pthread_key_t);
extern "C" int __real_pthread_setspecific(pthread_key_t, const void *);
extern "C" int __wrap_pthread_key_create(pthread_key_t *key, void (*destructor)(void *)) {
    if (++create_calls == fail_create_at) return EAGAIN;
    const int rc = __real_pthread_key_create(key, destructor);
    if (rc == 0) ++live_keys;
    return rc;
}
extern "C" int __wrap_pthread_key_delete(pthread_key_t key) {
    const int rc = __real_pthread_key_delete(key);
    if (rc == 0) --live_keys;
    return rc;
}
extern "C" int __wrap_pthread_setspecific(pthread_key_t key, const void *value) {
    if (fail_bind) {
        fail_bind = false;
        return ENOMEM;
    }
    return __real_pthread_setspecific(key, value);
}
extern "C" int capture_tls_probe(int creation_failure, bool binding_failure) {
    create_calls = 0;
    fail_create_at = creation_failure;
    fail_bind = binding_failure;
    try {
        HostDependencyCaptureScope capture(true);
        return 0;
    } catch (const std::system_error &error) {
        return error.code().value();
    }
}
extern "C" int capture_live_keys() { return live_keys; }

extern "C" void capture_fail_clear() {
    HostDependencyCaptureScope capture(true);
    fail_bind = true;
}
extern "C" bool capture_rejected_nesting_preserves_state() {
    HostDependencyCaptureScope capture(true);
    try {
        HostDependencyCaptureScope nested(true);
    } catch (const std::logic_error &) {
        return dep_gen_host_graph_enabled();
    }
    return false;
}
#endif
