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

// Create and bind an enabled session on this thread; throw on nesting or failure.
// Runtimes without host dependency capture return nullptr from their weak stub.
extern "C" void *dep_gen_host_graph_create_capture_session();
// Destroy on the creating thread, after take/export and prepare error cleanup.
extern "C" void dep_gen_host_graph_destroy_capture_session(void *session) noexcept;

// The prepare call owns dependency capture through normal and exception cleanup.
// Enabled scopes create a session ready for this prepare's orchestration; disabled
// scopes allocate nothing. begin_capture() still marks each orchestration entry,
// while a restored parked graph can be exported without starting a new capture.
// The token is opaque to platform code; all graph storage stays in the runtime.
class HostDependencyCaptureScope {
public:
    explicit HostDependencyCaptureScope(bool enabled) :
        session_(enabled ? dep_gen_host_graph_create_capture_session() : nullptr) {}
    ~HostDependencyCaptureScope() {
        if (session_) dep_gen_host_graph_destroy_capture_session(session_);
    }
    HostDependencyCaptureScope(const HostDependencyCaptureScope &) = delete;
    HostDependencyCaptureScope &operator=(const HostDependencyCaptureScope &) = delete;

private:
    void *session_;
};
