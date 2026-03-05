/**
 * HCCL device-side helpers: HcclRemotePtr, WindowAlloc.
 * Extracted from pto-comm-isa common.hpp for use in simpler-PTO cpt_and_comm.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "hccl_context.h"

#ifndef AICORE
#define AICORE
#endif

#ifndef __gm__
#define __gm__
#endif

// Convert local window pointer to remote rank's equivalent address
template <typename T>
AICORE inline __gm__ T* HcclRemotePtr(__gm__ HcclDeviceContext* ctx, __gm__ T* localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T*)(ctx->windowsIn[pe] + offset);
}

// Allocate from window at (windowBase + offset), advance offset
inline void* WindowAlloc(uint64_t windowBase, size_t& offset, size_t bytes) {
    void* ptr = reinterpret_cast<void*>(windowBase + offset);
    offset += bytes;
    return ptr;
}
