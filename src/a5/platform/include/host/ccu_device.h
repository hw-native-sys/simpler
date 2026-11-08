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

#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include "host/ccu_execution.h"

// SDK-free seam; production and no-hardware tests link different factories.
// 不含 SDK 类型；正式构建与无硬件测试链接不同工厂。
struct CcuSdkError : std::runtime_error {
    int code;
    CcuSdkError(int value, const char *operation) :
        std::runtime_error(operation),
        code(value) {}
};

struct CcuDeviceObservation {
    CcuCompletionRecordV1 record{};
    bool trigger_drained{};
    bool ccu_drained{};
};

class CcuDevice {
public:
    virtual ~CcuDevice() = default;
    virtual void Open(const CcuCreateOptionsV1 &options, uint64_t context, uint64_t generation) = 0;
    virtual uint64_t Token(uint64_t base, uint64_t size) = 0;
    virtual uint64_t Program(uint32_t root) const = 0;
    virtual void Launch(uint32_t root, const std::array<uint64_t, 8> &args) = 0;
    virtual void Trigger() = 0;
    virtual CcuDeviceObservation Observe() = 0;
    virtual uint64_t Gate() const = 0;
    virtual uint64_t Completion() const = 0;
    // Close is retryable and leaves failed resources owned by this object.
    // 关闭失败的资源仍由对象持有，调用方必须保留对象并重试。
    virtual void Close() = 0;
};

std::shared_ptr<CcuDevice> ccu_make_device();
bool ccu_device_available();
uint32_t ccu_device_root_info_size();
void ccu_device_root_info(int device, void *bytes);
