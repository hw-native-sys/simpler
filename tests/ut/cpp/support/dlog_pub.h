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

#ifndef TESTS_UT_CPP_STUBS_DLOG_PUB_H_
#define TESTS_UT_CPP_STUBS_DLOG_PUB_H_

constexpr int AICPU = 0;
constexpr int DLOG_DEBUG = 0;
constexpr int DLOG_INFO = 1;
constexpr int DLOG_WARN = 2;
constexpr int DLOG_ERROR = 3;

inline bool CheckLogLevel(int /* module */, int /* level */) { return true; }

inline void dlog_debug(int /* module */, const char * /* fmt */, ...) {}
inline void dlog_info(int /* module */, const char * /* fmt */, ...) {}
inline void dlog_warn(int /* module */, const char * /* fmt */, ...) {}
inline void dlog_error(int /* module */, const char * /* fmt */, ...) {}

#endif  // TESTS_UT_CPP_STUBS_DLOG_PUB_H_
