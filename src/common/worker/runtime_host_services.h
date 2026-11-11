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

// Internal same-build contract between RuntimeModule and a runtime's host services.
// A runtime exports both hooks or neither. Successful acquire returns zero and
// owns one lease; failed acquire leaves no lease to release. Acquisition may wait
// for a previous owner's final shutdown. A lease need not start any workers.
//
// Release runs outside service workers, after this owner's binds and native runs
// retire. The last release blocks until accepted jobs and their closures retire,
// workers join and their storage is destroyed. Only then may the loader dlclose.
// A failed join or cleanup terminates; the loader must not continue with a
// half-closed service set. Other live leases keep the shared services available.
// Terminal abandonment retains the loader reference and lease together instead
// of calling release.
extern "C" int simpler_runtime_acquire_host_services() noexcept;
extern "C" void simpler_runtime_release_host_services() noexcept;

namespace simpler {
using AcquireHostServices = decltype(&simpler_runtime_acquire_host_services);
using ReleaseHostServices = decltype(&simpler_runtime_release_host_services);
inline constexpr char kAcquireHostServicesSymbol[] = "simpler_runtime_acquire_host_services";
inline constexpr char kReleaseHostServicesSymbol[] = "simpler_runtime_release_host_services";
}  // namespace simpler
