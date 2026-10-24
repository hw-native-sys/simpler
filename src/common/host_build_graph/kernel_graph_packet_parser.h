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

#include "host_build_graph/kernel_graph_packet.h"

namespace hbg {

// The patched payload address and checksum itself are excluded. This is a
// corruption check, not a source of resource or callable authority.
uint64_t graph_packet_checksum(const void *packet, size_t size) noexcept;

// The caller supplies immutable, readable storage. No device/cache operation or
// registry lookup occurs here. On failure out is unchanged; on success it borrows
// payload storage and owns copies of the parsed headers and region descriptors.
GraphPacketStatus
parse_graph_packet(const void *packet, size_t size, GraphPacketAddress address_mode, ParsedGraphPacket &out) noexcept;

// Status-only convenience entry; uses the same parser and format rules.
GraphPacketStatus validate_graph_packet(const void *packet, size_t size, GraphPacketAddress address_mode) noexcept;

}  // namespace hbg
