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

#include <cstring>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <vector>

#include "ccu_device_protocol.h"

namespace simpler::ccu {

struct PublicationDomain {
    uint64_t owner, id, generation;
    uint32_t rank, rank_count;
    uint32_t version = device::PROTOCOL_VERSION;
};

struct PreparedProgramBinding {
    uint64_t platform_program;
    uint64_t doorbell_addr;
    uint64_t doorbell_value;
};

// The owner retains the prepared programs, module, domain and uploaded resources.
// Images and their device allocations stay leased until every device reader stops.
class PublicationImage {
public:
    const void *data() const { return storage_.get(); }
    size_t size() const { return bytes_; }
    const PublicationDomain &domain() const { return domain_; }
    int32_t callable_id() const { return callable_id_; }
    bool open() const { return open_; }

    // Caller confirms backend preparation before sealing the upload image. Device
    // pointers may be published only after the complete transport copy succeeds.
    void open_for_publication() {
        if (callable_id_ < 0) {
            static_cast<device::DomainDirectory *>(storage_.get())->admission = device::Admission::OPEN;
        } else {
            static_cast<device::Workspace *>(storage_.get())->admission = device::Admission::OPEN;
        }
        open_ = true;
    }

    static std::shared_ptr<PublicationImage> instance(
        PublicationDomain domain, int32_t callable_id, const std::vector<device::MemoryRange> &ranges,
        const std::vector<PreparedProgramBinding> &programs, uint64_t timeout_ns, std::shared_ptr<const void> owner,
        uint64_t platform_context = 0, uint64_t platform_context_bytes = 0
    ) {
        validate_domain(domain);
        if (callable_id < 0 || ranges.empty() || ranges.size() > UINT32_MAX || programs.empty() ||
            programs.size() > UINT32_MAX || !timeout_ns || !owner) {
            throw std::invalid_argument("invalid CCU instance publication");
        }
        for (const auto &range : ranges) {
            if (!range.addr || !range.bytes || range.bytes > UINT64_MAX - range.addr) {
                throw std::invalid_argument("invalid CCU publication range");
            }
        }
        for (const auto &program : programs) {
            if (!program.platform_program || !program.doorbell_addr || program.doorbell_addr % alignof(uint64_t)) {
                throw std::invalid_argument("CCU program and doorbell must be explicitly prepared");
            }
        }
        const size_t ranges_offset = sizeof(device::Workspace);
        const size_t bindings_offset = end_offset(ranges_offset, ranges.size(), sizeof(device::MemoryRange));
        const size_t request_offset =
            align(end_offset(bindings_offset, programs.size(), sizeof(device::ProgramBinding)));
        auto image = std::shared_ptr<PublicationImage>(
            new PublicationImage(domain, callable_id, end_offset(request_offset, 1, sizeof(device::Request)))
        );
        image->owners_.push_back(std::move(owner));
        auto &header = image->construct<device::Workspace>(0);
        header.version = domain.version;
        header.header_bytes = sizeof(header);
        header.image_bytes = image->size();
        header.domain_owner = domain.owner;
        header.domain_id = domain.id;
        header.generation = domain.generation;
        header.rank = domain.rank;
        header.rank_count = domain.rank_count;
        header.admission = device::Admission::CLOSED;
        header.ranges_offset = ranges_offset;
        header.bindings_offset = bindings_offset;
        header.range_count = static_cast<uint32_t>(ranges.size());
        header.binding_count = static_cast<uint32_t>(programs.size());
        header.timeout_ns = timeout_ns;
        header.platform_context = platform_context;
        header.platform_context_bytes = platform_context_bytes;
        for (size_t i = 0; i < ranges.size(); ++i) {
            image->construct<device::MemoryRange>(ranges_offset + i * sizeof(device::MemoryRange)) = ranges[i];
        }
        for (size_t i = 0; i < programs.size(); ++i) {
            image->construct<device::ProgramBinding>(bindings_offset + i * sizeof(device::ProgramBinding)) = {
                callable_id,
                static_cast<uint32_t>(i),
                request_offset,
                programs[i].doorbell_addr,
                programs[i].doorbell_value,
                programs[i].platform_program
            };
        }
        image->construct<device::Request>(request_offset);
        return image;
    }

    struct UploadedInstance {
        int32_t callable_id;
        uint64_t device_addr;
        std::shared_ptr<const PublicationImage> image;
        std::shared_ptr<const void> device_owner;
    };

    static std::shared_ptr<PublicationImage>
    directory(PublicationDomain domain, const std::vector<UploadedInstance> &instances) {
        validate_domain(domain);
        if (instances.empty() || instances.size() > UINT32_MAX) {
            throw std::invalid_argument("invalid CCU directory size");
        }
        std::set<int32_t> ids;
        std::vector<std::pair<uint64_t, uint64_t>> spans;
        for (const auto &entry : instances) {
            if (entry.callable_id < 0 || !ids.insert(entry.callable_id).second || !entry.image ||
                entry.image->callable_id() != entry.callable_id || !entry.image->open() || !entry.device_owner ||
                !entry.device_addr || entry.device_addr % ALIGNMENT ||
                entry.image->size() > UINT64_MAX - entry.device_addr) {
                throw std::invalid_argument("invalid CCU directory entry");
            }
            const auto &other = entry.image->domain();
            if (other.owner != domain.owner || other.id != domain.id || other.generation != domain.generation ||
                other.rank != domain.rank || other.rank_count != domain.rank_count || other.version != domain.version) {
                throw std::invalid_argument("CCU directory instance belongs to a different domain");
            }
            const auto end = entry.device_addr + entry.image->size();
            for (const auto &span : spans) {
                if (entry.device_addr < span.second && span.first < end) {
                    throw std::invalid_argument("CCU directory instance allocations overlap");
                }
            }
            spans.emplace_back(entry.device_addr, end);
        }
        auto image = std::shared_ptr<PublicationImage>(new PublicationImage(
            domain, -1, end_offset(sizeof(device::DomainDirectory), instances.size(), sizeof(device::InstanceEntry))
        ));
        auto &header = image->construct<device::DomainDirectory>(0);
        header.version = domain.version;
        header.header_bytes = sizeof(header);
        header.image_bytes = image->size();
        header.domain_owner = domain.owner;
        header.domain_id = domain.id;
        header.generation = domain.generation;
        header.rank = domain.rank;
        header.rank_count = domain.rank_count;
        header.admission = device::Admission::CLOSED;
        header.entries_offset = sizeof(header);
        header.entry_count = static_cast<uint32_t>(instances.size());
        for (size_t i = 0; i < instances.size(); ++i) {
            const auto &entry = instances[i];
            image->construct<device::InstanceEntry>(sizeof(header) + i * sizeof(device::InstanceEntry)) = {
                entry.callable_id, 0, entry.device_addr, entry.image->size()
            };
            image->owners_.push_back(entry.image);
            image->owners_.push_back(entry.device_owner);
        }
        return image;
    }

private:
    static constexpr size_t ALIGNMENT = 128;
    struct Deleter {
        void operator()(void *address) const { ::operator delete(address, std::align_val_t{ALIGNMENT}); }
    };
    PublicationImage(PublicationDomain domain, int32_t callable_id, size_t bytes) :
        domain_(domain),
        callable_id_(callable_id),
        bytes_(bytes),
        storage_(::operator new(bytes, std::align_val_t{ALIGNMENT})) {
        std::memset(storage_.get(), 0, bytes);
    }
    template <typename T>
    T &construct(size_t offset) {
        return *new (static_cast<unsigned char *>(storage_.get()) + offset) T{};
    }
    static size_t end_offset(size_t start, size_t count, size_t stride) {
        if (count > (SIZE_MAX - start) / stride) throw std::invalid_argument("CCU publication size overflow");
        return start + count * stride;
    }
    static size_t align(size_t offset) { return end_offset(offset, ALIGNMENT - 1, 1) & ~(ALIGNMENT - 1); }
    static void validate_domain(const PublicationDomain &domain) {
        if (domain.version != device::PROTOCOL_VERSION || !domain.owner || !domain.generation || !domain.rank_count ||
            domain.rank_count > 64 || domain.rank >= domain.rank_count) {
            throw std::invalid_argument("invalid CCU publication domain or protocol version");
        }
    }
    PublicationDomain domain_;
    int32_t callable_id_;
    size_t bytes_;
    bool open_ = false;
    std::unique_ptr<void, Deleter> storage_;
    std::vector<std::shared_ptr<const void>> owners_;
};

}  // namespace simpler::ccu
