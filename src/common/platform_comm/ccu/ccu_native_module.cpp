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

#include "ccu_native_module.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <iterator>
#include <set>
#include <string_view>
#include <unistd.h>
#include <utility>
#if defined(__linux__)
#include <linux/memfd.h>
#include <sys/syscall.h>
#endif

namespace simpler::ccu {
namespace {

constexpr uint32_t kMaxKernelCount = 4096;
constexpr uint32_t kMaxManifestSize = 16U * 1024U * 1024U;
constexpr uint32_t kMaxEntrySymbolSize = 255;

class FileDescriptor final {
public:
    explicit FileDescriptor(int value) :
        value_(value) {}
    ~FileDescriptor() {
        if (value_ >= 0) (void)::close(value_);
    }
    FileDescriptor(const FileDescriptor &) = delete;
    FileDescriptor &operator=(const FileDescriptor &) = delete;
    int get() const { return value_; }
    int release() { return std::exchange(value_, -1); }

private:
    int value_;
};

[[noreturn]] void fail(const std::string &message) { throw std::invalid_argument("CCU native module: " + message); }

bool valid_symbol(std::string_view symbol) {
    const auto first = [](unsigned char value) {
        return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') || value == '_';
    };
    const auto rest = [&](unsigned char value) {
        return first(value) || (value >= '0' && value <= '9');
    };
    if (symbol.empty() || !first(static_cast<unsigned char>(symbol.front()))) return false;
    for (char value : symbol)
        if (!rest(static_cast<unsigned char>(value))) return false;
    return true;
}

bool ranges_overlap(uint64_t first_begin, uint64_t first_end, uint64_t second_begin, uint64_t second_end) {
    return first_begin < second_end && second_begin < first_end;
}

void validate_elf(const std::vector<uint8_t> &bytes) {
    if (bytes.size() < 64 || std::memcmp(bytes.data(), "\177ELF", 4) != 0 || bytes[4] != 2 || bytes[5] != 1 ||
        bytes[6] != 1 || bytes[16] != 3 || bytes[17] != 0)
        fail("module must be an ELF64 little-endian shared library");
}

void write_all(int fd, const std::vector<uint8_t> &bytes) {
    size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t count = ::write(fd, bytes.data() + written, bytes.size() - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("CCU native module exact-byte write failed");
        written += static_cast<size_t>(count);
    }
}

void append_u64(std::vector<uint8_t> &target, uint64_t value) {
    for (uint32_t shift = 0; shift < 64; shift += 8)
        target.push_back(static_cast<uint8_t>(value >> shift));
}

void append_bytes(std::vector<uint8_t> &target, const void *data, size_t size) {
    append_u64(target, size);
    const auto *begin = static_cast<const uint8_t *>(data);
    target.insert(target.end(), begin, begin + size);
}

void *find_symbol(void *handle, const std::string &symbol) {
    (void)::dlerror();
    void *result = ::dlsym(handle, symbol.c_str());
    const char *error = ::dlerror();
    if (error != nullptr || result == nullptr) fail("missing entry symbol " + symbol);
    return result;
}

}  // namespace

NativeModule::~NativeModule() {
    if (handle_ != nullptr) (void)::dlclose(handle_);
    if (backing_fd_ >= 0) (void)::close(backing_fd_);
}

std::shared_ptr<NativeModule>
NativeModule::load(std::vector<uint8_t> module_bytes, const ExpectedBuildIdentity &expected_identity) {
    validate_elf(module_bytes);
    if (module_bytes.size() > std::numeric_limits<uint32_t>::max()) fail("module exceeds wire size limit");

    int raw_fd = -1;
#if defined(__linux__)
    // MFD_EXEC is 0x10 on kernels that support executable-memfd policy.
    raw_fd = ::syscall(SYS_memfd_create, "simpler-ccu-native", MFD_CLOEXEC | 0x10U);
    if (raw_fd < 0 && errno == EINVAL) raw_fd = ::syscall(SYS_memfd_create, "simpler-ccu-native", MFD_CLOEXEC);
#endif
    if (raw_fd < 0) {
        const char *directory = std::getenv("TMPDIR");
        std::string path =
            std::string(directory != nullptr && *directory != '\0' ? directory : "/tmp") + "/simpler-ccu-native-XXXXXX";
        raw_fd = ::mkstemp(path.data());
        if (raw_fd < 0) throw std::runtime_error("CCU native module temporary file creation failed");
        if (::unlink(path.c_str()) != 0) {
            (void)::close(raw_fd);
            throw std::runtime_error("CCU native module temporary unlink failed");
        }
    }
    FileDescriptor fd(raw_fd);
    write_all(fd.get(), module_bytes);

    const std::string proc_path = "/proc/self/fd/" + std::to_string(fd.get());
    void *handle = ::dlopen(proc_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char *error = ::dlerror();
        throw std::runtime_error("CCU native module load failed: " + std::string(error == nullptr ? "unknown" : error));
    }

    auto module = std::shared_ptr<NativeModule>(new NativeModule());
    module->handle_ = handle;
    // Keep /proc/self/fd/<n> unique among live modules until after dlclose.
    module->backing_fd_ = fd.release();
    module->module_bytes_ = std::move(module_bytes);

    const auto manifest_entry =
        reinterpret_cast<SimplerCcuModuleManifestEntry>(find_symbol(handle, SIMPLER_CCU_MODULE_MANIFEST_SYMBOL));
    const SimplerCcuModuleManifest *manifest = nullptr;
    try {
        manifest = manifest_entry();
    } catch (...) {
        fail("manifest entry threw an exception");
    }
    if (manifest == nullptr) fail("manifest entry returned null");
    if (manifest->struct_size != sizeof(SimplerCcuModuleManifest) ||
        manifest->kernel_record_size != sizeof(SimplerCcuKernelRecord))
        fail("invalid manifest or kernel record size");
    if (manifest->abi_version != SIMPLER_CCU_ABI_VERSION || manifest->reserved != 0)
        fail("unsupported domain-rank ABI version");
    if (manifest->manifest_size < sizeof(SimplerCcuModuleManifest) || manifest->manifest_size > kMaxManifestSize ||
        manifest->kernel_count == 0 || manifest->kernel_count > kMaxKernelCount || manifest->capability_flags != 0)
        fail("invalid manifest fields");
    if (!std::equal(
            std::begin(manifest->sdk_build_id), std::end(manifest->sdk_build_id), expected_identity.sdk.begin()
        ) ||
        !std::equal(
            std::begin(manifest->pto_isa_build_id), std::end(manifest->pto_isa_build_id),
            expected_identity.pto_isa.begin()
        ))
        fail("SDK or PTO-ISA build identity mismatch");

    const uint64_t records_size = static_cast<uint64_t>(manifest->kernel_count) * sizeof(SimplerCcuKernelRecord);
    const uint64_t records_end = static_cast<uint64_t>(manifest->kernel_records_offset) + records_size;
    const uint64_t strings_end = static_cast<uint64_t>(manifest->string_table_offset) + manifest->string_table_size;
    if (manifest->kernel_records_offset < sizeof(SimplerCcuModuleManifest) ||
        manifest->kernel_records_offset % alignof(SimplerCcuKernelRecord) != 0 ||
        records_end > manifest->manifest_size || manifest->string_table_size == 0 ||
        manifest->string_table_offset < sizeof(SimplerCcuModuleManifest) || strings_end > manifest->manifest_size ||
        ranges_overlap(manifest->kernel_records_offset, records_end, manifest->string_table_offset, strings_end))
        fail("manifest record or string-table range is invalid");

    const auto *manifest_data = reinterpret_cast<const uint8_t *>(manifest);
    module->manifest_bytes_.assign(manifest_data, manifest_data + manifest->manifest_size);
    std::set<uint32_t> kernel_ids;
    module->kernels_.reserve(manifest->kernel_count);
    for (uint32_t i = 0; i < manifest->kernel_count; ++i) {
        SimplerCcuKernelRecord record{};
        std::memcpy(
            &record, module->manifest_bytes_.data() + manifest->kernel_records_offset + i * sizeof(record),
            sizeof(record)
        );
        if (record.struct_size != sizeof(SimplerCcuKernelRecord) || record.kernel_id == UINT32_MAX ||
            !kernel_ids.insert(record.kernel_id).second || record.parameter_alignment == 0 ||
            (record.parameter_alignment & (record.parameter_alignment - 1)) != 0 || record.parameter_alignment > 4096 ||
            record.parameter_size > UINT64_MAX - static_cast<uint64_t>(record.parameter_alignment - 1) ||
            record.capability_flags != SIMPLER_CCU_KERNEL_CAP_MASK ||
            record.registration_binding_size != sizeof(SimplerCcuRegistrationBinding) ||
            record.registration_argument_count != SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT ||
            record.task_argument_count != SIMPLER_CCU_TASK_ARGUMENT_COUNT ||
            record.required_rank_count > SIMPLER_CCU_MAX_RANKS || record.reserved != 0)
            fail("invalid kernel ID, parameter, capability, or argument size");
        const uint64_t entry_end = static_cast<uint64_t>(record.entry_symbol_offset) + record.entry_symbol_size;
        if (record.entry_symbol_size == 0 || record.entry_symbol_size > kMaxEntrySymbolSize ||
            record.entry_symbol_offset < manifest->string_table_offset || entry_end > strings_end)
            fail("kernel entry range is invalid");
        const char *entry_begin =
            reinterpret_cast<const char *>(module->manifest_bytes_.data()) + record.entry_symbol_offset;
        const std::string entry_symbol(entry_begin, record.entry_symbol_size);
        if (!valid_symbol(entry_symbol)) fail("kernel entry is not a valid C symbol");
        module->kernels_.push_back({record, entry_symbol, find_symbol(handle, entry_symbol)});
    }

    append_bytes(module->cache_identity_, module->module_bytes_.data(), module->module_bytes_.size());
    append_bytes(module->cache_identity_, module->manifest_bytes_.data(), module->manifest_bytes_.size());
    for (const auto &kernel : module->kernels_)
        append_bytes(module->cache_identity_, kernel.entry_symbol.data(), kernel.entry_symbol.size());
    append_bytes(module->cache_identity_, expected_identity.sdk.data(), expected_identity.sdk.size());
    append_bytes(module->cache_identity_, expected_identity.pto_isa.data(), expected_identity.pto_isa.size());
    return module;
}

std::shared_ptr<NativeModule>
NativeModuleCache::load(std::vector<uint8_t> module_bytes, const ExpectedBuildIdentity &expected_identity) {
    auto candidate = NativeModule::load(std::move(module_bytes), expected_identity);
    std::lock_guard<std::mutex> lock(mutex_);
    auto &entry = modules_[candidate->cache_identity()];
    auto cached = entry.lock();
    if (cached != nullptr) return cached;
    entry = candidate;
    prune_expired();
    return candidate;
}

size_t NativeModuleCache::cached_module_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (const auto &[identity, module] : modules_) {
        (void)identity;
        if (!module.expired()) ++count;
    }
    return count;
}

void NativeModuleCache::prune_expired() {
    for (auto it = modules_.begin(); it != modules_.end();) {
        if (it->second.expired()) it = modules_.erase(it);
        else ++it;
    }
}

NativeModuleCache &native_module_cache() {
    static NativeModuleCache cache;
    return cache;
}

NativeKernelSelection select_native_kernel(
    const std::vector<std::vector<uint8_t>> &modules, uint32_t ccu_id, const ExpectedBuildIdentity &expected_identity,
    NativeModuleCache &cache
) {
    size_t remaining = ccu_id;
    for (const auto &bytes : modules) {
        auto module = cache.load(bytes, expected_identity);
        if (remaining >= module->kernels().size()) {
            remaining -= module->kernels().size();
            continue;
        }
        NativeKernelSelection selected{module, static_cast<uint32_t>(remaining), module->cache_identity()};
        append_bytes(selected.cache_identity, &selected.kernel_index, sizeof(selected.kernel_index));
        return selected;
    }
    throw std::invalid_argument("CCU definition ID is out of range");
}

}  // namespace simpler::ccu
