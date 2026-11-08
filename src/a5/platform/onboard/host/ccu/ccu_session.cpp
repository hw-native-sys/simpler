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
#include "ccu_session_backend.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <openssl/evp.h>
#include <unistd.h>
#include <utility>

namespace {
struct SessionFailure {
    CcuStatus status;
};
void Require(bool condition, CcuStatus status) {
    if (!condition) throw SessionFailure{status};
}
uint64_t Align(uint64_t value, uint64_t alignment) {
    Require(value <= UINT64_MAX - alignment, CcuStatus::INVALID_ARGUMENT);
    return (value + alignment - 1) / alignment * alignment;
}

class Digest {
public:
    Digest() :
        value_(EVP_MD_CTX_new(), EVP_MD_CTX_free) {
        Require(bool(value_) && EVP_DigestInit_ex(value_.get(), EVP_sha256(), nullptr) == 1, CcuStatus::INIT_FAILED);
    }
    void Bytes(const void *data, size_t size) {
        Require(EVP_DigestUpdate(value_.get(), data, size) == 1, CcuStatus::INIT_FAILED);
    }
    void Number(uint64_t value) {
        uint8_t encoded[8];
        for (uint32_t i = 0; i < 8; ++i)
            encoded[i] = static_cast<uint8_t>(value >> (i * 8));
        Bytes(encoded, sizeof(encoded));
    }
    void Text(const std::string &value) {
        Number(value.size());
        Bytes(value.data(), value.size());
    }
    CcuProgramDigest Finish() {
        CcuProgramDigest result{};
        unsigned int size = 0;
        Require(
            EVP_DigestFinal_ex(value_.get(), result.data(), &size) == 1 && size == result.size(), CcuStatus::INIT_FAILED
        );
        return result;
    }

private:
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> value_;
};

CcuProgramDigest ComputeDigest(const CcuEmbeddedFunction &function) {
    Digest digest;
    digest.Text(ccu_session_sdk_build_id());
    digest.Number(CCU_POSITIONAL_ABI_VERSION);
    digest.Text(function.entry);
    digest.Bytes(function.binary.data(), function.binary.size());
    return digest.Finish();
}

void *LoadEmbeddedModule(const std::vector<uint8_t> &bytes) {
    Require(
        bytes.size() >= 4 && bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F',
        CcuStatus::INVALID_ARGUMENT
    );
    char path[] = "/tmp/simpler-ccu-builder-XXXXXX";
    const int fd = mkstemp(path);
    Require(fd >= 0, CcuStatus::INIT_FAILED);
    size_t written = 0;
    while (written < bytes.size()) {
        const auto count = write(fd, bytes.data() + written, bytes.size() - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            ::close(fd);
            unlink(path);
            throw SessionFailure{CcuStatus::INIT_FAILED};
        }
        written += static_cast<size_t>(count);
    }
    if (::close(fd) != 0) {
        unlink(path);
        throw SessionFailure{CcuStatus::INIT_FAILED};
    }
    void *module = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    unlink(path);
    Require(module != nullptr, CcuStatus::INIT_FAILED);
    return module;
}
}  // namespace

class CcuSession::Impl {
public:
    explicit Impl(CcuSessionConfig value) :
        config(std::move(value)) {}
    struct Memory {
        CcuMemoryRange range{};
        bool prepared{};
    };
    CcuSessionConfig config;
    std::unique_ptr<CcuSessionBackend> backend;
    std::mutex mutex;
    bool attempted{}, initialized{}, closing{}, closed{}, busy{}, dirty{true}, poisoned{};
    uint64_t run_refs{};
    std::atomic<int32_t> error{};
    std::map<uint64_t, Memory> memories;
    std::vector<CcuRegisteredFunction> functions;
    std::map<std::vector<uint8_t>, void *> modules;
    CcuDeviceContextView view{};

    class Claim {
    public:
        explicit Claim(Impl &owner) :
            owner_(owner) {
            std::lock_guard<std::mutex> lock(owner_.mutex);
            Require(!owner_.busy, CcuStatus::BUSY);
            owner_.busy = true;
        }
        ~Claim() {
            std::lock_guard<std::mutex> lock(owner_.mutex);
            owner_.busy = false;
        }

    private:
        Impl &owner_;
    };
    template <typename Fn>
    CcuStatus Protect(Fn &&fn) noexcept {
        try {
            Claim claim(*this);
            error = 0;
            fn();
            return CcuStatus::OK;
        } catch (const SessionFailure &e) {
            return e.status;
        } catch (const CcuBackendError &e) {
            error = e.native_error;
            std::fprintf(stderr, "CCU backend error %d: %s\n", e.native_error, e.what());
            return e.status;
        } catch (const CcuSdkError &e) {
            error = e.code;
            std::fprintf(stderr, "CCU SDK error %d: %s\n", e.code, e.what());
            return CcuStatus::INIT_FAILED;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "CCU session exception: %s\n", e.what());
            return CcuStatus::INIT_FAILED;
        } catch (...) {
            return CcuStatus::INIT_FAILED;
        }
    }
    void Editable() {
        Require(initialized && !closing && !poisoned, CcuStatus::STALE_HANDLE);
        Require(run_refs == 0, CcuStatus::BUSY);
        try {
            Require(backend->Idle(), CcuStatus::BUSY);
        } catch (const CcuSdkError &e) {
            poisoned = true;
            throw CcuBackendError(CcuStatus::RECLAMATION_UNCONFIRMED, e.code, e.what());
        }
    }
    void CloseModules() {
        while (!modules.empty()) {
            const auto it = modules.begin();
            if (it->second) Require(dlclose(it->second) == 0, CcuStatus::RECLAMATION_UNCONFIRMED);
            modules.erase(it);
        }
    }
    CcuBuildEntry LoadEntry(const CcuEmbeddedFunction &function) {
        auto [it, inserted] = modules.try_emplace(function.binary, nullptr);
        if (inserted) it->second = LoadEmbeddedModule(it->first);
        const auto version = reinterpret_cast<uint32_t (*)()>(dlsym(it->second, "simpler_ccu_module_abi_version"));
        Require(version && version() == CCU_POSITIONAL_ABI_VERSION, CcuStatus::ABI_MISMATCH);
        const auto entry = reinterpret_cast<CcuBuildEntry>(dlsym(it->second, function.entry.c_str()));
        Require(entry != nullptr, CcuStatus::INVALID_ARGUMENT);
        return entry;
    }
    std::vector<CcuImageBlock> Image() const {
        Require(!functions.empty(), CcuStatus::INVALID_ARGUMENT);
        CcuDeviceContext header{};
        header.abi_version = CCU_DEVICE_ABI_VERSION;
        header.struct_size = sizeof(header);
        header.allocation_id = config.allocation_id;
        header.domain_owner = config.domain_owner;
        header.callable_id = config.callable_id;
        header.session_epoch = config.epoch;
        header.next_sequence = 1;
        header.rank = config.rank;
        header.rank_count = config.rank_count;
        header.timeout_ns = uint64_t{config.timeout_ms} * 1000000;
        header.admission_offset = sizeof(header);
        header.bindings_offset = sizeof(header) + sizeof(CcuAdmissionRecord);
        header.binding_count = functions.size();
        header.arguments_size = sizeof(CcuPositionalArguments);
        uint64_t cursor = header.bindings_offset + functions.size() * sizeof(CcuProgramBinding);
        header.ranges_offset = cursor;
        for (const auto &m : memories)
            if (m.second.prepared) ++header.range_count;
        cursor += header.range_count * sizeof(CcuMemoryRange);
        header.record_offset = Align(cursor, 128);
        cursor = header.record_offset + sizeof(CcuRequestRecord);
        header.arguments_offset = cursor;
        header.image_size = Align(cursor + header.arguments_size, 128);
        std::vector<CcuImageBlock> image(header.image_size / sizeof(CcuImageBlock));
        auto *data = reinterpret_cast<uint8_t *>(image.data());
        std::memcpy(data, &header, sizeof(header));
        auto *bindings = reinterpret_cast<CcuProgramBinding *>(data + header.bindings_offset);
        for (size_t i = 0; i < functions.size(); ++i)
            bindings[i] = {config.epoch, 0, i, header.arguments_size, 0};
        cursor = header.ranges_offset;
        for (const auto &m : memories)
            if (m.second.prepared) {
                std::memcpy(data + cursor, &m.second.range, sizeof(CcuMemoryRange));
                cursor += sizeof(CcuMemoryRange);
            }
        return image;
    }
};

CcuSession::CcuSession(CcuSessionConfig config) :
    impl_(std::make_unique<Impl>(std::move(config))) {}
CcuSession::~CcuSession() {
    if (impl_ && impl_->attempted && !impl_->closed) (void)impl_.release();
}
CcuStatus CcuSession::initialize() {
    return impl_->Protect([&] {
        Require(!impl_->attempted && !impl_->closed, CcuStatus::BUSY);
        const auto &c = impl_->config;
        Require(
            c.epoch && c.device_id >= 0 && c.rank_count == 2 && c.rank < 2 && c.timeout_ms && c.parent_owner &&
                c.exchange,
            CcuStatus::INVALID_ARGUMENT
        );
        impl_->backend = ccu_make_session_backend();
        impl_->attempted = true;
        impl_->backend->Initialize(c);
        impl_->initialized = true;
    });
}
CcuStatus CcuSession::register_functions(const std::vector<CcuEmbeddedFunction> &functions) {
    return impl_->Protect([&] {
        impl_->Editable();
        Require(impl_->functions.empty() && !impl_->view.addr, CcuStatus::BUSY);
        Require(!functions.empty() && functions.size() <= UINT32_MAX, CcuStatus::INVALID_ARGUMENT);
        for (const auto &function : functions) {
            Require(
                function.kind == CcuEmbeddedModuleKind::HOST_BUILDER &&
                    function.abi_version == CCU_POSITIONAL_ABI_VERSION,
                CcuStatus::ABI_MISMATCH
            );
            Require(
                !function.binary.empty() && !function.entry.empty() && function.entry.find('\0') == std::string::npos,
                CcuStatus::INVALID_ARGUMENT
            );
        }
        std::vector<CcuRegisteredFunction> registered;
        registered.reserve(functions.size());
        try {
            for (const auto &function : functions)
                registered.push_back({impl_->LoadEntry(function), ComputeDigest(function)});
        } catch (...) {
            try {
                impl_->CloseModules();
            } catch (...) {
                impl_->poisoned = true;
                throw;
            }
            throw;
        }
        impl_->functions = std::move(registered);
    });
}
CcuStatus CcuSession::prepare_memory(uint64_t id, uint64_t addr, uint64_t size) {
    return impl_->Protect([&] {
        impl_->Editable();
        Require(addr && size && size <= UINT64_MAX - addr, CcuStatus::INVALID_ARGUMENT);
        for (const auto &memory : impl_->memories) {
            const auto &range = memory.second.range;
            Require(
                memory.first == id || addr >= range.addr + range.byte_size || range.addr >= addr + size,
                CcuStatus::INVALID_ARGUMENT
            );
        }
        const auto it = impl_->memories.try_emplace(id, Impl::Memory{{id, addr, size, 0}, false}).first;
        Require(it->second.range.addr == addr && it->second.range.byte_size == size, CcuStatus::INVALID_ARGUMENT);
        if (it->second.prepared) return;
        it->second.range.token = impl_->backend->PrepareMemory(addr, size);
        it->second.prepared = true;
        impl_->dirty = true;
    });
}
CcuStatus CcuSession::get_device_context(CcuDeviceContextView &out) const {
    out = {};
    return impl_->Protect([&] {
        Require(impl_->initialized && !impl_->closing && !impl_->poisoned, CcuStatus::STALE_HANDLE);
        if (impl_->dirty) {
            impl_->Editable();
            auto image = impl_->Image();
            try {
                impl_->backend->Publish(impl_->functions, std::move(image), impl_->view);
            } catch (...) {
                impl_->poisoned = true;
                throw;
            }
            impl_->dirty = false;
        }
        out = impl_->view;
    });
}
CcuStatus CcuSession::retain_run() {
    return impl_->Protect([&] {
        Require(
            impl_->initialized && !impl_->closing && !impl_->dirty && !impl_->poisoned && impl_->view.addr,
            CcuStatus::STALE_HANDLE
        );
        Require(impl_->run_refs == 0, CcuStatus::BUSY);
        ++impl_->run_refs;
    });
}
CcuStatus CcuSession::release_run() {
    return impl_->Protect([&] {
        Require(impl_->run_refs != 0, CcuStatus::INVALID_ARGUMENT);
        --impl_->run_refs;
    });
}
CcuPollResult CcuSession::poll_idle() {
    bool idle = false;
    const auto status = impl_->Protect([&] {
        Require(impl_->initialized && !impl_->closed && !impl_->poisoned, CcuStatus::STALE_HANDLE);
        idle = impl_->backend->Idle();
    });
    if (status != CcuStatus::OK) return {CcuRequestState::UNCONFIRMED, status, native_error()};
    return {idle ? CcuRequestState::SUCCEEDED : CcuRequestState::PENDING, CcuStatus::OK, 0};
}
CcuStatus CcuSession::drain(uint64_t timeout_ms) {
    return impl_->Protect([&] {
        Require(impl_->initialized && !impl_->closed && !impl_->poisoned, CcuStatus::STALE_HANDLE);
        Require(impl_->backend->Drain(timeout_ms), CcuStatus::RECLAMATION_UNCONFIRMED);
    });
}
CcuStatus CcuSession::close(CcuCloseMode mode) {
    return impl_->Protect([&] {
        if (impl_->closed) return;
        Require(mode == CcuCloseMode::NORMAL || mode == CcuCloseMode::ABORT_UNPUBLISHED, CcuStatus::INVALID_ARGUMENT);
        if (mode == CcuCloseMode::ABORT_UNPUBLISHED) Require(!impl_->view.addr, CcuStatus::INVALID_ARGUMENT);
        impl_->closing = true;
        try {
            if (impl_->backend) impl_->backend->PreventRequests();
        } catch (const CcuSdkError &e) {
            throw CcuBackendError(CcuStatus::RECLAMATION_UNCONFIRMED, e.code, e.what());
        }
        Require(impl_->run_refs == 0, CcuStatus::BUSY);
        if (impl_->backend) {
            try {
                Require(impl_->backend->Stop(impl_->config.timeout_ms), CcuStatus::RECLAMATION_UNCONFIRMED);
                impl_->backend->Close();
            } catch (const CcuSdkError &e) {
                throw CcuBackendError(CcuStatus::RECLAMATION_UNCONFIRMED, e.code, e.what());
            }
        }
        impl_->CloseModules();
        impl_->functions.clear();
        impl_->memories.clear();
        impl_->closed = true;
        impl_->config.exchange = {};
        impl_->config.parent_owner.reset();
    });
}
int32_t CcuSession::native_error() const { return impl_->error.load(); }
