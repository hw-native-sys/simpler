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

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <unistd.h>
#include <cerrno>
#include <openssl/evp.h>

#include "host/ccu_device.h"

namespace {
struct SessionFailure {
    CcuStatus status;
};
void Require(bool condition, CcuStatus status) {
    if (!condition) throw SessionFailure{status};
}
bool SameKey(const CcuProgramKey &a, const CcuProgramKey &b) { return std::memcmp(a.bytes, b.bytes, 32) == 0; }
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
    CcuProgramKey Finish() {
        CcuProgramKey result{};
        unsigned int size = 0;
        Require(EVP_DigestFinal_ex(value_.get(), result.bytes, &size) == 1 && size == 32, CcuStatus::INIT_FAILED);
        return result;
    }

private:
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> value_;
};

std::vector<uint8_t> LibraryBytes(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    Require(bool(file), CcuStatus::INVALID_ARGUMENT);
    const auto size = file.tellg();
    Require(size > 0 && static_cast<uint64_t>(size) <= SIZE_MAX, CcuStatus::INVALID_ARGUMENT);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.seekg(0);
    Require(bool(file.read(reinterpret_cast<char *>(bytes.data()), bytes.size())), CcuStatus::INVALID_ARGUMENT);
    return bytes;
}

CcuProgramKey ComputeKey(const CcuFunctionSpec &f, const std::vector<uint8_t> *library_bytes = nullptr) {
    Digest digest;
    digest.Text("simpler.ccu.program.v1");
    digest.Number(CCU_BUILD_ABI_VERSION);
    digest.Text(ccu_session_sdk_build_id());
    digest.Text(f.entry);
    if (f.positional_id) {
        digest.Number(CCU_POSITIONAL_ABI_VERSION);
        digest.Number(f.positional_id);
    }
    if (!f.module_binary.empty()) {
        digest.Number(f.module_binary.size());
        digest.Bytes(f.module_binary.data(), f.module_binary.size());
    } else if (f.library_path.empty()) digest.Text("simpler.builtin.v1");
    else {
        const auto owned_bytes = library_bytes ? std::vector<uint8_t>{} : LibraryBytes(f.library_path);
        const auto &bytes = library_bytes ? *library_bytes : owned_bytes;
        digest.Number(bytes.size());
        digest.Bytes(bytes.data(), bytes.size());
    }
    const auto tensors = [&](const auto &parameters) {
        digest.Number(parameters.size());
        for (const auto &p : parameters) {
            digest.Text(p.name);
            digest.Number(static_cast<uint32_t>(p.type));
        }
    };
    tensors(f.signature.inputs);
    tensors(f.signature.outputs);
    tensors(f.signature.workspaces);
    tensors(f.signature.scalars);
    auto aliases = f.signature.aliases;
    for (auto &a : aliases)
        if (a.first > a.second) std::swap(a.first, a.second);
    std::sort(aliases.begin(), aliases.end());
    digest.Number(aliases.size());
    for (const auto &a : aliases) {
        digest.Text(a.first);
        digest.Text(a.second);
    }
    return digest.Finish();
}

CcuRegisteredFunction MakeFunction(const CcuFunctionSpec &spec) {
    CcuRegisteredFunction function;
    function.spec = spec;
    if (spec.positional_id) {
        function.argument_bytes = sizeof(CcuPositionalArguments);
        return function;
    }
    uint32_t words = 0;
    const auto add = [&](const auto &parameters, CcuArgumentKind kind) {
        for (const auto &p : parameters) {
            Require(
                !p.name.empty() &&
                    std::find(function.names.begin(), function.names.end(), p.name) == function.names.end(),
                CcuStatus::INVALID_ARGUMENT
            );
            Require(
                (kind == CcuArgumentKind::TENSOR && p.type == CcuArgumentType::FLOAT32) ||
                    (kind == CcuArgumentKind::SCALAR &&
                     (p.type == CcuArgumentType::UINT64 || p.type == CcuArgumentType::INT64)),
                CcuStatus::UNSUPPORTED
            );
            function.names.push_back(p.name);
            function.parameters.push_back({kind, p.type, words, 0, 0});
            words += kind == CcuArgumentKind::TENSOR ? 3 : 1;
        }
    };
    add(spec.signature.inputs, CcuArgumentKind::TENSOR);
    add(spec.signature.outputs, CcuArgumentKind::TENSOR);
    add(spec.signature.workspaces, CcuArgumentKind::TENSOR);
    add(spec.signature.scalars, CcuArgumentKind::SCALAR);
    Require(!function.parameters.empty() && function.parameters.size() <= 64, CcuStatus::UNSUPPORTED);
    for (const auto &alias : spec.signature.aliases) {
        const auto first = std::find(function.names.begin(), function.names.end(), alias.first);
        const auto second = std::find(function.names.begin(), function.names.end(), alias.second);
        Require(
            first != function.names.end() && second != function.names.end() && first != second,
            CcuStatus::INVALID_ARGUMENT
        );
        const auto i = static_cast<size_t>(first - function.names.begin()),
                   j = static_cast<size_t>(second - function.names.begin());
        Require(
            function.parameters[i].kind == CcuArgumentKind::TENSOR &&
                function.parameters[j].kind == CcuArgumentKind::TENSOR,
            CcuStatus::INVALID_ARGUMENT
        );
        function.parameters[i].allowed_aliases |= uint64_t{1} << j;
        function.parameters[j].allowed_aliases |= uint64_t{1} << i;
    }
    function.argument_bytes = words * sizeof(uint64_t);
    return function;
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

CcuStatus ccu_compute_program_key(const CcuFunctionSpec &function, CcuProgramKey &out) {
    out = {};
    try {
        out = ComputeKey(function);
        return CcuStatus::OK;
    } catch (const SessionFailure &f) {
        return f.status;
    } catch (...) {
        return CcuStatus::INIT_FAILED;
    }
}
CcuFunctionSpec ccu_builtin_broadcast(uint32_t root) {
    CcuFunctionSpec spec;
    spec.name = "simpler.broadcast";
    spec.entry = root == 0 ? "builtin.broadcast.root0" : "builtin.broadcast.root1";
    if (root > 1) return {};
    spec.signature.inputs.push_back({"src", CcuArgumentType::FLOAT32});
    spec.signature.outputs.push_back({"dst", CcuArgumentType::FLOAT32});
    spec.signature.scalars.push_back({"count", CcuArgumentType::UINT64});
    spec.signature.aliases.emplace_back("src", "dst");
    (void)ccu_compute_program_key(spec, spec.program_key);
    return spec;
}

class CcuSession::Impl {
public:
    explicit Impl(CcuSessionConfig value) :
        config(std::move(value)) {}
    struct EmbeddedModule {
        void *handle{};
    };
    struct FunctionOwner {
        CcuRegisteredFunction function;
        void *module{};
        std::shared_ptr<EmbeddedModule> embedded_module;
        bool committed{};
        bool ready{};
        std::set<uint64_t> registrations;
    };
    struct Memory {
        CcuMemoryRange range{};
        bool prepared{};
    };
    CcuSessionConfig config;
    std::unique_ptr<CcuSessionBackend> backend;
    std::mutex mutex;
    bool attempted{}, initialized{}, closing{}, closed{}, busy{}, dirty{true}, poisoned{};
    bool positional{};
    uint64_t run_refs{};
    std::atomic<int32_t> error{};
    std::vector<FunctionOwner> functions;
    std::map<std::vector<uint8_t>, std::weak_ptr<EmbeddedModule>> modules;
    std::map<uint64_t, size_t> registrations;
    std::map<uint64_t, Memory> memories;
    std::set<uint64_t> retired_memories;
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
    void ReleaseModule(FunctionOwner &function) {
        if (!function.module) return;
        if (!function.embedded_module || function.embedded_module.use_count() == 1)
            Require(dlclose(function.module) == 0, CcuStatus::RECLAMATION_UNCONFIRMED);
        function.module = nullptr;
        function.embedded_module.reset();
    }
    void PruneModules() {
        for (auto it = modules.begin(); it != modules.end();) {
            if (it->second.expired()) it = modules.erase(it);
            else ++it;
        }
    }
    void Register(uint64_t id, const CcuFunctionSpec &spec, CcuProgramRef &out) {
        Editable();
        Require(id != 0 && !spec.entry.empty(), CcuStatus::INVALID_ARGUMENT);
        const auto path_bytes = spec.module_binary.empty() && !spec.library_path.empty() ?
                                    LibraryBytes(spec.library_path) :
                                    std::vector<uint8_t>{};
        Require(SameKey(ComputeKey(spec, &path_bytes), spec.program_key), CcuStatus::ABI_MISMATCH);
        const auto known = registrations.find(id);
        if (known != registrations.end()) {
            Require(
                functions[known->second].ready &&
                    SameKey(functions[known->second].function.spec.program_key, spec.program_key),
                CcuStatus::INVALID_ARGUMENT
            );
            out = {0, config.epoch, known->second};
            return;
        }
        for (size_t i = 0; i < functions.size(); ++i) {
            auto &f = functions[i];
            if (f.ready && SameKey(f.function.spec.program_key, spec.program_key)) {
                f.registrations.insert(id);
                try {
                    registrations.emplace(id, i);
                } catch (...) {
                    f.registrations.erase(id);
                    throw;
                }
                out = {0, config.epoch, i};
                return;
            }
        }
        Impl::FunctionOwner owner;
        owner.function = MakeFunction(spec);
        owner.registrations.insert(id);
        functions.push_back(std::move(owner));
        const auto index = functions.size() - 1;
        try {
            registrations.emplace(id, index);
        } catch (...) {
            functions.pop_back();
            throw;
        }
        auto &f = functions.back();
        if (spec.library_path.empty() && spec.module_binary.empty()) {
            Require(
                spec.entry == "builtin.broadcast.root0" || spec.entry == "builtin.broadcast.root1",
                CcuStatus::UNSUPPORTED
            );
            const auto canonical = ccu_builtin_broadcast(spec.entry == "builtin.broadcast.root0" ? 0 : 1);
            Require(SameKey(canonical.program_key, spec.program_key), CcuStatus::ABI_MISMATCH);
        } else {
            // The key and loaded artifact consume the same immutable byte snapshot.
            const auto &bytes = spec.module_binary.empty() ? path_bytes : spec.module_binary;
            auto &cached = modules[bytes];
            f.embedded_module = cached.lock();
            if (!f.embedded_module) {
                f.embedded_module = std::make_shared<EmbeddedModule>();
                f.embedded_module->handle = LoadEmbeddedModule(bytes);
                cached = f.embedded_module;
            }
            f.module = f.embedded_module->handle;
            Require(f.module != nullptr, CcuStatus::INIT_FAILED);
            if (spec.positional_id) {
                const auto version =
                    reinterpret_cast<uint32_t (*)()>(dlsym(f.module, "simpler_ccu_module_abi_version"));
                Require(version && version() == CCU_POSITIONAL_ABI_VERSION, CcuStatus::ABI_MISMATCH);
            } else {
                const auto *abi = static_cast<const CcuBuildModuleAbi *>(dlsym(f.module, "ccu_build_module_abi"));
                Require(
                    abi && abi->abi_version == CCU_BUILD_ABI_VERSION && abi->struct_size == sizeof(*abi) &&
                        abi->sdk_build_id && std::strcmp(abi->sdk_build_id, ccu_session_sdk_build_id()) == 0,
                    CcuStatus::ABI_MISMATCH
                );
            }
            f.function.entry = reinterpret_cast<CcuBuildEntry>(dlsym(f.module, spec.entry.c_str()));
            Require(f.function.entry != nullptr, CcuStatus::INVALID_ARGUMENT);
        }
        f.ready = true;
        dirty = true;
        out = {0, config.epoch, index};
    }
    std::vector<CcuImageBlock> Image(std::vector<CcuRegisteredFunction> &active) {
        for (const auto &owner : functions)
            if (owner.ready && (owner.committed || !owner.registrations.empty())) active.push_back(owner.function);
        Require(!active.empty(), CcuStatus::INVALID_ARGUMENT);
        std::sort(active.begin(), active.end(), [](const auto &a, const auto &b) {
            if (a.spec.positional_id || b.spec.positional_id) return a.spec.positional_id < b.spec.positional_id;
            return std::memcmp(a.spec.program_key.bytes, b.spec.program_key.bytes, 32) < 0;
        });
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
        header.binding_count = active.size();
        uint64_t cursor = header.bindings_offset + active.size() * sizeof(CcuProgramBinding);
        std::vector<CcuProgramBinding> bindings;
        for (size_t i = 0; i < active.size(); ++i) {
            const auto &f = active[i];
            bindings.push_back(
                {f.spec.program_key, config.epoch, cursor, 0, i, static_cast<uint32_t>(f.parameters.size()),
                 f.argument_bytes, f.spec.positional_id}
            );
            cursor += f.parameters.size() * sizeof(CcuParameterLayout);
            header.arguments_size = std::max(header.arguments_size, f.argument_bytes);
        }
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
        std::memcpy(data + header.bindings_offset, bindings.data(), bindings.size() * sizeof(CcuProgramBinding));
        for (size_t i = 0; i < active.size(); ++i)
            std::memcpy(
                data + bindings[i].signature_offset, active[i].parameters.data(),
                active[i].parameters.size() * sizeof(CcuParameterLayout)
            );
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
CcuStatus CcuSession::register_function(uint64_t id, const CcuFunctionSpec &spec, CcuProgramRef &out) {
    out = {};
    return impl_->Protect([&] {
        Require(!impl_->positional && !spec.positional_id, CcuStatus::INVALID_ARGUMENT);
        impl_->Register(id, spec, out);
    });
}
CcuStatus CcuSession::register_functions(const std::vector<CcuEmbeddedFunction> &functions) {
    return impl_->Protect([&] {
        impl_->Editable();
        Require(impl_->registrations.empty() && !impl_->view.addr, CcuStatus::BUSY);
        Require(!functions.empty() && functions.size() <= UINT32_MAX, CcuStatus::INVALID_ARGUMENT);
        std::vector<CcuFunctionSpec> specs;
        specs.reserve(functions.size());
        for (size_t i = 0; i < functions.size(); ++i) {
            const auto &function = functions[i];
            Require(
                function.kind == CcuEmbeddedModuleKind::HOST_BUILDER &&
                    function.abi_version == CCU_POSITIONAL_ABI_VERSION,
                CcuStatus::ABI_MISMATCH
            );
            Require(
                !function.binary.empty() && !function.entry.empty() && function.entry.find('\0') == std::string::npos,
                CcuStatus::INVALID_ARGUMENT
            );
            CcuFunctionSpec spec;
            spec.entry = function.entry;
            spec.module_binary = function.binary;
            spec.positional_id = i + 1;
            spec.program_key = ComputeKey(spec);
            specs.push_back(std::move(spec));
        }
        try {
            for (size_t i = 0; i < specs.size(); ++i) {
                CcuProgramRef reference{};
                impl_->Register(i + 1, specs[i], reference);
            }
        } catch (...) {
            // No image was published; no hardware can reference these modules.
            for (auto &function : impl_->functions) {
                if (function.module) {
                    try {
                        impl_->ReleaseModule(function);
                    } catch (...) {
                        impl_->poisoned = true;
                        throw;
                    }
                }
            }
            impl_->functions.clear();
            impl_->registrations.clear();
            impl_->PruneModules();
            throw;
        }
        impl_->positional = true;
    });
}
CcuStatus CcuSession::abort_registration(uint64_t id) {
    return impl_->Protect([&] {
        Require(!impl_->positional, CcuStatus::INVALID_ARGUMENT);
        Require(impl_->run_refs == 0, CcuStatus::BUSY);
        Require(!impl_->poisoned, CcuStatus::RECLAMATION_UNCONFIRMED);
        const auto found = impl_->registrations.find(id);
        if (found == impl_->registrations.end()) return;
        auto &f = impl_->functions[found->second];
        if (!f.committed && f.registrations.size() == 1 && f.module) {
            impl_->ReleaseModule(f);
        }
        f.registrations.erase(id);
        impl_->registrations.erase(found);
        if (!f.committed && f.registrations.empty()) {
            f.ready = false;
            CcuRegisteredFunction released;
            std::swap(f.function, released);
            f.embedded_module.reset();
            impl_->PruneModules();
            impl_->dirty = true;
        }
    });
}
CcuStatus CcuSession::prepare_memory(uint64_t id, uint64_t addr, uint64_t size) {
    return impl_->Protect([&] {
        impl_->Editable();
        Require(addr && size && size <= UINT64_MAX - addr, CcuStatus::INVALID_ARGUMENT);
        Require(impl_->memories.count(id) != 0 || impl_->retired_memories.count(id) == 0, CcuStatus::INVALID_ARGUMENT);
        for (const auto &memory : impl_->memories) {
            const auto &range = memory.second.range;
            Require(
                memory.first == id || addr >= range.addr + range.byte_size || range.addr >= addr + size,
                CcuStatus::INVALID_ARGUMENT
            );
        }
        auto [it, inserted] = impl_->memories.emplace(id, Impl::Memory{{id, addr, size, 0}, false});
        Require(it->second.range.addr == addr && it->second.range.byte_size == size, CcuStatus::INVALID_ARGUMENT);
        if (it->second.prepared) return;
        it->second.range.token = impl_->backend->PrepareMemory(addr, size);
        it->second.prepared = true;
        impl_->dirty = true;
        (void)inserted;
    });
}
CcuStatus CcuSession::release_memory(uint64_t id) {
    return impl_->Protect([&] {
        impl_->Editable();
        const auto found = impl_->memories.find(id);
        Require(found != impl_->memories.end(), CcuStatus::INVALID_ARGUMENT);
        impl_->retired_memories.insert(id);
        try {
            impl_->backend->ReleaseMemory(id);
        } catch (const CcuSdkError &e) {
            impl_->poisoned = true;
            throw CcuBackendError(CcuStatus::RECLAMATION_UNCONFIRMED, e.code, e.what());
        }
        impl_->memories.erase(found);
        impl_->dirty = true;
    });
}
CcuStatus CcuSession::get_device_context(CcuDeviceContextView &out) const {
    out = {};
    return impl_->Protect([&] {
        Require(impl_->initialized && !impl_->closing && !impl_->poisoned, CcuStatus::STALE_HANDLE);
        if (impl_->dirty) {
            impl_->Editable();
            std::vector<CcuRegisteredFunction> active;
            auto image = impl_->Image(active);
            try {
                impl_->backend->Publish(active, std::move(image), impl_->view);
            } catch (...) {
                impl_->poisoned = true;
                throw;
            }
            for (auto &f : impl_->functions)
                if (f.ready && !f.registrations.empty()) f.committed = true;
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
CcuStatus CcuSession::quiesce(uint64_t timeout) {
    return impl_->Protect([&] {
        impl_->closing = true;
        try {
            if (impl_->backend) impl_->backend->PreventRequests();
        } catch (const CcuSdkError &e) {
            throw CcuBackendError(CcuStatus::RECLAMATION_UNCONFIRMED, e.code, e.what());
        }
        Require(impl_->run_refs == 0, CcuStatus::BUSY);
        Require(impl_->backend != nullptr, CcuStatus::STALE_HANDLE);
        try {
            Require(impl_->backend->Stop(timeout), CcuStatus::RECLAMATION_UNCONFIRMED);
        } catch (const CcuSdkError &e) {
            throw CcuBackendError(CcuStatus::RECLAMATION_UNCONFIRMED, e.code, e.what());
        }
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
        for (auto &f : impl_->functions)
            if (f.module) {
                impl_->ReleaseModule(f);
            }
        impl_->memories.clear();
        impl_->closed = true;
        impl_->config.exchange = {};
        impl_->config.parent_owner.reset();
    });
}
int32_t CcuSession::native_error() const { return impl_->error.load(); }
