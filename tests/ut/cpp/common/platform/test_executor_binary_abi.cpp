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
#include "host/executor_binary_abi.h"
#include "../../support/native_admission/executor_elf.h"
#include <gtest/gtest.h>
#include <limits>
#include <string>

namespace {
using namespace simpler::executor_abi;

using native_admission_test::executor_elf;

template <class T>
T read(const std::vector<uint8_t> &bytes, size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}
template <class T>
void write(std::vector<uint8_t> &bytes, size_t offset, const T &value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

TEST(ExecutorBinaryAbi, AcceptsCompleteCurrentPair) {
    EXPECT_NO_THROW(
        validate(executor_elf(SHT_DYNSYM, {AICPU_ENTRY}), executor_elf(SHT_SYMTAB, {AIC_ENTRY, AIV_ENTRY}))
    );
}
TEST(ExecutorBinaryAbi, RejectsOldMixedAndMissingRoleBeforeSubmission) {
    const auto cpu = executor_elf(SHT_DYNSYM, {AICPU_ENTRY});
    const auto core = executor_elf(SHT_SYMTAB, {AIC_ENTRY, AIV_ENTRY});
    const auto old_cpu = executor_elf(SHT_DYNSYM, {"simpler_aicpu_exec"});
    const auto old_core = executor_elf(SHT_SYMTAB, {"aicore_kernel_0_mix_aic", "aicore_kernel_0_mix_aiv"});
    EXPECT_THROW(validate(old_cpu, old_core), std::invalid_argument);
    EXPECT_THROW(validate(old_cpu, core), std::invalid_argument);
    EXPECT_THROW(validate(cpu, old_core), std::invalid_argument);
    EXPECT_THROW(validate(cpu, executor_elf(SHT_SYMTAB, {AIC_ENTRY})), std::invalid_argument);
}
TEST(ExecutorBinaryAbi, DebugStringsAndUndefinedOrNonExecutableSymbolsDoNotAttestEntries) {
    auto bytes = executor_elf(SHT_DYNSYM, {"old_entry"});
    bytes.insert(bytes.end(), AICPU_ENTRY, AICPU_ENTRY + std::strlen(AICPU_ENTRY) + 1);
    EXPECT_FALSE(has_entry(bytes, AICPU_ENTRY, SHT_DYNSYM));
    bytes = executor_elf(SHT_DYNSYM, {AICPU_ENTRY});
    const auto table = read<Elf64_Shdr>(bytes, sizeof(Elf64_Ehdr) + 2 * sizeof(Elf64_Shdr));
    auto symbol = read<Elf64_Sym>(bytes, table.sh_offset + sizeof(Elf64_Sym));
    symbol.st_shndx = SHN_UNDEF;
    write(bytes, table.sh_offset + sizeof(Elf64_Sym), symbol);
    EXPECT_FALSE(has_entry(bytes, AICPU_ENTRY, SHT_DYNSYM));
    bytes = executor_elf(SHT_DYNSYM, {AICPU_ENTRY});
    auto code = read<Elf64_Shdr>(bytes, sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr));
    code.sh_flags = SHF_ALLOC;
    write(bytes, sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr), code);
    EXPECT_FALSE(has_entry(bytes, AICPU_ENTRY, SHT_DYNSYM));
}
TEST(ExecutorBinaryAbi, RejectsTruncationAndOverflowingOffsets) {
    const auto complete = executor_elf(SHT_DYNSYM, {AICPU_ENTRY});
    for (size_t size = 0; size < complete.size(); ++size) {
        const std::vector<uint8_t> truncated(complete.begin(), complete.begin() + size);
        EXPECT_FALSE(has_entry(truncated, AICPU_ENTRY, SHT_DYNSYM));
    }
    auto bytes = complete;
    auto header = read<Elf64_Ehdr>(bytes, 0);
    header.e_shoff = std::numeric_limits<uint64_t>::max();
    write(bytes, 0, header);
    EXPECT_FALSE(has_entry(bytes, AICPU_ENTRY, SHT_DYNSYM));
    bytes = complete;
    auto table = read<Elf64_Shdr>(bytes, sizeof(Elf64_Ehdr) + 2 * sizeof(Elf64_Shdr));
    table.sh_size = std::numeric_limits<uint64_t>::max() - 7;
    write(bytes, sizeof(Elf64_Ehdr) + 2 * sizeof(Elf64_Shdr), table);
    EXPECT_FALSE(has_entry(bytes, AICPU_ENTRY, SHT_DYNSYM));
}
}  // namespace
