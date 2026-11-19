# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""WSE tensor bounds, Shared slot publication and Block lifetime on Linx."""

import subprocess
from pathlib import Path

import pytest

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_SOURCE = r"""
#include <wse/tile_ops.h>
#include <array>
#include <atomic>
#include <cassert>
#include <latch>
#include <limits>
#include <thread>

struct Block {
    uint64_t epoch = 1;
    wse::SharedSlotTable slots{};
    wse::BlockScope scope() { return {&epoch, epoch, nullptr, 0, &slots}; }
};
using Plane = pto::Tile<pto::Location::Vec, float, 4, 8, pto::BLayout::RowMajor, 2, 3>;
using Slot = wse::SharedTileSlot<3, Plane>;
auto memory(std::array<float, 6> &data) { return wse::tile_memory<2, 3>(data.data(), data.size()); }
float read_last(Slot &slot, uint64_t generation) {
    std::array<float, 6> elements{};
    auto out = memory(elements);
    auto tile = slot.read(generation);
    TSTORE(out, tile);
    return elements.back();
}

template <class Fn>
void rejects(Fn call) {
    bool caught = false;
    try {
        call();
    } catch (const std::exception &) {
        caught = true;
    }
    assert(caught);
}

void ranges() {
    float data[] = {1, 2, 3, 77, 4, 5, 6};
    std::array<float, 10> out;
    out.fill(-1);
    auto source = wse::tile_memory<2, 3, 4>(data, 7);
    auto destination = wse::tile_memory<2, 3, 5>(out.data(), out.size(), 1);
    Plane tile;
    TLOAD(tile, source);
    TSTORE(destination, tile);
    assert((out == std::array<float, 10>{-1, 1, 2, 3, -1, -1, 4, 5, 6, -1}));
    rejects([&] {
        wse::tile_memory<2, 3>(data, 5);
    });
    rejects([&] {
        wse::tile_memory<2, 3, 4>(data, 6);
    });
    rejects([&] {
        wse::tile_memory<2, 3>(static_cast<float *>(nullptr), 6);
    });
    rejects([&] {
        wse::tile_memory<2, 3>(data, 6, std::numeric_limits<size_t>::max());
    });
    rejects([&] {
        wse::tile_memory<2, 3>(data, 6, 1);
    });
}
void shared() {
    auto launch = [](unsigned core) {
        Block block;
        pto::linx_cpu::runCore([&] {
            Slot slot(core, 15, block.scope());
            for (unsigned gen = 1; gen <= 4; ++gen) {
                std::array<float, 6> value{};
                value.back() = core * 100 + gen;
                if (get_thread_idx() == core) slot.publish(gen, memory(value));
                assert(read_last(slot, gen) == core * 100 + gen);
            }
        });
    };
    std::array<std::thread, 4> cores;
    for (unsigned core = 0; core < 4; ++core)
        cores[core] = std::thread(launch, core);
    for (auto &thread : cores)
        thread.join();
}
void overwrite_wait() {
    Block block;
    std::atomic<bool> waiting{false};
    auto trace = [](void *object, wse::OperationEvent event, uint32_t, uint64_t generation) {
        if (event == wse::OperationEvent::WAIT && generation == 2 && get_thread_idx() == 0) {
            auto &flag = *static_cast<std::atomic<bool> *>(object);
            flag.store(true);
            flag.notify_all();
        }
    };
    pto::linx_cpu::runCore([&] {
        Slot slot(0, 6, block.scope(), {&waiting, trace});
        auto pe = get_thread_idx();
        if (pe == 0) {
            std::array<float, 6> values{};
            values.back() = 11;
            slot.publish(1, memory(values));
            values.back() = 22;
            slot.publish(2, memory(values));
        } else if (pe < 3) {
            waiting.wait(false);
            assert(read_last(slot, 1) == 11);
            assert(read_last(slot, 2) == 22);
        }
    });
    Block next;
    std::latch first_read(1);
    pto::linx_cpu::runCore([&] {
        Slot slot(0, 6, next.scope());
        if (get_thread_idx() == 0) {
            std::array<float, 6> value{};
            slot.publish(1, memory(value));
            first_read.wait();
            rejects([&] { slot.publish(2, memory(value)); });
        } else if (get_thread_idx() == 1) {
            slot.read(1);
            first_read.count_down();
        }
        // PE 2 exits without consuming generation 1.
    });
}
void context_errors() {
    Block block;
    std::array<float, 6> values{};
    rejects([&] {
        Slot slot(0, 1, block.scope());
    });
    pto::linx_cpu::runCore(
        [&] {
            Slot slot(0, 1, block.scope());
            rejects([&] {
                Slot invalid(4, 1, block.scope());
            });
            rejects([&] {
                Slot invalid(0, 0, block.scope());
            });
            rejects([&] {
                Slot invalid(0, 2, block.scope());
            });
            rejects([&] {
                Slot invalid(0, 1, {&block.epoch, block.epoch});
            });
            using Wrong = pto::Tile<pto::Location::Vec, float, 4, 8, pto::BLayout::RowMajor, 3, 2>;
            rejects([&] {
                wse::SharedTileSlot<3, Wrong> mismatch(0, 1, block.scope());
            });
            using Padded = pto::Tile<pto::Location::Vec, float, 4, 8, pto::BLayout::RowMajor, 2, 3,
                                     pto::SLayout::NoneBox, 512, pto::PadValue::Zero>;
            rejects([&] {
                wse::SharedTileSlot<3, Padded> mismatch(0, 1, block.scope());
            });
            rejects([&] {
                slot.publish(2, memory(values));
            });
            rejects([&] {
                slot.read(1);
            });
            rejects([&] {
                slot.read(2);
            });
            slot.publish(1, memory(values));
            rejects([&] {
                slot.publish(2, memory(values));
            });
            assert(read_last(slot, 1) == 0);
            rejects([&] {
                slot.read(1);
            });
        },
        {1, {}}
    );
    block.slots = {};
    pto::linx_cpu::runCore(
        [&] {
            Slot slot(0, 2, block.scope());
            if (get_thread_idx() == 1) rejects([&] { slot.read(1); });
        },
        {2, {}}
    );
}
void block_lifetime() {
    uint64_t run_epoch = 1, block_serial = 1;
    wse::BlockScope recorded{&block_serial, 1, &run_epoch, 1};
    recorded.validate();
    run_epoch = (uint64_t{1} << 32) + 1;
    rejects([&] {
        recorded.validate();
    });

    Block block;
    for (int repeat = 1; repeat <= 2; ++repeat) {
        pto::linx_cpu::runCore(
            [&] {
                Slot slot(0, 1, block.scope());
                std::array<float, 6> value{};
                value.back() = repeat;
                slot.publish(1, memory(value));
                assert(read_last(slot, 1) == repeat);
                ++block.epoch;
                rejects([&] { slot.publish(2, memory(value)); });
            },
            {1, {}}
        );
        block.slots = {};
    }
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const std::string test = argv[1];
    if (test == "ranges") ranges();
    else if (test == "shared") shared();
    else if (test == "context") context_errors();
    else if (test == "overwrite") overwrite_wait();
    else if (test == "block") block_lifetime();
    else return 1;
}
"""


@pytest.fixture(scope="module")
def compile_flags():
    root = Path(__file__).resolve().parents[3]
    return [
        KernelCompiler("wsesim").gxx15.cxx_path,
        "-std=c++23",
        "-O2",
        "-pthread",
        "-ffp-contract=off",
        "-D__CPU_SIM",
        "-DPTO_CPU_SIM_TARGET_LINX",
        "-I" + str(Path(ensure_pto_isa_root()) / "include"),
        "-I" + str(root / "src/wse/platform/include"),
    ]


@pytest.fixture(scope="module")
def operation_binary(tmp_path_factory, compile_flags):
    directory = tmp_path_factory.mktemp("wse_operations")
    source = directory / "operations.cpp"
    source.write_text(_SOURCE)
    binary = directory / "operations"
    subprocess.run([*compile_flags, str(source), "-o", str(binary)], check=True, timeout=90)
    return binary


@pytest.mark.parametrize("case", ["ranges", "shared", "context", "overwrite", "block"])
def test_operation_contracts(operation_binary, case):
    subprocess.run([str(operation_binary), case], check=True, timeout=15)


@pytest.mark.parametrize(
    ("fragment", "diagnostic"),
    [
        ("static_assert(sizeof(wse::SharedTileSlot<64, Weight>) > 0);", "Linx Shared slot must be S0..S63"),
        (
            "void small(wse::SharedTileSlot<0, Weight> &slot, float *one) {"
            "slot.publish(1, wse::tile_memory<1, 1>(one, 1)); }",
            "Linx Shared tile exceeds the source matrix dimensions",
        ),
    ],
)
def test_invalid_shared_operands_have_explicit_diagnostics(tmp_path, compile_flags, fragment, diagnostic):
    source = tmp_path / "unsupported.cpp"
    source.write_text("#include <wse/tile_ops.h>\nusing Weight = pto::SharedMatrixRight<float, 32, 32>;\n" + fragment)
    result = subprocess.run(
        [*compile_flags, "-fsyntax-only", str(source)], check=False, capture_output=True, text=True, timeout=30
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr
