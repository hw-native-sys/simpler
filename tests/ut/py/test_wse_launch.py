# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Identity, receipt and partial-start contracts of the WSE execution transaction."""

import subprocess

from simpler_setup.kernel_compiler import KernelCompiler


def test_single_executor_launch_ownership(tmp_path):
    compiler = KernelCompiler("wsesim")
    root = compiler.project_root
    source = tmp_path / "launch.cpp"
    source.write_text(r"""
#include "wse_launch_transaction.h"
#include <cassert>
#include <stdexcept>
#include <type_traits>
// This isolated contract test supplies the permit owner's minting operation.
class WseHostRunner {
public:
    static WseLaunchPermit mint(NativeRunIdentity identity) { return WseLaunchPermit(identity); }
};
int main() {
    static_assert(!std::is_copy_constructible_v<WseLaunchPermit>);
    static_assert(!std::is_copy_constructible_v<WseLaunchReceipt>);
    NativeRunIdentity identity{17,23,29,1}, other=identity;
    ++other.run_epoch;
    int calls=0;
    auto submit=[&](LaunchProgressSink &) { ++calls; return 0; };
    auto mismatch=wse_launch_transaction(other,WseHostRunner::mint(identity),submit);
    assert(calls==0 && !mismatch.receipt.valid());
    auto permit=WseHostRunner::mint(identity);
    auto accepted=wse_launch_transaction(identity,std::move(permit),submit);
    assert(calls==1 && accepted.rc==0 && accepted.progress==LaunchProgress::Complete);
    assert(accepted.receipt.matches(identity) && !accepted.receipt.matches(other));
    auto reused=wse_launch_transaction(identity,std::move(permit),submit);
    assert(calls==1 && reused.rc!=0 && !reused.receipt.valid());
    auto refused=wse_launch_transaction(identity,WseHostRunner::mint(identity),
        [](LaunchProgressSink &) { return -7; });
    assert(refused.rc==-7 && refused.progress==LaunchProgress::NotStarted && !refused.receipt.valid());
    auto partial=wse_launch_transaction(identity,WseHostRunner::mint(identity),
        [](LaunchProgressSink &sink) { sink.mark_submitted(); return -8; });
    assert(partial.rc==-8 && partial.progress==LaunchProgress::Partial && !partial.receipt.valid());
    auto uncertain=wse_launch_transaction(identity,WseHostRunner::mint(identity),
        [](LaunchProgressSink &) -> int { throw std::bad_alloc(); });
    assert(uncertain.rc!=0 && uncertain.progress==LaunchProgress::Partial && !uncertain.receipt.valid());
}
""")
    includes = compiler.get_orchestration_include_dirs("host_build_graph") + [
        str(root / "src/common/worker"),
        str(root / "src/wse/platform/sim/host"),
    ]
    binary = tmp_path / "launch"
    subprocess.run(
        [
            compiler.gxx15.cxx_path,
            "-std=c++17",
            "-O2",
            str(source),
            *["-I" + path for path in includes],
            "-o",
            str(binary),
        ],
        check=True,
        timeout=90,
    )
    subprocess.run([str(binary)], check=True, timeout=20)
