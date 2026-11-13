# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical, lifetime and rejection contracts for the temporary WSE backend."""

import subprocess
from pathlib import Path

import pytest

from simpler_setup.kernel_compiler import KernelCompiler

_SOURCE = r"""
#include <wse/tile_ops.h>
#include <wse/block_lifecycle.h>
#include <cassert>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

alignas(8) uint64_t block_epoch=1;
wse::BlockOperationStorage storage[2][2];
wse::BlockScope scope_for(int device, int core, uint64_t *epoch=&block_epoch, uint64_t expected=1) {
    return {epoch,expected,nullptr,0,&storage[device][core],wse::BlockOperationStorage::get};
}
const wse::BlockScope scope=scope_for(0,0);

void rejects(const std::function<void()> &call) {
    bool caught=false;
    try { call(); } catch (const std::exception &) { caught=true; }
    assert(caught);
}
void numerical() {
    float a[] = {1,2,3,77,4,5,6,88};
    float b[] = {1,2,3,4,5,6};
    float out[12]; for(auto &v:out) v=-111;
    wse::Tile<2,3,wse::Layout::COLUMN_MAJOR> left;
    wse::Tile<3,2> right;
    wse::Tile<2,2,wse::Layout::COLUMN_MAJOR> product;
    wse::TLOAD(left,{a,8,2,3,4,1});
    wse::TLOAD(right,{b,6,3,2,2,1});
    wse::TMATMUL(product,left,right);
    wse::TMATMUL_ACC(product,product,left,right);
    wse::TSTORE({out,12,2,2,6,2},product);
    assert(out[0]==44 && out[2]==56 && out[6]==98 && out[8]==128);
    for(int i:{1,3,4,5,7,9,10,11}) assert(out[i]==-111);
    wse::Tile<2,2> x;
    x.at(0,0)=-3; x.at(0,1)=4; x.at(1,0)=-0.0f; x.at(1,1)=std::numeric_limits<float>::quiet_NaN();
    wse::TRELU(x,x);
    assert(x.at(0,0)==0 && x.at(0,1)==4 && std::signbit(x.at(1,0)) && std::isnan(x.at(1,1)));
    wse::Tile<2,2> m; m.at(0,0)=1;m.at(0,1)=2;m.at(1,0)=3;m.at(1,1)=4;
    wse::TMATMUL(m,m,m);
    assert(m.at(0,0)==7 && m.at(0,1)==10 && m.at(1,0)==15 && m.at(1,1)==22);
}
void fractional() {
    wse::Tile<3,4,wse::Layout::COLUMN_MAJOR> a;
    wse::Tile<4,5> b;
    wse::Tile<3,5> result;
    for(size_t r=0;r<3;++r)for(size_t k=0;k<4;++k)a.at(r,k)=(float(r*4+k)-5.0f)/7.0f;
    for(size_t k=0;k<4;++k)for(size_t c=0;c<5;++c)b.at(k,c)=(float(k*5+c)-11.0f)/9.0f;
    wse::TMATMUL(result,a,b);
    wse::TMATMUL_ACC(result,result,a,b);
    for(size_t r=0;r<3;++r)for(size_t c=0;c<5;++c) {
        double expected=0;
        for(size_t k=0;k<4;++k)expected+=2.0*double(a.at(r,k))*double(b.at(k,c));
        assert(std::abs(double(result.at(r,c))-expected)<2e-5);
    }
}
void ranges() {
    float data[6]{}; wse::Tile<2,3> tile;
    rejects([&]{wse::TLOAD(tile,{data,5,2,3,3,1});});
    rejects([&]{wse::TSTORE({data,6,2,3,std::numeric_limits<size_t>::max(),1},tile);});
    rejects([&]{wse::TLOAD(tile,{nullptr,6,2,3,3,1});});
    rejects([&]{wse::TLOAD(tile,{data,6,1,3,3,1});});
    rejects([&]{wse::TLOAD(tile,{data,6,2,3,0,1});});
    for(float v:data) assert(v==0);
}
void shared() {
    std::vector<std::thread> threads;
    std::array<std::array<float,4>,8> results{};
    for(uint32_t core=0;core<2;++core) for(uint32_t pe=0;pe<4;++pe) threads.emplace_back([&,core,pe]{


        wse::SharedTile<wse::Tile<2,3>> tile(pe, 0, 0, 15, scope_for(0,core));
        for(uint64_t gen=1;gen<=4;++gen) {
            if(pe==0) {
                wse::Tile<2,3> value;
                for(size_t r=0;r<2;++r) for(size_t c=0;c<3;++c) value.at(r,c)=core*100+gen*10+r*3+c;
                tile.publish(gen,value);
            }
            auto value=tile.read(gen);
            results[core*4+pe][gen-1]=value.at(1,2);
        }
    });
    for(auto &thread:threads)thread.join();
    for(int core=0;core<2;++core)for(int pe=0;pe<4;++pe)for(int gen=1;gen<=4;++gen)
        assert(results[core*4+pe][gen-1]==core*100+gen*10+5);
    for(auto &core:storage[0]) core.clear();
}
void overwrite_wait() {
    using Shared=wse::SharedTile<wse::Tile<2,3>>;
    std::atomic<uint32_t> waiting{0};
    std::atomic<bool> finished{false};
    auto notify=[](void *owner,wse::OperationEvent event,uint32_t,uint64_t generation) {
        if(event==wse::OperationEvent::WAIT && generation==2) {
            auto &flag=*static_cast<std::atomic<uint32_t>*>(owner);
            flag.store(1,std::memory_order_release);flag.notify_one();
        }
    };
    std::thread producer([&] {


        Shared slot(0,0,0,6,scope,{&waiting,notify});
        wse::Tile<2,3> value;value.at(0,0)=11;slot.publish(1,value);
        value.at(0,0)=22;slot.publish(2,value);
        finished.store(true,std::memory_order_release);
    });
    waiting.wait(0,std::memory_order_acquire);
    Shared first(1,0,0,6,scope), last(2,0,0,6,scope);
    assert(first.read(1).at(0,0)==11);
    assert(!finished.load(std::memory_order_acquire));
    assert(last.read(1).at(0,0)==11);
    producer.join();
    assert(finished.load(std::memory_order_acquire));
    assert(first.read(2).at(0,0)==22 && last.read(2).at(0,0)==22);
}
void context_errors() {
    using Shared=wse::SharedTile<wse::Tile<2,3>>;
    rejects([&]{Shared s(0,0,0,1,{&block_epoch,1});});
    rejects([&]{Shared s(4,0,0,1,scope);});
    rejects([&]{Shared s(0,64,0,1,scope);});
    rejects([&]{Shared s(0,0,4,1,scope);});
    rejects([&]{Shared s(0,0,0,16,scope);});
    Shared s(0,0,0,1,scope);wse::Tile<2,3> t;
    rejects([&]{s.publish(2,t);});
    rejects([&]{s.read(2);});
    s.publish(1,t);
    auto value=s.read(1);rejects([&]{s.read(1);});
    rejects([&]{Shared mismatch(0,0,0,3,scope);});
    rejects([&]{wse::SharedTile<wse::Tile<3,2>> mismatch(0,0,0,1,scope);});
    Shared other(0,1,0,1,scope);other.publish(1,t);assert(other.read(1).at(0,0)==0);
}
void lifecycle() {
    using Shared=wse::SharedTile<wse::Tile<2,3>>;
    for(int repeat=0;repeat<2;++repeat) {
        for(int device=0;device<2;++device) {



            Shared slot(0,0,0,1,scope_for(device,0));
            wse::Tile<2,3> value;value.at(0,0)=device+repeat*10;
            slot.publish(1,value);
        }
        for(int device=0;device<2;++device) {

            Shared slot(0,0,0,1,scope_for(device,0));
            assert(slot.read(1).at(0,0)==device+repeat*10);
            storage[device][0].clear();
        }
    }


}
void block_lifetime() {
    uint64_t run_epoch=1, block_serial=1;
    wse::BlockScope recorded{&block_serial,1,&run_epoch,1};
    recorded.validate();
    run_epoch=(uint64_t{1}<<32)+1;
    rejects([&]{recorded.validate();});
    uint64_t epochs[2]{1,1};
    using Shared=wse::SharedTile<wse::Tile<2,3>>;
    Shared old(0,0,0,1,scope_for(0,0,&epochs[0],1));
    wse::Tile<2,3> value;value.at(0,0)=17;old.publish(1,value);

    Shared other(0,0,0,1,scope_for(0,1,&epochs[1],1));
    value.at(0,0)=23;other.publish(1,value);
    __atomic_store_n(&epochs[0],0,__ATOMIC_RELEASE);
    storage[0][0].clear();
    rejects([&]{old.read(1);});
    assert(other.read(1).at(0,0)==23);

    __atomic_store_n(&epochs[0],2,__ATOMIC_RELEASE);
    // Reusing the same slot may change its shape and starts at generation one.
    wse::SharedTile<wse::Tile<3,2>> next(0,0,0,1,scope_for(0,0,&epochs[0],2));
    wse::Tile<3,2> fresh;fresh.at(0,0)=41;next.publish(1,fresh);
    assert(next.read(1).at(0,0)==41);
    rejects([&]{old.publish(1,value);});
    rejects([&]{Shared invalid(0,1,0,1,{&epochs[0],1});});
}
int main(int argc,char **argv) {
    assert(argc==2);



    const std::string test=argv[1];
    if(test=="numerical")numerical();
    else if(test=="fractional")fractional();
    else if(test=="ranges")ranges();
    else if(test=="shared")shared();
    else if(test=="context")context_errors();
    else if(test=="overwrite")overwrite_wait();
    else if(test=="lifecycle")lifecycle();
    else if(test=="block")block_lifetime();
    else return 1;

}
"""


@pytest.fixture(scope="module")
def operation_binary(tmp_path_factory):
    root = Path(__file__).resolve().parents[3]
    directory = tmp_path_factory.mktemp("wse_operations")
    source = directory / "operations.cpp"
    source.write_text(_SOURCE)
    binary = directory / "operations"
    compiler = KernelCompiler("wsesim")
    subprocess.run(
        [
            compiler.gxx15.cxx_path,
            "-std=c++23",
            "-O2",
            "-pthread",
            "-I" + str(root / "src/wse/platform/include"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=90,
    )
    return binary


@pytest.mark.parametrize(
    "case", ["numerical", "fractional", "ranges", "shared", "context", "lifecycle", "overwrite", "block"]
)
def test_operation_contracts(operation_binary, case):
    subprocess.run([str(operation_binary), case], check=True, timeout=15)


def test_unsupported_tile_shape_has_explicit_diagnostic(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "unsupported.cpp"
    source.write_text("#include <wse/tile_ops.h>\nwse::Tile<33,32> unsupported;\n")
    compiler = KernelCompiler("wsesim")
    result = subprocess.run(
        [
            compiler.gxx15.cxx_path,
            "-std=c++23",
            "-fsyntax-only",
            "-I" + str(root / "src/wse/platform/include"),
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "temporary WSE tiles require dimensions in [1,32]" in result.stderr
