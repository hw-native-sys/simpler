#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

constexpr int M = 16;
constexpr int K = 16;
constexpr int N = 16;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {}
