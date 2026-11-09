#include "c_api/asc_simd.h"
#include "host_build_graph/tensor.h"
#include "intrinsic.h"
#pragma push_macro("__global__")
#undef __global__
#define __global__ inline __attribute__((always_inline))
#pragma push_macro("__cube__")
#undef __cube__
#define __cube__ 
#pragma push_macro("__vector__")
#undef __vector__
#define __vector__ 
#pragma push_macro("KERNEL_TASK_TYPE_DEFAULT")
#undef KERNEL_TASK_TYPE_DEFAULT
#define KERNEL_TASK_TYPE_DEFAULT(...) ((void)0)
#include "device_source.hpp"
#pragma pop_macro("KERNEL_TASK_TYPE_DEFAULT")
#pragma pop_macro("__vector__")
#pragma pop_macro("__cube__")
#pragma pop_macro("__global__")
template<unsigned I, typename T, typename... Rest> struct PyPTOArgAt : PyPTOArgAt<I-1, Rest...> {};
template<typename T, typename... Rest> struct PyPTOArgAt<0, T, Rest...> { using type = T; };
template<unsigned I, typename Signature> struct PyPTODeviceArg;
template<unsigned I, typename R, typename... Args>
struct PyPTODeviceArg<I, R(Args...)> : PyPTOArgAt<I, Args...> {};
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    using Arg0 = typename PyPTODeviceArg<0, decltype(kernel_shared_swiglu_quant_grid)>::type;
    static_assert(sizeof(Arg0) == 16, "device argument size mismatch");
    auto* t0 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[0]);
    Arg0 arg0{};
    arg0.ptr = reinterpret_cast<decltype(arg0.ptr)>(t0->buffer.addr) + t0->start_offset;
    arg0.stride[0] = t0->strides[0];
    using Arg1 = typename PyPTODeviceArg<1, decltype(kernel_shared_swiglu_quant_grid)>::type;
    static_assert(sizeof(Arg1) == 16, "device argument size mismatch");
    auto* t1 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[1]);
    Arg1 arg1{};
    arg1.ptr = reinterpret_cast<decltype(arg1.ptr)>(t1->buffer.addr) + t1->start_offset;
    arg1.stride[0] = t1->strides[0];
    using Arg2 = typename PyPTODeviceArg<2, decltype(kernel_shared_swiglu_quant_grid)>::type;
    static_assert(sizeof(Arg2) == 16, "device argument size mismatch");
    auto* t2 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[2]);
    Arg2 arg2{};
    arg2.ptr = reinterpret_cast<decltype(arg2.ptr)>(t2->buffer.addr) + t2->start_offset;
    arg2.stride[0] = t2->strides[0];
    using Arg3 = typename PyPTODeviceArg<3, decltype(kernel_shared_swiglu_quant_grid)>::type;
    static_assert(sizeof(Arg3) == 16, "device argument size mismatch");
    auto* t3 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[3]);
    Arg3 arg3{};
    arg3.ptr = reinterpret_cast<decltype(arg3.ptr)>(t3->buffer.addr) + t3->start_offset;
    arg3.stride[0] = t3->strides[0];
    using Arg4 = typename PyPTODeviceArg<4, decltype(kernel_shared_swiglu_quant_grid)>::type;
    static_assert(sizeof(Arg4) == 8, "device argument size mismatch");
    Arg4 arg4 = get_block_idx(args);
    using Arg5 = typename PyPTODeviceArg<5, decltype(kernel_shared_swiglu_quant_grid)>::type;
    static_assert(sizeof(Arg5) == 8, "device argument size mismatch");
    Arg5 arg5 = get_block_num(args);
    uint64_t saved_ctrl = asc_get_ctrl();
    set_atomic_none();
    kernel_shared_swiglu_quant_grid(arg0, arg1, arg2, arg3, arg4, arg5);
    pipe_barrier(PIPE_ALL);
    asc_dcci_entire_out();
    dsb(DSB_DDR);
    set_atomic_none();
    asc_set_ctrl(saved_ctrl);
}
