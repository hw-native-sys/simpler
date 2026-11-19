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
    using Arg0 = typename PyPTODeviceArg<0, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg0) == 24, "device argument size mismatch");
    auto* t0 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[0]);
    Arg0 arg0{};
    arg0.ptr = reinterpret_cast<decltype(arg0.ptr)>(t0->buffer.addr) + t0->start_offset;
    arg0.shape[0] = t0->shapes[0];
    arg0.stride[0] = t0->strides[0];
    using Arg1 = typename PyPTODeviceArg<1, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg1) == 24, "device argument size mismatch");
    auto* t1 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[1]);
    Arg1 arg1{};
    arg1.ptr = reinterpret_cast<decltype(arg1.ptr)>(t1->buffer.addr) + t1->start_offset;
    arg1.shape[0] = t1->shapes[0];
    arg1.stride[0] = t1->strides[0];
    using Arg2 = typename PyPTODeviceArg<2, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg2) == 24, "device argument size mismatch");
    auto* t2 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[2]);
    Arg2 arg2{};
    arg2.ptr = reinterpret_cast<decltype(arg2.ptr)>(t2->buffer.addr) + t2->start_offset;
    arg2.shape[0] = t2->shapes[0];
    arg2.stride[0] = t2->strides[0];
    using Arg3 = typename PyPTODeviceArg<3, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg3) == 40, "device argument size mismatch");
    auto* t3 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[3]);
    Arg3 arg3{};
    arg3.ptr = reinterpret_cast<decltype(arg3.ptr)>(t3->buffer.addr) + t3->start_offset;
    arg3.shape[0] = t3->shapes[0];
    arg3.stride[0] = t3->strides[0];
    arg3.stride[1] = t3->strides[1];
    arg3.stride[2] = t3->strides[2];
    using Arg4 = typename PyPTODeviceArg<4, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg4) == 40, "device argument size mismatch");
    auto* t4 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[4]);
    Arg4 arg4{};
    arg4.ptr = reinterpret_cast<decltype(arg4.ptr)>(t4->buffer.addr) + t4->start_offset;
    arg4.shape[0] = t4->shapes[0];
    arg4.stride[0] = t4->strides[0];
    arg4.stride[1] = t4->strides[1];
    arg4.stride[2] = t4->strides[2];
    using Arg5 = typename PyPTODeviceArg<5, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg5) == 32, "device argument size mismatch");
    auto* t5 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[5]);
    Arg5 arg5{};
    arg5.ptr = reinterpret_cast<decltype(arg5.ptr)>(t5->buffer.addr) + t5->start_offset;
    arg5.shape[0] = t5->shapes[0];
    arg5.shape[1] = t5->shapes[1];
    arg5.stride[0] = t5->strides[0];
    using Arg6 = typename PyPTODeviceArg<6, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg6) == 16, "device argument size mismatch");
    auto* t6 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[6]);
    Arg6 arg6{};
    arg6.ptr = reinterpret_cast<decltype(arg6.ptr)>(t6->buffer.addr) + t6->start_offset;
    arg6.shape[0] = t6->shapes[0];
    using Arg7 = typename PyPTODeviceArg<7, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg7) == 16, "device argument size mismatch");
    auto* t7 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[7]);
    Arg7 arg7{};
    arg7.ptr = reinterpret_cast<decltype(arg7.ptr)>(t7->buffer.addr) + t7->start_offset;
    arg7.shape[0] = t7->shapes[0];
    using Arg8 = typename PyPTODeviceArg<8, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg8) == 24, "device argument size mismatch");
    auto* t8 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[8]);
    Arg8 arg8{};
    arg8.ptr = reinterpret_cast<decltype(arg8.ptr)>(t8->buffer.addr) + t8->start_offset;
    arg8.shape[0] = t8->shapes[0];
    arg8.stride[0] = t8->strides[0];
    using Arg9 = typename PyPTODeviceArg<9, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg9) == 24, "device argument size mismatch");
    auto* t9 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[9]);
    Arg9 arg9{};
    arg9.ptr = reinterpret_cast<decltype(arg9.ptr)>(t9->buffer.addr) + t9->start_offset;
    arg9.shape[0] = t9->shapes[1];
    arg9.stride[0] = t9->strides[0];
    using Arg10 = typename PyPTODeviceArg<10, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg10) == 32, "device argument size mismatch");
    auto* t10 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[10]);
    Arg10 arg10{};
    arg10.ptr = reinterpret_cast<decltype(arg10.ptr)>(t10->buffer.addr) + t10->start_offset;
    arg10.shape[0] = t10->shapes[0];
    arg10.shape[1] = t10->shapes[1];
    arg10.stride[0] = t10->strides[0];
    using Arg11 = typename PyPTODeviceArg<11, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg11) == 32, "device argument size mismatch");
    auto* t11 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[11]);
    Arg11 arg11{};
    arg11.ptr = reinterpret_cast<decltype(arg11.ptr)>(t11->buffer.addr) + t11->start_offset;
    arg11.shape[0] = t11->shapes[0];
    arg11.shape[1] = t11->shapes[1];
    arg11.stride[0] = t11->strides[0];
    using Arg12 = typename PyPTODeviceArg<12, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg12) == 24, "device argument size mismatch");
    auto* t12 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[12]);
    Arg12 arg12{};
    arg12.ptr = reinterpret_cast<decltype(arg12.ptr)>(t12->buffer.addr) + t12->start_offset;
    arg12.shape[0] = t12->shapes[1];
    arg12.stride[0] = t12->strides[0];
    using Arg13 = typename PyPTODeviceArg<13, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg13) == 24, "device argument size mismatch");
    auto* t13 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[13]);
    Arg13 arg13{};
    arg13.ptr = reinterpret_cast<decltype(arg13.ptr)>(t13->buffer.addr) + t13->start_offset;
    arg13.shape[0] = t13->shapes[1];
    arg13.stride[0] = t13->strides[0];
    using Arg14 = typename PyPTODeviceArg<14, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg14) == 32, "device argument size mismatch");
    auto* t14 = reinterpret_cast<__gm__ simpler::hbg::Tensor*>(args[14]);
    Arg14 arg14{};
    arg14.ptr = reinterpret_cast<decltype(arg14.ptr)>(t14->buffer.addr) + t14->start_offset;
    arg14.shape[0] = t14->shapes[0];
    arg14.shape[1] = t14->shapes[1];
    arg14.stride[0] = t14->strides[0];
    using Arg16 = typename PyPTODeviceArg<15, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg16) == 8, "device argument size mismatch");
    Arg16 arg16 = *reinterpret_cast<__gm__ Arg16*>(&args[16]);
    using Arg18 = typename PyPTODeviceArg<16, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg18) == 8, "device argument size mismatch");
    Arg18 arg18 = *reinterpret_cast<__gm__ Arg18*>(&args[18]);
    using Arg20 = typename PyPTODeviceArg<17, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg20) == 8, "device argument size mismatch");
    Arg20 arg20 = get_block_idx(args);
    using Arg21 = typename PyPTODeviceArg<18, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg21) == 8, "device argument size mismatch");
    Arg21 arg21 = get_block_num(args);
    using Arg22 = typename PyPTODeviceArg<19, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg22) == 8, "device argument size mismatch");
    Arg22 arg22 = 0;
#if defined(__DAV_VEC__)
    arg22 = get_sub_block_id(args);
#endif
    using Arg15 = typename PyPTODeviceArg<20, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg15) == 4, "device argument size mismatch");
    Arg15 arg15 = *reinterpret_cast<__gm__ Arg15*>(&args[15]);
    using Arg17 = typename PyPTODeviceArg<21, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg17) == 4, "device argument size mismatch");
    Arg17 arg17 = *reinterpret_cast<__gm__ Arg17*>(&args[17]);
    using Arg19 = typename PyPTODeviceArg<22, decltype(kernel_score_topk_mix)>::type;
    static_assert(sizeof(Arg19) == 4, "device argument size mismatch");
    Arg19 arg19 = *reinterpret_cast<__gm__ Arg19*>(&args[19]);
    uint64_t saved_ctrl = asc_get_ctrl();
    set_atomic_none();
    kernel_score_topk_mix(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg16, arg18, arg20, arg21, arg22, arg15, arg17, arg19);
    pipe_barrier(PIPE_ALL);
    asc_dcci_entire_out();
    dsb(DSB_DDR);
    set_atomic_none();
    asc_set_ctrl(saved_ctrl);
}
