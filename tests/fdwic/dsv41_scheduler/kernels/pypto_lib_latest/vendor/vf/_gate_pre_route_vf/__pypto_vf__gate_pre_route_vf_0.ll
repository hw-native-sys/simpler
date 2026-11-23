; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32)

declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare void @llvm.hivm.mem.bar.vst.vst()

define void @__pypto_vf__gate_pre_route_vf_0(ptr addrspace(6) %0, i64 %1) #0 {
  br label %3

3:                                                ; preds = %aivscope.latch, %2
  %4 = phi i64 [ %18, %aivscope.latch ], [ 0, %2 ]
  %5 = icmp slt i64 %4, 1
  br i1 %5, label %6, label %19

6:                                                ; preds = %3
  %7 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %8 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0xC7EFFFFFE0000000, <256 x i1> %7, i32 1)
  br label %9

9:                                                ; preds = %12, %6
  %10 = phi i64 [ %16, %12 ], [ 0, %6 ]
  %11 = icmp slt i64 %10, %1
  br i1 %11, label %12, label %17

12:                                               ; preds = %9
  %13 = mul i64 %10, 384
  %14 = mul i64 %13, 4
  %15 = getelementptr i8, ptr addrspace(6) %0, i64 %14
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %15, i32 0, i32 2, i32 0, <256 x i1> %7)
  call void @llvm.hivm.mem.bar.vst.vst()
  %16 = add i64 %10, 1
  br label %9

17:                                               ; preds = %9
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %17
  %18 = add i64 %4, 1
  br label %3, !llvm.loop !1

19:                                               ; preds = %3
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
