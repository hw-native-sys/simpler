; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32)

declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare void @llvm.hivm.mem.bar.vst.vst()

define void @__pypto_vf__gate_pre_route_vf_1(ptr addrspace(6) %0, i64 %1) #0 {
  br label %3

3:                                                ; preds = %aivscope.latch, %2
  %4 = phi i64 [ %19, %aivscope.latch ], [ 0, %2 ]
  %5 = icmp slt i64 %4, 1
  br i1 %5, label %6, label %20

6:                                                ; preds = %3
  %7 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %8 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0xC7EFFFFFE0000000, <256 x i1> %7, i32 1)
  br label %9

9:                                                ; preds = %12, %6
  %10 = phi i64 [ %17, %12 ], [ 0, %6 ]
  %11 = icmp slt i64 %10, %1
  br i1 %11, label %12, label %18

12:                                               ; preds = %9
  %13 = mul i64 %10, 384
  %14 = add i64 %13, 64
  %15 = mul i64 %14, 4
  %16 = getelementptr i8, ptr addrspace(6) %0, i64 %15
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %8, ptr addrspace(6) %16, i32 0, i32 2, i32 0, <256 x i1> %7)
  call void @llvm.hivm.mem.bar.vst.vst()
  %17 = add i64 %10, 1
  br label %9

18:                                               ; preds = %9
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %18
  %19 = add i64 %4, 1
  br label %3, !llvm.loop !1

20:                                               ; preds = %3
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
