; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x i32> @llvm.hivm.vci.v64i32(i32, i32)

declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32)

declare <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <256 x i1> @llvm.hivm.vcmp.lt.u.z.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6), <64 x i32>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsel.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare void @llvm.hivm.mem.bar.vld.vst()

declare { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vmin.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vln.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float>, <256 x i1>)

declare ptr addrspace(6) @llvm.hivm.vstsx1.post.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

define void @__pypto_vf__gate_scores_vf_aiv_0(ptr addrspace(6) %0, i64 %1) #0 {
  %3 = getelementptr float, ptr addrspace(6) %0, i64 256
  %4 = getelementptr float, ptr addrspace(6) %0, i64 272
  %5 = getelementptr float, ptr addrspace(6) %0, i64 528
  br label %6

6:                                                ; preds = %aivscope.latch, %2
  %7 = phi i64 [ %58, %aivscope.latch ], [ 0, %2 ]
  %8 = icmp slt i64 %7, 1
  br i1 %8, label %9, label %59

9:                                                ; preds = %6
  %10 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %11 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %12 = trunc i64 %1 to i32
  %13 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 15, <256 x i1> %10, i32 1)
  %14 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %12, <256 x i1> %10, i32 1)
  %15 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %10, i32 1)
  %16 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0xC7EFFFFFE0000000, <256 x i1> %10, i32 1)
  %17 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.000000e+00, <256 x i1> %10, i32 1)
  %18 = call <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32> %11, <64 x i32> %13, <256 x i1> %10)
  %19 = call <256 x i1> @llvm.hivm.vcmp.lt.u.z.v64i32(<64 x i32> %18, <64 x i32> %14, <256 x i1> %10)
  %20 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %3, <64 x i32> %18, <256 x i1> %19)
  %21 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %20, <64 x float> %15, <256 x i1> %19)
  %22 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %21, <64 x float> %16, <256 x i1> %19)
  call void @llvm.hivm.mem.bar.vld.vst()
  %23 = getelementptr float, ptr addrspace(6) %0, i64 0
  %24 = getelementptr float, ptr addrspace(6) %4, i64 0
  %25 = getelementptr float, ptr addrspace(6) %5, i64 0
  br label %26

26:                                               ; preds = %32, %9
  %27 = phi i16 [ %56, %32 ], [ 0, %9 ]
  %28 = phi ptr addrspace(6) [ %35, %32 ], [ %23, %9 ]
  %29 = phi ptr addrspace(6) [ %54, %32 ], [ %24, %9 ]
  %30 = phi ptr addrspace(6) [ %55, %32 ], [ %25, %9 ]
  %31 = icmp slt i16 %27, 4
  br i1 %31, label %32, label %57

32:                                               ; preds = %26
  %33 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %28, i32 256, i32 0, i32 1)
  %34 = extractvalue { <64 x float>, ptr addrspace(6) } %33, 0
  %35 = extractvalue { <64 x float>, ptr addrspace(6) } %33, 1
  %36 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %34, <64 x float> %15, <256 x i1> %19)
  %37 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %36, <256 x i1> %10)
  %38 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %36, float -1.000000e+00, <256 x i1> %10)
  %39 = call <64 x float> @llvm.hivm.vmin.s.x.v64f32(<64 x float> %36, <64 x float> %15, <256 x i1> %10)
  %40 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %36, <64 x float> %15, <256 x i1> %10)
  %41 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %37, float -1.000000e+00, <256 x i1> %10)
  %42 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %38, float -1.000000e+01, <256 x i1> %10)
  %43 = call <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float> %39, <256 x i1> %10)
  %44 = call <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float> %41, <256 x i1> %10)
  %45 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %42, <64 x float> %15, <256 x i1> %10)
  %46 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %44, float 1.000000e+00, <256 x i1> %10)
  %47 = call <64 x float> @llvm.hivm.vmin.s.x.v64f32(<64 x float> %45, <64 x float> %17, <256 x i1> %10)
  %48 = call <64 x float> @llvm.hivm.vln.x.v64f32(<64 x float> %46, <256 x i1> %10)
  %49 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %47, <64 x float> %43, <256 x i1> %10)
  %50 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %40, <64 x float> %48, <256 x i1> %10)
  %51 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %50, <64 x float> %49, <256 x i1> %10)
  %52 = call <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float> %51, <256 x i1> %10)
  %53 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %52, <64 x float> %22, <256 x i1> %10)
  %54 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v64f32(<64 x float> %52, ptr addrspace(6) %29, i32 256, i32 2, i32 1, <256 x i1> %10)
  %55 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v64f32(<64 x float> %53, ptr addrspace(6) %30, i32 256, i32 2, i32 1, <256 x i1> %10)
  %56 = add i16 %27, 1
  br label %26

57:                                               ; preds = %26
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %57
  %58 = add i64 %7, 1
  br label %6, !llvm.loop !1

59:                                               ; preds = %6
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
