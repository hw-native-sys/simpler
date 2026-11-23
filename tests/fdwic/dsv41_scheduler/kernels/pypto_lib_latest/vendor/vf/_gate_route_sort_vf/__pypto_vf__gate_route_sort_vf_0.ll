; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x i32> @llvm.hivm.vci.v64i32(i32, i32)

declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32)

declare <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vmuls.u.x.v64i32(<64 x i32>, i32, <256 x i1>)

declare <256 x i1> @llvm.hivm.vcmp.lt.u.z.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vadd.u.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <256 x i1> @llvm.hivm.pand.z(<256 x i1>, <256 x i1>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32>, i32, <256 x i1>)

declare <64 x i32> @llvm.hivm.vgather2.bc.v64s32(ptr addrspace(6), <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare void @llvm.hivm.mem.bar.vld.vst()

declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32)

declare void @llvm.hivm.vstsx1.v64i32(<64 x i32>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare void @llvm.hivm.mem.bar.vst.vld()

declare <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6), <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vxor.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsel.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32>, <64 x i32>)

declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

define void @__pypto_vf__gate_route_sort_vf_0(ptr addrspace(6) %0, float %1) #0 {
  %3 = getelementptr i32, ptr addrspace(6) %0, i64 4352
  %4 = getelementptr float, ptr addrspace(6) %0, i64 2048
  %5 = getelementptr float, ptr addrspace(6) %0, i64 3328
  br label %6

6:                                                ; preds = %aivscope.latch, %2
  %7 = phi i64 [ %52, %aivscope.latch ], [ 0, %2 ]
  %8 = icmp slt i64 %7, 1
  br i1 %8, label %9, label %53

9:                                                ; preds = %6
  %10 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %11 = call <256 x i1> @llvm.hivm.pset.b32(i32 7)
  %12 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %13 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 7, <256 x i1> %10, i32 1)
  %14 = call <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32> %12, i16 3, <256 x i1> %11)
  %15 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 6, <256 x i1> %10, i32 1)
  %16 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 0, <256 x i1> %10, i32 1)
  %17 = call <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32> %12, <64 x i32> %13, <256 x i1> %11)
  %18 = call <64 x i32> @llvm.hivm.vmuls.u.x.v64i32(<64 x i32> %14, i32 16, <256 x i1> %11)
  %19 = call <64 x i32> @llvm.hivm.vmuls.u.x.v64i32(<64 x i32> %17, i32 2, <256 x i1> %11)
  %20 = call <256 x i1> @llvm.hivm.vcmp.lt.u.z.v64i32(<64 x i32> %17, <64 x i32> %15, <256 x i1> %10)
  %21 = call <64 x i32> @llvm.hivm.vadd.u.x.v64i32(<64 x i32> %18, <64 x i32> %19, <256 x i1> %11)
  %22 = call <256 x i1> @llvm.hivm.pand.z(<256 x i1> %20, <256 x i1> %11, <256 x i1> %10)
  %23 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %21, i32 1, <256 x i1> %11)
  %24 = call <64 x i32> @llvm.hivm.vgather2.bc.v64s32(ptr addrspace(6) %3, <64 x i32> %23, <256 x i1> %11)
  %25 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %24, <64 x i32> %16, <256 x i1> %11)
  call void @llvm.hivm.mem.bar.vld.vst()
  %26 = call <64 x i32> @llvm.hivm.vmuls.u.x.v64i32(<64 x i32> %14, i32 256, <256 x i1> %11)
  %27 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %10, i32 1)
  %28 = call <64 x i32> @llvm.hivm.vadd.u.x.v64i32(<64 x i32> %26, <64 x i32> %25, <256 x i1> %11)
  call void @llvm.hivm.vstsx1.v64i32(<64 x i32> %25, ptr addrspace(6) %0, i32 0, i32 2, i32 0, <256 x i1> %11)
  call void @llvm.hivm.mem.bar.vst.vld()
  %29 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 4, <256 x i1> %10, i32 1)
  %30 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 2, <256 x i1> %10, i32 1)
  %31 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 1, <256 x i1> %10, i32 1)
  %32 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %4, <64 x i32> %28, <256 x i1> %22)
  %33 = call <64 x i32> @llvm.hivm.vxor.x.v64i32(<64 x i32> %12, <64 x i32> %29, <256 x i1> %11)
  %34 = call <64 x i32> @llvm.hivm.vxor.x.v64i32(<64 x i32> %12, <64 x i32> %30, <256 x i1> %11)
  %35 = call <64 x i32> @llvm.hivm.vxor.x.v64i32(<64 x i32> %12, <64 x i32> %31, <256 x i1> %11)
  %36 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %32, <64 x float> %27, <256 x i1> %22)
  %37 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %36, <64 x float> %27, <256 x i1> %22)
  %38 = bitcast <64 x float> %37 to <64 x i32>
  %39 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %38, <64 x i32> %33)
  %40 = bitcast <64 x i32> %39 to <64 x float>
  %41 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %37, <64 x float> %40, <256 x i1> %11)
  %42 = bitcast <64 x float> %41 to <64 x i32>
  %43 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %42, <64 x i32> %34)
  %44 = bitcast <64 x i32> %43 to <64 x float>
  %45 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %41, <64 x float> %44, <256 x i1> %11)
  %46 = bitcast <64 x float> %45 to <64 x i32>
  %47 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %46, <64 x i32> %35)
  %48 = bitcast <64 x i32> %47 to <64 x float>
  %49 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %45, <64 x float> %48, <256 x i1> %11)
  %50 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %37, <64 x float> %49, <256 x i1> %11)
  %51 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %50, float %1, <256 x i1> %11)
  call void @llvm.hivm.mem.bar.vld.vst()
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %51, ptr addrspace(6) %5, i32 0, i32 2, i32 0, <256 x i1> %11)
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %9
  %52 = add i64 %7, 1
  br label %6, !llvm.loop !1

53:                                               ; preds = %6
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
