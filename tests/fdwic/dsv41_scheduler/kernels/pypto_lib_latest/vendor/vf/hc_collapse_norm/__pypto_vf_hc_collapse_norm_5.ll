; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x i32> @llvm.hivm.vci.v64i32(i32, i32)

declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32)

declare { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6), i32, i32, i32)

declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float>, <256 x i1>)

declare <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float>, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vsel.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <256 x i1> @llvm.hivm.pset.b16(i32)

declare <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float>, <64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare { <128 x bfloat>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v128bf16(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat>, <256 x i1>, i32)

declare <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float>, <256 x i1>, i32, i32, i32)

declare ptr addrspace(6) @llvm.hivm.vstsx1.post.v128bf16(<128 x bfloat>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

define void @__pypto_vf_hc_collapse_norm_5(ptr addrspace(6) %0) #0 {
  %2 = getelementptr float, ptr addrspace(6) %0, i64 30752
  %3 = getelementptr bfloat, ptr addrspace(6) %0, i64 56320
  %4 = getelementptr bfloat, ptr addrspace(6) %0, i64 102464
  br label %5

5:                                                ; preds = %aivscope.latch, %1
  %6 = phi i64 [ %79, %aivscope.latch ], [ 0, %1 ]
  %7 = icmp slt i64 %6, 1
  br i1 %7, label %8, label %80

8:                                                ; preds = %5
  %9 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %10 = getelementptr float, ptr addrspace(6) %0, i64 0
  %11 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %12 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %9, i32 1)
  br label %13

13:                                               ; preds = %18, %8
  %14 = phi i16 [ %29, %18 ], [ 0, %8 ]
  %15 = phi <64 x float> [ %28, %18 ], [ %12, %8 ]
  %16 = phi ptr addrspace(6) [ %21, %18 ], [ %10, %8 ]
  %17 = icmp slt i16 %14, 64
  br i1 %17, label %18, label %30

18:                                               ; preds = %13
  %19 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %16, i32 256, i32 0, i32 1)
  %20 = extractvalue { <64 x float>, ptr addrspace(6) } %19, 0
  %21 = extractvalue { <64 x float>, ptr addrspace(6) } %19, 1
  %22 = sext i16 %14 to i64
  %23 = trunc i64 %22 to i32
  %24 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %23, <256 x i1> %9, i32 1)
  %25 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %20, <256 x i1> %9)
  %26 = call <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32> %11, <64 x i32> %24, <256 x i1> %9)
  %27 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %25, <256 x i1> %9, i32 1)
  %28 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %27, <64 x float> %15, <256 x i1> %26)
  %29 = add i16 %14, 1
  br label %13

30:                                               ; preds = %13
  %31 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %15, <256 x i1> %9)
  %32 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.000000e+00, <256 x i1> %9, i32 1)
  %33 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.500000e+00, <256 x i1> %9, i32 1)
  %34 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 5.000000e-01, <256 x i1> %9, i32 1)
  %35 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 2139095040, <256 x i1> %9, i32 1)
  %36 = getelementptr float, ptr addrspace(6) %2, i64 0
  %37 = getelementptr bfloat, ptr addrspace(6) %3, i64 0
  %38 = getelementptr bfloat, ptr addrspace(6) %4, i64 0
  %39 = call <256 x i1> @llvm.hivm.pset.b16(i32 0)
  %40 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %31, <256 x i1> %9, i32 1)
  %41 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %40, float 0x3F299999A0000000, <256 x i1> %9)
  %42 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %41, float 0x3EB0C6F7A0000000, <256 x i1> %9)
  %43 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %32, <64 x float> %42, <256 x i1> %9)
  %44 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %42, float -5.000000e-01, <256 x i1> %9)
  %45 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %42, float -1.000000e+00, <256 x i1> %9)
  %46 = bitcast <64 x float> %42 to <64 x i32>
  %47 = call <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32> %46, <64 x i32> %35, <256 x i1> %9)
  %48 = call <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float> %43, <256 x i1> %9)
  %49 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %32, <64 x float> %45, <64 x float> %43, <256 x i1> %9)
  %50 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %44, <64 x float> %48, <256 x i1> %9)
  %51 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %33, <64 x float> %50, <64 x float> %48, <256 x i1> %9)
  %52 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %48, <64 x float> %51, <256 x i1> %9)
  %53 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %52, float -1.000000e+00, <256 x i1> %9)
  %54 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %43, <64 x float> %53, <64 x float> %52, <256 x i1> %9)
  %55 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %49, <64 x float> %42, <64 x float> %54, <256 x i1> %9)
  %56 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %55, <64 x float> %52, <256 x i1> %9)
  %57 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %52, <64 x float> %56, <64 x float> %34, <256 x i1> %9)
  %58 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %12, <64 x float> %57, <256 x i1> %47)
  br label %59

59:                                               ; preds = %65, %30
  %60 = phi i16 [ %77, %65 ], [ 0, %30 ]
  %61 = phi ptr addrspace(6) [ %68, %65 ], [ %36, %30 ]
  %62 = phi ptr addrspace(6) [ %71, %65 ], [ %37, %30 ]
  %63 = phi ptr addrspace(6) [ %76, %65 ], [ %38, %30 ]
  %64 = icmp slt i16 %60, 80
  br i1 %64, label %65, label %78

65:                                               ; preds = %59
  %66 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %61, i32 256, i32 0, i32 1)
  %67 = extractvalue { <64 x float>, ptr addrspace(6) } %66, 0
  %68 = extractvalue { <64 x float>, ptr addrspace(6) } %66, 1
  %69 = call { <128 x bfloat>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v128bf16(ptr addrspace(6) %62, i32 128, i32 14, i32 1)
  %70 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %69, 0
  %71 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %69, 1
  %72 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %70, <256 x i1> %39, i32 0)
  %73 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %67, <64 x float> %58, <256 x i1> %9)
  %74 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %73, <64 x float> %72, <256 x i1> %9)
  %75 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %74, <256 x i1> %9, i32 0, i32 0, i32 0)
  %76 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v128bf16(<128 x bfloat> %75, ptr addrspace(6) %63, i32 128, i32 7, i32 1, <256 x i1> %9)
  %77 = add i16 %60, 1
  br label %59

78:                                               ; preds = %59
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %78
  %79 = add i64 %6, 1
  br label %5, !llvm.loop !1

80:                                               ; preds = %5
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
