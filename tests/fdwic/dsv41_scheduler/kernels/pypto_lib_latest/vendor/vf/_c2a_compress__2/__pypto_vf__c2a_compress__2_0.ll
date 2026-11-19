; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <256 x i1> @llvm.hivm.pset.b16(i32)

declare { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float>, <256 x i1>, i32, i32, i32)

declare <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat>, <256 x i1>, i32)

declare ptr addrspace(6) @llvm.hivm.vstsx1.post.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare <64 x i32> @llvm.hivm.vci.v64i32(i32, i32)

declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32)

declare void @llvm.hivm.mem.bar.vst.vld()

declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32)

declare <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float>, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vsel.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float>, <64 x float>, <64 x float>, <256 x i1>)

declare { <128 x bfloat>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v128bf16(ptr addrspace(6), i32, i32, i32)

declare ptr addrspace(6) @llvm.hivm.vstsx1.post.v128bf16(<128 x bfloat>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

define void @__pypto_vf__c2a_compress__2_0(ptr addrspace(6) %0) #0 {
  %2 = getelementptr float, ptr addrspace(6) %0, i64 1792
  %3 = getelementptr float, ptr addrspace(6) %0, i64 768
  %4 = getelementptr float, ptr addrspace(6) %0, i64 1280
  %5 = getelementptr float, ptr addrspace(6) %0, i64 256
  %6 = getelementptr float, ptr addrspace(6) %0, i64 3328
  %7 = getelementptr bfloat, ptr addrspace(6) %0, i64 7680
  %8 = getelementptr bfloat, ptr addrspace(6) %0, i64 8192
  br label %9

9:                                                ; preds = %aivscope.latch, %1
  %10 = phi i64 [ %135, %aivscope.latch ], [ 0, %1 ]
  %11 = icmp slt i64 %10, 1
  br i1 %11, label %12, label %136

12:                                               ; preds = %9
  %13 = getelementptr float, ptr addrspace(6) %2, i64 0
  %14 = getelementptr float, ptr addrspace(6) %3, i64 0
  %15 = getelementptr float, ptr addrspace(6) %4, i64 0
  %16 = getelementptr float, ptr addrspace(6) %5, i64 0
  %17 = getelementptr float, ptr addrspace(6) %6, i64 0
  %18 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %19 = call <256 x i1> @llvm.hivm.pset.b16(i32 0)
  br label %20

20:                                               ; preds = %28, %12
  %21 = phi i16 [ %55, %28 ], [ 0, %12 ]
  %22 = phi ptr addrspace(6) [ %31, %28 ], [ %13, %12 ]
  %23 = phi ptr addrspace(6) [ %34, %28 ], [ %14, %12 ]
  %24 = phi ptr addrspace(6) [ %37, %28 ], [ %15, %12 ]
  %25 = phi ptr addrspace(6) [ %40, %28 ], [ %16, %12 ]
  %26 = phi ptr addrspace(6) [ %54, %28 ], [ %17, %12 ]
  %27 = icmp slt i16 %21, 8
  br i1 %27, label %28, label %56

28:                                               ; preds = %20
  %29 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %22, i32 256, i32 0, i32 1)
  %30 = extractvalue { <64 x float>, ptr addrspace(6) } %29, 0
  %31 = extractvalue { <64 x float>, ptr addrspace(6) } %29, 1
  %32 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %23, i32 256, i32 0, i32 1)
  %33 = extractvalue { <64 x float>, ptr addrspace(6) } %32, 0
  %34 = extractvalue { <64 x float>, ptr addrspace(6) } %32, 1
  %35 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %24, i32 256, i32 0, i32 1)
  %36 = extractvalue { <64 x float>, ptr addrspace(6) } %35, 0
  %37 = extractvalue { <64 x float>, ptr addrspace(6) } %35, 1
  %38 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %25, i32 256, i32 0, i32 1)
  %39 = extractvalue { <64 x float>, ptr addrspace(6) } %38, 0
  %40 = extractvalue { <64 x float>, ptr addrspace(6) } %38, 1
  %41 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %30, <64 x float> %33, <256 x i1> %18)
  %42 = call <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float> %30, <64 x float> %41, <256 x i1> %18)
  %43 = call <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float> %33, <64 x float> %41, <256 x i1> %18)
  %44 = call <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float> %42, <256 x i1> %18)
  %45 = call <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float> %43, <256 x i1> %18)
  %46 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %44, <64 x float> %45, <256 x i1> %18)
  %47 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %44, <64 x float> %46, <256 x i1> %18)
  %48 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %45, <64 x float> %46, <256 x i1> %18)
  %49 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %36, <64 x float> %47, <256 x i1> %18)
  %50 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %39, <64 x float> %48, <256 x i1> %18)
  %51 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %49, <64 x float> %50, <256 x i1> %18)
  %52 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %51, <256 x i1> %18, i32 0, i32 0, i32 0)
  %53 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %52, <256 x i1> %19, i32 0)
  %54 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v64f32(<64 x float> %53, ptr addrspace(6) %26, i32 256, i32 2, i32 1, <256 x i1> %18)
  %55 = add i16 %21, 1
  br label %20

56:                                               ; preds = %20
  %57 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %58 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %18, i32 1)
  call void @llvm.hivm.mem.bar.vst.vld()
  %59 = getelementptr float, ptr addrspace(6) %6, i64 256
  br label %60

60:                                               ; preds = %66, %56
  %61 = phi i16 [ %83, %66 ], [ 0, %56 ]
  %62 = phi <64 x float> [ %82, %66 ], [ %58, %56 ]
  %63 = phi ptr addrspace(6) [ %69, %66 ], [ %17, %56 ]
  %64 = phi ptr addrspace(6) [ %72, %66 ], [ %59, %56 ]
  %65 = icmp slt i16 %61, 4
  br i1 %65, label %66, label %84

66:                                               ; preds = %60
  %67 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %63, i32 256, i32 0, i32 1)
  %68 = extractvalue { <64 x float>, ptr addrspace(6) } %67, 0
  %69 = extractvalue { <64 x float>, ptr addrspace(6) } %67, 1
  %70 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %64, i32 256, i32 0, i32 1)
  %71 = extractvalue { <64 x float>, ptr addrspace(6) } %70, 0
  %72 = extractvalue { <64 x float>, ptr addrspace(6) } %70, 1
  %73 = sext i16 %61 to i64
  %74 = trunc i64 %73 to i32
  %75 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %74, <256 x i1> %18, i32 1)
  %76 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %68, <64 x float> %68, <256 x i1> %18)
  %77 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %71, <64 x float> %71, <256 x i1> %18)
  %78 = call <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32> %57, <64 x i32> %75, <256 x i1> %18)
  %79 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %76, <64 x float> %77, <256 x i1> %18)
  %80 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %79, <256 x i1> %18)
  %81 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %80, <256 x i1> %18, i32 1)
  %82 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %81, <64 x float> %62, <256 x i1> %78)
  %83 = add i16 %61, 1
  br label %60

84:                                               ; preds = %60
  %85 = call <256 x i1> @llvm.hivm.pset.b32(i32 4)
  %86 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.000000e+00, <256 x i1> %18, i32 1)
  %87 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.500000e+00, <256 x i1> %18, i32 1)
  %88 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 5.000000e-01, <256 x i1> %18, i32 1)
  %89 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 2139095040, <256 x i1> %18, i32 1)
  %90 = getelementptr bfloat, ptr addrspace(6) %0, i64 0
  %91 = getelementptr bfloat, ptr addrspace(6) %7, i64 0
  %92 = getelementptr bfloat, ptr addrspace(6) %8, i64 0
  %93 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %62, <256 x i1> %85)
  %94 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %93, <256 x i1> %18, i32 1)
  %95 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %94, float 0x3F60000000000000, <256 x i1> %18)
  %96 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %95, float 0x3EB0C6F7A0000000, <256 x i1> %18)
  %97 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %86, <64 x float> %96, <256 x i1> %18)
  %98 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %96, float -5.000000e-01, <256 x i1> %18)
  %99 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %96, float -1.000000e+00, <256 x i1> %18)
  %100 = bitcast <64 x float> %96 to <64 x i32>
  %101 = call <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32> %100, <64 x i32> %89, <256 x i1> %18)
  %102 = call <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float> %97, <256 x i1> %18)
  %103 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %86, <64 x float> %99, <64 x float> %97, <256 x i1> %18)
  %104 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %98, <64 x float> %102, <256 x i1> %18)
  %105 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %87, <64 x float> %104, <64 x float> %102, <256 x i1> %18)
  %106 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %102, <64 x float> %105, <256 x i1> %18)
  %107 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %106, float -1.000000e+00, <256 x i1> %18)
  %108 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %97, <64 x float> %107, <64 x float> %106, <256 x i1> %18)
  %109 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %103, <64 x float> %96, <64 x float> %108, <256 x i1> %18)
  %110 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %109, <64 x float> %106, <256 x i1> %18)
  %111 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %106, <64 x float> %110, <64 x float> %88, <256 x i1> %18)
  %112 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %58, <64 x float> %111, <256 x i1> %101)
  br label %113

113:                                              ; preds = %120, %84
  %114 = phi i16 [ %133, %120 ], [ 0, %84 ]
  %115 = phi ptr addrspace(6) [ %123, %120 ], [ %17, %84 ]
  %116 = phi ptr addrspace(6) [ %126, %120 ], [ %90, %84 ]
  %117 = phi ptr addrspace(6) [ %131, %120 ], [ %91, %84 ]
  %118 = phi ptr addrspace(6) [ %132, %120 ], [ %92, %84 ]
  %119 = icmp slt i16 %114, 8
  br i1 %119, label %120, label %134

120:                                              ; preds = %113
  %121 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %115, i32 256, i32 0, i32 1)
  %122 = extractvalue { <64 x float>, ptr addrspace(6) } %121, 0
  %123 = extractvalue { <64 x float>, ptr addrspace(6) } %121, 1
  %124 = call { <128 x bfloat>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v128bf16(ptr addrspace(6) %116, i32 128, i32 14, i32 1)
  %125 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %124, 0
  %126 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %124, 1
  %127 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %125, <256 x i1> %19, i32 0)
  %128 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %122, <64 x float> %112, <256 x i1> %18)
  %129 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %128, <64 x float> %127, <256 x i1> %18)
  %130 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %129, <256 x i1> %18, i32 0, i32 0, i32 0)
  %131 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v128bf16(<128 x bfloat> %130, ptr addrspace(6) %117, i32 128, i32 7, i32 1, <256 x i1> %18)
  %132 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v128bf16(<128 x bfloat> %130, ptr addrspace(6) %118, i32 128, i32 7, i32 1, <256 x i1> %18)
  %133 = add i16 %114, 1
  br label %113

134:                                              ; preds = %113
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %134
  %135 = add i64 %10, 1
  br label %9, !llvm.loop !1

136:                                              ; preds = %9
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
