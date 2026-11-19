; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x i32> @llvm.hivm.vci.v64i32(i32, i32)

declare <256 x i1> @llvm.hivm.pset.b16(i32)

declare <256 x i1> @llvm.hivm.pset.b8(i32)

declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32)

declare <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vshls.u.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32>, i32, <256 x i1>)

declare { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vldsx2.v128bf16(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat>, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vcgmax.s.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x i32> @llvm.hivm.vadds.s.x.v64i32(<64 x i32>, i32, <256 x i1>)

declare <64 x i32> @llvm.hivm.vmaxs.s.x.v64i32(<64 x i32>, i32, <256 x i1>)

declare <64 x i32> @llvm.hivm.vmuls.s.x.v64i32(<64 x i32>, i32, <256 x i1>)

declare <64 x i32> @llvm.hivm.vmins.s.x.v64i32(<64 x i32>, i32, <256 x i1>)

declare <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32>, <64 x i32>)

declare <64 x i32> @llvm.hivm.vmul.s.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32>, <64 x i32>)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vmins.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <256 x float8e4m3> @llvm.hivm.vcvtff.f322f8e4m3.x(<64 x float>, <256 x i1>, i32, i32, i32)

declare <256 x float8e4m3> @llvm.hivm.vor.x.v256fp8e4m3(<256 x float8e4m3>, <256 x float8e4m3>, <256 x i1>)

declare void @llvm.hivm.vstsx1.v64i32(<64 x i32>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare void @llvm.hivm.mem.bar.vv.all()

declare void @llvm.hivm.mem.bar.vst.vst()

define void @__pypto_vf_qkv_quant_0(ptr addrspace(6) %0, i64 %1, i64 %2) #0 {
  %4 = getelementptr i32, ptr addrspace(6) %0, i64 4096
  %5 = getelementptr i32, ptr addrspace(6) %0, i64 6144
  br label %6

6:                                                ; preds = %aivscope.latch, %3
  %7 = phi i64 [ %117, %aivscope.latch ], [ 0, %3 ]
  %8 = icmp slt i64 %7, 1
  br i1 %8, label %9, label %118

9:                                                ; preds = %6
  %10 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %11 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %12 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %13 = mul i64 %1, 16
  %14 = mul i64 %1, 64
  %15 = call <256 x i1> @llvm.hivm.pset.b32(i32 5)
  %16 = call <256 x i1> @llvm.hivm.pset.b16(i32 0)
  %17 = call <256 x i1> @llvm.hivm.pset.b8(i32 0)
  %18 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 3, <256 x i1> %10, i32 1)
  %19 = call <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32> %11, i16 3, <256 x i1> %10)
  %20 = call <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32> %12, i16 3, <256 x i1> %10)
  %21 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 7, <256 x i1> %10, i32 1)
  %22 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 0, <256 x i1> %10, i32 1)
  %23 = call <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32> %19, <64 x i32> %18, <256 x i1> %10)
  %24 = call <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32> %11, <64 x i32> %21, <256 x i1> %10)
  %25 = call <64 x i32> @llvm.hivm.vshls.u.x.v64i32(<64 x i32> %23, i16 4, <256 x i1> %10)
  %26 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %25, i32 8, <256 x i1> %10)
  br label %27

27:                                               ; preds = %31, %9
  %28 = phi i64 [ %101, %31 ], [ 0, %9 ]
  %29 = phi <64 x i32> [ %100, %31 ], [ %22, %9 ]
  %30 = icmp slt i64 %28, %2
  br i1 %30, label %31, label %102

31:                                               ; preds = %27
  %32 = add i64 %13, %28
  %33 = mul i64 %32, 256
  %34 = mul i64 %33, 2
  %35 = getelementptr i8, ptr addrspace(6) %0, i64 %34
  %36 = call { <128 x bfloat>, <128 x bfloat> } @llvm.hivm.vldsx2.v128bf16(ptr addrspace(6) %35, i32 0, i32 12, i32 0)
  %37 = extractvalue { <128 x bfloat>, <128 x bfloat> } %36, 0
  %38 = extractvalue { <128 x bfloat>, <128 x bfloat> } %36, 1
  %39 = sdiv i64 %28, 2
  %40 = trunc i64 %39 to i32
  %41 = srem i64 %28, 2
  %42 = mul i64 %41, 16
  %43 = trunc i64 %42 to i16
  %44 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %40, <256 x i1> %10, i32 1)
  %45 = mul i64 %32, 64
  %46 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %37, <256 x i1> %16, i32 0)
  %47 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %37, <256 x i1> %16, i32 1)
  %48 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %38, <256 x i1> %16, i32 0)
  %49 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %38, <256 x i1> %16, i32 1)
  %50 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %46, <256 x i1> %10)
  %51 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %48, <256 x i1> %10)
  %52 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %47, <256 x i1> %10)
  %53 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %49, <256 x i1> %10)
  %54 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %50, <64 x float> %51, <256 x i1> %10)
  %55 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %54, <64 x float> %52, <256 x i1> %10)
  %56 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %55, <64 x float> %53, <256 x i1> %10)
  %57 = call <64 x float> @llvm.hivm.vcgmax.s.x.v64f32(<64 x float> %56, <256 x i1> %10)
  %58 = bitcast <64 x float> %57 to <64 x i32>
  %59 = call <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32> %58, i16 23, <256 x i1> %15)
  %60 = call <64 x i32> @llvm.hivm.vadds.s.x.v64i32(<64 x i32> %59, i32 -8, <256 x i1> %15)
  %61 = call <64 x i32> @llvm.hivm.vmaxs.s.x.v64i32(<64 x i32> %60, i32 0, <256 x i1> %15)
  %62 = call <64 x i32> @llvm.hivm.vmuls.s.x.v64i32(<64 x i32> %61, i32 -1, <256 x i1> %15)
  %63 = call <64 x i32> @llvm.hivm.vmins.s.x.v64i32(<64 x i32> %61, i32 1, <256 x i1> %15)
  %64 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %61, <64 x i32> %20)
  %65 = call <64 x i32> @llvm.hivm.vadds.s.x.v64i32(<64 x i32> %62, i32 254, <256 x i1> %15)
  %66 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %64, <64 x i32> %26)
  %67 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %64, <64 x i32> %25)
  %68 = call <64 x i32> @llvm.hivm.vmul.s.x.v64i32(<64 x i32> %65, <64 x i32> %63, <256 x i1> %15)
  %69 = call <64 x i32> @llvm.hivm.vshls.u.x.v64i32(<64 x i32> %66, i16 8, <256 x i1> %10)
  %70 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %68, i16 23, <256 x i1> %15)
  %71 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %67, <64 x i32> %69, <256 x i1> %10)
  %72 = bitcast <64 x i32> %70 to <64 x float>
  %73 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %70, <64 x i32> %20)
  %74 = bitcast <64 x i32> %73 to <64 x float>
  %75 = call <64 x i32> @llvm.hivm.vshls.u.x.v64i32(<64 x i32> %71, i16 %43, <256 x i1> %10)
  %76 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %46, <64 x float> %74, <256 x i1> %10)
  %77 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %48, <64 x float> %74, <256 x i1> %10)
  %78 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %47, <64 x float> %74, <256 x i1> %10)
  %79 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %49, <64 x float> %74, <256 x i1> %10)
  %80 = call <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float> %76, float -4.480000e+02, <256 x i1> %10)
  %81 = call <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float> %77, float -4.480000e+02, <256 x i1> %10)
  %82 = call <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float> %78, float -4.480000e+02, <256 x i1> %10)
  %83 = call <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float> %79, float -4.480000e+02, <256 x i1> %10)
  %84 = call <64 x float> @llvm.hivm.vmins.s.x.v64f32(<64 x float> %80, float 4.480000e+02, <256 x i1> %10)
  %85 = call <64 x float> @llvm.hivm.vmins.s.x.v64f32(<64 x float> %81, float 4.480000e+02, <256 x i1> %10)
  %86 = call <64 x float> @llvm.hivm.vmins.s.x.v64f32(<64 x float> %82, float 4.480000e+02, <256 x i1> %10)
  %87 = call <64 x float> @llvm.hivm.vmins.s.x.v64f32(<64 x float> %83, float 4.480000e+02, <256 x i1> %10)
  %88 = call <256 x float8e4m3> @llvm.hivm.vcvtff.f322f8e4m3.x(<64 x float> %84, <256 x i1> %10, i32 0, i32 0, i32 0)
  %89 = call <256 x float8e4m3> @llvm.hivm.vcvtff.f322f8e4m3.x(<64 x float> %85, <256 x i1> %10, i32 0, i32 0, i32 1)
  %90 = call <256 x float8e4m3> @llvm.hivm.vcvtff.f322f8e4m3.x(<64 x float> %86, <256 x i1> %10, i32 0, i32 0, i32 2)
  %91 = call <256 x float8e4m3> @llvm.hivm.vcvtff.f322f8e4m3.x(<64 x float> %87, <256 x i1> %10, i32 0, i32 0, i32 3)
  %92 = call <256 x float8e4m3> @llvm.hivm.vor.x.v256fp8e4m3(<256 x float8e4m3> %88, <256 x float8e4m3> %89, <256 x i1> %17)
  %93 = call <256 x float8e4m3> @llvm.hivm.vor.x.v256fp8e4m3(<256 x float8e4m3> %92, <256 x float8e4m3> %90, <256 x i1> %17)
  %94 = call <256 x float8e4m3> @llvm.hivm.vor.x.v256fp8e4m3(<256 x float8e4m3> %93, <256 x float8e4m3> %91, <256 x i1> %17)
  %95 = bitcast <256 x float8e4m3> %94 to <64 x i32>
  %96 = mul i64 %45, 4
  %97 = getelementptr i8, ptr addrspace(6) %4, i64 %96
  call void @llvm.hivm.vstsx1.v64i32(<64 x i32> %95, ptr addrspace(6) %97, i32 0, i32 2, i32 0, <256 x i1> %10)
  %98 = call <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32> %24, <64 x i32> %44, <256 x i1> %10)
  %99 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %75, <64 x i32> %22, <256 x i1> %98)
  %100 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %29, <64 x i32> %99, <256 x i1> %10)
  call void @llvm.hivm.mem.bar.vv.all()
  %101 = add i64 %28, 1
  br label %27

102:                                              ; preds = %27
  call void @llvm.hivm.mem.bar.vst.vst()
  %103 = mul i64 %14, 4
  %104 = getelementptr i8, ptr addrspace(6) %5, i64 %103
  call void @llvm.hivm.vstsx1.v64i32(<64 x i32> %29, ptr addrspace(6) %104, i32 0, i32 2, i32 0, <256 x i1> %10)
  %105 = sub i64 16, %2
  call void @llvm.hivm.mem.bar.vst.vst()
  br label %106

106:                                              ; preds = %109, %102
  %107 = phi i64 [ %115, %109 ], [ 0, %102 ]
  %108 = icmp slt i64 %107, %105
  br i1 %108, label %109, label %116

109:                                              ; preds = %106
  %110 = add i64 %2, %107
  %111 = add i64 %13, %110
  %112 = mul i64 %111, 64
  %113 = mul i64 %112, 4
  %114 = getelementptr i8, ptr addrspace(6) %4, i64 %113
  call void @llvm.hivm.vstsx1.v64i32(<64 x i32> %22, ptr addrspace(6) %114, i32 0, i32 2, i32 0, <256 x i1> %10)
  call void @llvm.hivm.mem.bar.vst.vst()
  %115 = add i64 %107, 1
  br label %106

116:                                              ; preds = %106
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %116
  %117 = add i64 %7, 1
  br label %6, !llvm.loop !1

118:                                              ; preds = %6
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
