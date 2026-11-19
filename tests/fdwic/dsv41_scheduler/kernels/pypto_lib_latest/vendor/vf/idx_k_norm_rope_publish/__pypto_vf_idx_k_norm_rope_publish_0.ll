; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x i32> @llvm.hivm.vci.v64i32(i32, i32)

declare <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6), i32, i32, i32)

declare <256 x i1> @llvm.hivm.pset.b16(i32)

declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat>, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32)

declare <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare void @llvm.hivm.mem.bar.vld.vst()

declare <256 x i1> @llvm.hivm.pnot.z(<256 x i1>, <256 x i1>)

declare <256 x i1> @llvm.hivm.pset.b8(i32)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float>, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float>, <256 x i1>, i32, i32, i32)

declare <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsel.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x i32> @llvm.hivm.vadds.s.x.v64i32(<64 x i32>, i32, <256 x i1>)

declare <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <256 x i1> @llvm.hivm.vcmp.ge.s.z.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare { <64 x i32>, <64 x i32> } @llvm.hivm.vdintlv.v64i32(<64 x i32>, <64 x i32>)

declare <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32>, i32, <256 x i1>)

declare <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6), <64 x i32>, <256 x i1>)

declare void @llvm.hivm.vstsx1.v256i8(<256 x i8>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare { <64 x float>, <64 x float> } @llvm.hivm.vdintlv.v64f32(<64 x float>, <64 x float>)

declare <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare { <64 x float>, <64 x float> } @llvm.hivm.vintlv.v64f32(<64 x float>, <64 x float>)

declare void @llvm.hivm.mem.bar.vst.vst()

declare <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32>, <64 x i32>)

declare void @llvm.hivm.mem.bar.vv.all()

define void @__pypto_vf_idx_k_norm_rope_publish_0(ptr addrspace(6) %0, i64 %1) #0 {
  %3 = getelementptr bfloat, ptr addrspace(6) %0, i64 2688
  %4 = getelementptr float, ptr addrspace(6) %0, i64 1920
  %5 = getelementptr bfloat, ptr addrspace(6) %0, i64 2816
  %6 = getelementptr i8, ptr addrspace(6) %0, i64 1024
  %7 = getelementptr i8, ptr addrspace(6) %0, i64 3072
  br label %8

8:                                                ; preds = %aivscope.latch, %2
  %9 = phi i64 [ %190, %aivscope.latch ], [ 0, %2 ]
  %10 = icmp slt i64 %9, 1
  br i1 %10, label %11, label %191

11:                                               ; preds = %8
  %12 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %13 = call <256 x i1> @llvm.hivm.pset.b32(i32 7)
  %14 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %15 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %3, i32 0, i32 14, i32 0)
  %16 = call <256 x i1> @llvm.hivm.pset.b16(i32 0)
  %17 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %3, i32 128, i32 14, i32 0)
  %18 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 31, <256 x i1> %12, i32 1)
  %19 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %15, <256 x i1> %16, i32 0)
  %20 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %17, <256 x i1> %16, i32 0)
  %21 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.000000e+00, <256 x i1> %12, i32 1)
  %22 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 0, <256 x i1> %12, i32 1)
  %23 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 32, <256 x i1> %12, i32 1)
  %24 = call <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32> %14, <64 x i32> %18, <256 x i1> %12)
  call void @llvm.hivm.mem.bar.vld.vst()
  %25 = call <256 x i1> @llvm.hivm.pnot.z(<256 x i1> %13, <256 x i1> %12)
  %26 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 2.500000e-01, <256 x i1> %12, i32 1)
  %27 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 1, <256 x i1> %12, i32 1)
  %28 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 7.500000e-01, <256 x i1> %12, i32 1)
  %29 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 2, <256 x i1> %12, i32 1)
  %30 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.250000e+00, <256 x i1> %12, i32 1)
  %31 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 3, <256 x i1> %12, i32 1)
  %32 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.750000e+00, <256 x i1> %12, i32 1)
  %33 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 4, <256 x i1> %12, i32 1)
  %34 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 2.500000e+00, <256 x i1> %12, i32 1)
  %35 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 5, <256 x i1> %12, i32 1)
  %36 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 3.500000e+00, <256 x i1> %12, i32 1)
  %37 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 6, <256 x i1> %12, i32 1)
  %38 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 5.000000e+00, <256 x i1> %12, i32 1)
  %39 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 7, <256 x i1> %12, i32 1)
  %40 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 8, <256 x i1> %12, i32 1)
  %41 = call <256 x i1> @llvm.hivm.pset.b8(i32 4)
  br label %42

42:                                               ; preds = %45, %11
  %43 = phi i64 [ %188, %45 ], [ 0, %11 ]
  %44 = icmp slt i64 %43, %1
  br i1 %44, label %45, label %189

45:                                               ; preds = %42
  %46 = mul i64 %43, 128
  %47 = add i64 %46, 64
  %48 = mul i64 %46, 2
  %49 = getelementptr i8, ptr addrspace(6) %5, i64 %48
  %50 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %49, i32 0, i32 14, i32 0)
  %51 = mul i64 %43, 256
  %52 = mul i64 %43, 32
  %53 = trunc i64 %52 to i32
  %54 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %50, <256 x i1> %16, i32 0)
  %55 = mul i64 %47, 2
  %56 = getelementptr i8, ptr addrspace(6) %5, i64 %55
  %57 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %56, i32 0, i32 14, i32 0)
  %58 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %54, <64 x float> %54, <256 x i1> %12)
  %59 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %57, <256 x i1> %16, i32 0)
  %60 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %58, <256 x i1> %12)
  %61 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %60, <256 x i1> %12, i32 1)
  %62 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %59, <64 x float> %59, <256 x i1> %12)
  %63 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %62, <256 x i1> %12)
  %64 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %63, <256 x i1> %12, i32 1)
  %65 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %61, <64 x float> %64, <256 x i1> %12)
  %66 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %65, float 7.812500e-03, <256 x i1> %12)
  %67 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %66, float 0x3EB0C6F7A0000000, <256 x i1> %12)
  %68 = call <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float> %67, <256 x i1> %12)
  %69 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %21, <64 x float> %68, <256 x i1> %12)
  %70 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %54, <64 x float> %69, <256 x i1> %12)
  %71 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %70, <64 x float> %19, <256 x i1> %12)
  %72 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %71, <256 x i1> %12, i32 0, i32 0, i32 0)
  %73 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %72, <256 x i1> %16, i32 0)
  %74 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %73, <256 x i1> %12)
  %75 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %74, <256 x i1> %13)
  %76 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %74, <256 x i1> %25)
  %77 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %75, <256 x i1> %12, i32 1)
  %78 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %76, <256 x i1> %12, i32 1)
  %79 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %78, <64 x float> %77, <256 x i1> %25)
  %80 = call <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float> %79, float 0x3838000000000000, <256 x i1> %12)
  %81 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %80, float 0x3FC5555560000000, <256 x i1> %12)
  %82 = bitcast <64 x float> %81 to <64 x i32>
  %83 = call <64 x i32> @llvm.hivm.vadds.s.x.v64i32(<64 x i32> %82, i32 8388607, <256 x i1> %12)
  %84 = call <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32> %83, i16 23, <256 x i1> %12)
  %85 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %84, i16 23, <256 x i1> %12)
  %86 = bitcast <64 x i32> %85 to <64 x float>
  %87 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %73, <64 x float> %86, <256 x i1> %12)
  %88 = bitcast <64 x float> %87 to <64 x i32>
  %89 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %87, <256 x i1> %12)
  %90 = call <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32> %88, i16 28, <256 x i1> %12)
  %91 = call <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32> %90, <64 x i32> %40, <256 x i1> %12)
  %92 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %89, <64 x float> %26, <256 x i1> %12)
  %93 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %27, <64 x i32> %22, <256 x i1> %92)
  %94 = call <256 x i1> @llvm.hivm.vcmp.ge.s.z.v64f32(<64 x float> %89, <64 x float> %28, <256 x i1> %12)
  %95 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %29, <64 x i32> %93, <256 x i1> %94)
  %96 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %89, <64 x float> %30, <256 x i1> %12)
  %97 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %31, <64 x i32> %95, <256 x i1> %96)
  %98 = call <256 x i1> @llvm.hivm.vcmp.ge.s.z.v64f32(<64 x float> %89, <64 x float> %32, <256 x i1> %12)
  %99 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %33, <64 x i32> %97, <256 x i1> %98)
  %100 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %89, <64 x float> %34, <256 x i1> %12)
  %101 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %35, <64 x i32> %99, <256 x i1> %100)
  %102 = call <256 x i1> @llvm.hivm.vcmp.ge.s.z.v64f32(<64 x float> %89, <64 x float> %36, <256 x i1> %12)
  %103 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %37, <64 x i32> %101, <256 x i1> %102)
  %104 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %89, <64 x float> %38, <256 x i1> %12)
  %105 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %39, <64 x i32> %103, <256 x i1> %104)
  %106 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %105, <64 x i32> %91, <256 x i1> %12)
  %107 = call { <64 x i32>, <64 x i32> } @llvm.hivm.vdintlv.v64i32(<64 x i32> %106, <64 x i32> %106)
  %108 = extractvalue { <64 x i32>, <64 x i32> } %107, 0
  %109 = extractvalue { <64 x i32>, <64 x i32> } %107, 1
  %110 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %109, i16 4, <256 x i1> %13)
  %111 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %108, <64 x i32> %110, <256 x i1> %12)
  %112 = bitcast <64 x i32> %111 to <256 x i8>
  %113 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %24, i32 %53, <256 x i1> %12)
  %114 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %4, <64 x i32> %113, <256 x i1> %12)
  %115 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %0, <64 x i32> %113, <256 x i1> %12)
  call void @llvm.hivm.mem.bar.vld.vst()
  %116 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %59, <64 x float> %69, <256 x i1> %12)
  %117 = getelementptr i8, ptr addrspace(6) %6, i64 %51
  call void @llvm.hivm.vstsx1.v256i8(<256 x i8> %112, ptr addrspace(6) %117, i32 0, i32 12, i32 0, <256 x i1> %13)
  %118 = add i64 %51, 32
  %119 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %116, <64 x float> %20, <256 x i1> %12)
  %120 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %119, <256 x i1> %12, i32 0, i32 0, i32 0)
  %121 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %120, <256 x i1> %16, i32 0)
  %122 = call { <64 x float>, <64 x float> } @llvm.hivm.vdintlv.v64f32(<64 x float> %121, <64 x float> %121)
  %123 = extractvalue { <64 x float>, <64 x float> } %122, 0
  %124 = extractvalue { <64 x float>, <64 x float> } %122, 1
  %125 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %123, <64 x float> %114, <256 x i1> %12)
  %126 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %124, <64 x float> %114, <256 x i1> %12)
  %127 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %124, <64 x float> %115, <256 x i1> %12)
  %128 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %123, <64 x float> %115, <256 x i1> %12)
  %129 = call <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float> %125, <64 x float> %127, <256 x i1> %12)
  %130 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %128, <64 x float> %126, <256 x i1> %12)
  %131 = call { <64 x float>, <64 x float> } @llvm.hivm.vintlv.v64f32(<64 x float> %129, <64 x float> %130)
  %132 = extractvalue { <64 x float>, <64 x float> } %131, 0
  %133 = extractvalue { <64 x float>, <64 x float> } %131, 1
  %134 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %132, <256 x i1> %12, i32 0, i32 0, i32 0)
  %135 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %134, <256 x i1> %16, i32 0)
  %136 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %135, <256 x i1> %12)
  %137 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %136, <256 x i1> %13)
  %138 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %136, <256 x i1> %25)
  %139 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %137, <256 x i1> %12, i32 1)
  %140 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %138, <256 x i1> %12, i32 1)
  %141 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %140, <64 x float> %139, <256 x i1> %25)
  %142 = call <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float> %141, float 0x3838000000000000, <256 x i1> %12)
  %143 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %142, float 0x3FC5555560000000, <256 x i1> %12)
  %144 = bitcast <64 x float> %143 to <64 x i32>
  %145 = call <64 x i32> @llvm.hivm.vadds.s.x.v64i32(<64 x i32> %144, i32 8388607, <256 x i1> %12)
  %146 = call <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32> %145, i16 23, <256 x i1> %12)
  %147 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %146, i16 23, <256 x i1> %12)
  %148 = bitcast <64 x i32> %147 to <64 x float>
  %149 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %135, <64 x float> %148, <256 x i1> %12)
  %150 = bitcast <64 x float> %149 to <64 x i32>
  %151 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %149, <256 x i1> %12)
  %152 = call <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32> %150, i16 28, <256 x i1> %12)
  %153 = call <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32> %152, <64 x i32> %40, <256 x i1> %12)
  %154 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %151, <64 x float> %26, <256 x i1> %12)
  %155 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %27, <64 x i32> %22, <256 x i1> %154)
  %156 = call <256 x i1> @llvm.hivm.vcmp.ge.s.z.v64f32(<64 x float> %151, <64 x float> %28, <256 x i1> %12)
  %157 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %29, <64 x i32> %155, <256 x i1> %156)
  %158 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %151, <64 x float> %30, <256 x i1> %12)
  %159 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %31, <64 x i32> %157, <256 x i1> %158)
  %160 = call <256 x i1> @llvm.hivm.vcmp.ge.s.z.v64f32(<64 x float> %151, <64 x float> %32, <256 x i1> %12)
  %161 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %33, <64 x i32> %159, <256 x i1> %160)
  %162 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %151, <64 x float> %34, <256 x i1> %12)
  %163 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %35, <64 x i32> %161, <256 x i1> %162)
  %164 = call <256 x i1> @llvm.hivm.vcmp.ge.s.z.v64f32(<64 x float> %151, <64 x float> %36, <256 x i1> %12)
  %165 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %37, <64 x i32> %163, <256 x i1> %164)
  %166 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %151, <64 x float> %38, <256 x i1> %12)
  %167 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %39, <64 x i32> %165, <256 x i1> %166)
  %168 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %167, <64 x i32> %153, <256 x i1> %12)
  %169 = call { <64 x i32>, <64 x i32> } @llvm.hivm.vdintlv.v64i32(<64 x i32> %168, <64 x i32> %168)
  %170 = extractvalue { <64 x i32>, <64 x i32> } %169, 0
  %171 = extractvalue { <64 x i32>, <64 x i32> } %169, 1
  %172 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %171, i16 4, <256 x i1> %13)
  %173 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %170, <64 x i32> %172, <256 x i1> %12)
  %174 = bitcast <64 x i32> %173 to <256 x i8>
  call void @llvm.hivm.mem.bar.vst.vst()
  %175 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %84, <64 x i32> %23)
  %176 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %84, <64 x i32> %22)
  %177 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %146, <64 x i32> %22)
  %178 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %146, <64 x i32> %23)
  %179 = getelementptr i8, ptr addrspace(6) %6, i64 %118
  call void @llvm.hivm.vstsx1.v256i8(<256 x i8> %174, ptr addrspace(6) %179, i32 0, i32 12, i32 0, <256 x i1> %13)
  %180 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %177, i16 16, <256 x i1> %12)
  %181 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %178, i16 24, <256 x i1> %12)
  %182 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %175, i16 8, <256 x i1> %12)
  %183 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %176, <64 x i32> %182, <256 x i1> %12)
  %184 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %180, <64 x i32> %181, <256 x i1> %12)
  %185 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %183, <64 x i32> %184, <256 x i1> %12)
  %186 = bitcast <64 x i32> %185 to <256 x i8>
  call void @llvm.hivm.mem.bar.vst.vst()
  %187 = getelementptr i8, ptr addrspace(6) %7, i64 %51
  call void @llvm.hivm.vstsx1.v256i8(<256 x i8> %186, ptr addrspace(6) %187, i32 0, i32 0, i32 0, <256 x i1> %41)
  call void @llvm.hivm.mem.bar.vv.all()
  %188 = add i64 %43, 1
  br label %42

189:                                              ; preds = %42
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %189
  %190 = add i64 %9, 1
  br label %8, !llvm.loop !1

191:                                              ; preds = %8
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
