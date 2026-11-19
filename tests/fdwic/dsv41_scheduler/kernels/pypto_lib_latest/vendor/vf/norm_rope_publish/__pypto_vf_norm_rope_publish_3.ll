; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x i32> @llvm.hivm.vci.v64i32(i32, i32)

declare <256 x i1> @llvm.hivm.pset.b16(i32)

declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32)

declare <256 x i1> @llvm.hivm.pnot.z(<256 x i1>, <256 x i1>)

declare <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat>, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float>, <256 x i1>, i32)

declare <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsel.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float>, <64 x float>, <64 x float>, <256 x i1>)

declare { <128 x bfloat>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v128bf16(ptr addrspace(6), i32, i32, i32)

declare <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float>, <256 x i1>, i32, i32, i32)

declare <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x i32> @llvm.hivm.vadds.s.x.v64i32(<64 x i32>, i32, <256 x i1>)

declare <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32>, <64 x i32>)

declare <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <256 x float8e4m3> @llvm.hivm.vcvtff.f322f8e4m3.x(<64 x float>, <256 x i1>, i32, i32, i32)

declare void @llvm.hivm.vstsx1.v256i8(<256 x i8>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32)

declare <256 x i1> @llvm.hivm.pxor.z(<256 x i1>, <256 x i1>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare { <64 x float>, <64 x float> } @llvm.hivm.vdintlv.v64f32(<64 x float>, <64 x float>)

declare <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare { <64 x float>, <64 x float> } @llvm.hivm.vintlv.v64f32(<64 x float>, <64 x float>)

declare ptr addrspace(6) @llvm.hivm.vstsx1.post.v256i8(<256 x i8>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare void @llvm.hivm.vstsx1.v64i32(<64 x i32>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

define void @__pypto_vf_norm_rope_publish_3(ptr addrspace(6) %0) #0 {
  %2 = getelementptr bfloat, ptr addrspace(6) %0, i64 11392
  %3 = getelementptr float, ptr addrspace(6) %0, i64 7744
  %4 = getelementptr float, ptr addrspace(6) %0, i64 8000
  %5 = getelementptr bfloat, ptr addrspace(6) %0, i64 10880
  %6 = getelementptr i32, ptr addrspace(6) %0, i64 1024
  br label %7

7:                                                ; preds = %aivscope.latch, %1
  %8 = phi i64 [ %229, %aivscope.latch ], [ 0, %1 ]
  %9 = icmp slt i64 %8, 1
  br i1 %9, label %10, label %230

10:                                               ; preds = %7
  %11 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %12 = call <256 x i1> @llvm.hivm.pset.b32(i32 7)
  %13 = getelementptr float, ptr addrspace(6) %3, i64 0
  %14 = getelementptr float, ptr addrspace(6) %4, i64 0
  %15 = getelementptr bfloat, ptr addrspace(6) %2, i64 448
  %16 = getelementptr i8, ptr addrspace(6) %0, i64 448
  %17 = getelementptr bfloat, ptr addrspace(6) %5, i64 0
  %18 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %19 = call <256 x i1> @llvm.hivm.pset.b16(i32 0)
  %20 = call <256 x i1> @llvm.hivm.pset.b32(i32 4)
  %21 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 2139095040, <256 x i1> %11, i32 1)
  %22 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 0, <256 x i1> %11, i32 1)
  %23 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 63, <256 x i1> %11, i32 1)
  %24 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.000000e+00, <256 x i1> %11, i32 1)
  %25 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %11, i32 1)
  %26 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.500000e+00, <256 x i1> %11, i32 1)
  %27 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 5.000000e-01, <256 x i1> %11, i32 1)
  %28 = bitcast <64 x i32> %21 to <64 x float>
  %29 = call <256 x i1> @llvm.hivm.pnot.z(<256 x i1> %12, <256 x i1> %11)
  br label %30

30:                                               ; preds = %161, %10
  %31 = phi i16 [ %227, %161 ], [ 0, %10 ]
  %32 = phi <64 x i32> [ %226, %161 ], [ %22, %10 ]
  %33 = phi ptr addrspace(6) [ %73, %161 ], [ %13, %10 ]
  %34 = phi ptr addrspace(6) [ %76, %161 ], [ %14, %10 ]
  %35 = phi ptr addrspace(6) [ %164, %161 ], [ %15, %10 ]
  %36 = phi ptr addrspace(6) [ %218, %161 ], [ %16, %10 ]
  %37 = icmp slt i16 %31, 4
  br i1 %37, label %38, label %228

38:                                               ; preds = %30
  %39 = sext i16 %31 to i64
  %40 = mul i64 %39, 512
  br label %41

41:                                               ; preds = %45, %38
  %42 = phi i16 [ %67, %45 ], [ 0, %38 ]
  %43 = phi <64 x float> [ %66, %45 ], [ %25, %38 ]
  %44 = icmp slt i16 %42, 4
  br i1 %44, label %45, label %68

45:                                               ; preds = %41
  %46 = sext i16 %42 to i64
  %47 = mul i64 %46, 64
  %48 = add i64 %40, %47
  %49 = add i64 %48, 256
  %50 = mul i64 %48, 2
  %51 = getelementptr i8, ptr addrspace(6) %2, i64 %50
  %52 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %51, i32 0, i32 14, i32 0)
  %53 = mul i64 %49, 2
  %54 = getelementptr i8, ptr addrspace(6) %2, i64 %53
  %55 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %54, i32 0, i32 14, i32 0)
  %56 = trunc i64 %46 to i32
  %57 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %56, <256 x i1> %11, i32 1)
  %58 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %52, <256 x i1> %19, i32 0)
  %59 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %55, <256 x i1> %19, i32 0)
  %60 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %58, <64 x float> %58, <256 x i1> %11)
  %61 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %59, <64 x float> %59, <256 x i1> %11)
  %62 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %60, <64 x float> %61, <256 x i1> %11)
  %63 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %62, <256 x i1> %11)
  %64 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %63, <256 x i1> %11, i32 1)
  %65 = call <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32> %18, <64 x i32> %57, <256 x i1> %11)
  %66 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %64, <64 x float> %43, <256 x i1> %65)
  %67 = add i16 %42, 1
  br label %41

68:                                               ; preds = %41
  %69 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %43, <256 x i1> %20)
  %70 = mul i64 %39, 8
  %71 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %33, i32 128, i32 0, i32 1)
  %72 = extractvalue { <64 x float>, ptr addrspace(6) } %71, 0
  %73 = extractvalue { <64 x float>, ptr addrspace(6) } %71, 1
  %74 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %34, i32 128, i32 0, i32 1)
  %75 = extractvalue { <64 x float>, ptr addrspace(6) } %74, 0
  %76 = extractvalue { <64 x float>, ptr addrspace(6) } %74, 1
  %77 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %69, <256 x i1> %11, i32 1)
  %78 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %77, float 0x3F60000000000000, <256 x i1> %11)
  %79 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %78, float 0x3EB0C6F7A0000000, <256 x i1> %11)
  %80 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %24, <64 x float> %79, <256 x i1> %11)
  %81 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %79, float -5.000000e-01, <256 x i1> %11)
  %82 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %79, float -1.000000e+00, <256 x i1> %11)
  %83 = bitcast <64 x float> %79 to <64 x i32>
  %84 = call <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float> %80, <256 x i1> %11)
  %85 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %24, <64 x float> %82, <64 x float> %80, <256 x i1> %11)
  %86 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %81, <64 x float> %84, <256 x i1> %11)
  %87 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %26, <64 x float> %86, <64 x float> %84, <256 x i1> %11)
  %88 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %84, <64 x float> %87, <256 x i1> %11)
  %89 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %88, float -1.000000e+00, <256 x i1> %11)
  %90 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %80, <64 x float> %89, <64 x float> %88, <256 x i1> %11)
  %91 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %85, <64 x float> %79, <64 x float> %90, <256 x i1> %11)
  %92 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %91, <64 x float> %88, <256 x i1> %11)
  %93 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %88, <64 x float> %92, <64 x float> %27, <256 x i1> %11)
  %94 = call <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32> %83, <64 x i32> %21, <256 x i1> %11)
  %95 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %25, <64 x float> %93, <256 x i1> %94)
  %96 = call <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32> %83, <64 x i32> %22, <256 x i1> %11)
  %97 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %28, <64 x float> %95, <256 x i1> %96)
  br label %98

98:                                               ; preds = %103, %68
  %99 = phi i16 [ %160, %103 ], [ 0, %68 ]
  %100 = phi <64 x i32> [ %159, %103 ], [ %32, %68 ]
  %101 = phi ptr addrspace(6) [ %112, %103 ], [ %17, %68 ]
  %102 = icmp slt i16 %99, 7
  br i1 %102, label %103, label %161

103:                                              ; preds = %98
  %104 = sext i16 %99 to i64
  %105 = mul i64 %104, 64
  %106 = add i64 %40, %105
  %107 = mul i64 %106, 2
  %108 = getelementptr i8, ptr addrspace(6) %2, i64 %107
  %109 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %108, i32 0, i32 14, i32 0)
  %110 = call { <128 x bfloat>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v128bf16(ptr addrspace(6) %101, i32 128, i32 14, i32 1)
  %111 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %110, 0
  %112 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %110, 1
  %113 = add i64 %70, %104
  %114 = trunc i64 %113 to i32
  %115 = icmp sgt i32 %114, 0
  %116 = select i1 %115, i32 %114, i32 0
  %117 = icmp ult i32 %116, 64
  %118 = select i1 %117, i32 %116, i32 64
  %119 = add i64 %113, 1
  %120 = trunc i64 %119 to i32
  %121 = icmp sgt i32 %120, 0
  %122 = select i1 %121, i32 %120, i32 0
  %123 = icmp ult i32 %122, 64
  %124 = select i1 %123, i32 %122, i32 64
  %125 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %109, <256 x i1> %19, i32 0)
  %126 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %111, <256 x i1> %19, i32 0)
  %127 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %125, <64 x float> %97, <256 x i1> %11)
  %128 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %127, <64 x float> %126, <256 x i1> %11)
  %129 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %128, <256 x i1> %11, i32 0, i32 0, i32 0)
  %130 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %129, <256 x i1> %19, i32 0)
  %131 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %130, <256 x i1> %11)
  %132 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %131, <256 x i1> %12)
  %133 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %131, <256 x i1> %29)
  %134 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %132, <256 x i1> %11, i32 1)
  %135 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %133, <256 x i1> %11, i32 1)
  %136 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %134, <64 x float> %135, <256 x i1> %12)
  %137 = call <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float> %136, float 0x3F1A36E2E0000000, <256 x i1> %11)
  %138 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %137, float 0x3F624924A0000000, <256 x i1> %11)
  %139 = bitcast <64 x float> %138 to <64 x i32>
  %140 = call <64 x i32> @llvm.hivm.vadds.s.x.v64i32(<64 x i32> %139, i32 8388607, <256 x i1> %11)
  %141 = call <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32> %140, i16 23, <256 x i1> %11)
  %142 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %141, i16 23, <256 x i1> %11)
  %143 = bitcast <64 x i32> %142 to <64 x float>
  %144 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %130, <64 x float> %143, <256 x i1> %11)
  %145 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %142, <64 x i32> %23)
  %146 = call <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32> %142, i16 16, <256 x i1> %11)
  %147 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %146, <64 x i32> %145, <256 x i1> %11)
  %148 = call <256 x float8e4m3> @llvm.hivm.vcvtff.f322f8e4m3.x(<64 x float> %144, <256 x i1> %11, i32 0, i32 0, i32 0)
  %149 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %147, <64 x i32> %22)
  %150 = bitcast <256 x float8e4m3> %148 to <256 x i8>
  %151 = getelementptr i8, ptr addrspace(6) %0, i64 %106
  call void @llvm.hivm.vstsx1.v256i8(<256 x i8> %150, ptr addrspace(6) %151, i32 0, i32 12, i32 0, <256 x i1> %11)
  %152 = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 %118)
  %153 = extractvalue { <256 x i1>, i32 } %152, 0
  %154 = extractvalue { <256 x i1>, i32 } %152, 1
  %155 = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 %124)
  %156 = extractvalue { <256 x i1>, i32 } %155, 0
  %157 = extractvalue { <256 x i1>, i32 } %155, 1
  %158 = call <256 x i1> @llvm.hivm.pxor.z(<256 x i1> %153, <256 x i1> %156, <256 x i1> %11)
  %159 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %149, <64 x i32> %100, <256 x i1> %158)
  %160 = add i16 %99, 1
  br label %98

161:                                              ; preds = %98
  %162 = call { <128 x bfloat>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v128bf16(ptr addrspace(6) %35, i32 1024, i32 14, i32 1)
  %163 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %162, 0
  %164 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %162, 1
  %165 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %5, i32 896, i32 14, i32 0)
  %166 = add i64 %70, 7
  %167 = trunc i64 %166 to i32
  %168 = icmp sgt i32 %167, 0
  %169 = select i1 %168, i32 %167, i32 0
  %170 = icmp ult i32 %169, 64
  %171 = select i1 %170, i32 %169, i32 64
  %172 = add i64 %70, 8
  %173 = trunc i64 %172 to i32
  %174 = icmp sgt i32 %173, 0
  %175 = select i1 %174, i32 %173, i32 0
  %176 = icmp ult i32 %175, 64
  %177 = select i1 %176, i32 %175, i32 64
  %178 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %163, <256 x i1> %19, i32 0)
  %179 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %165, <256 x i1> %19, i32 0)
  %180 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %178, <64 x float> %97, <256 x i1> %11)
  %181 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %180, <64 x float> %179, <256 x i1> %11)
  %182 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %181, <256 x i1> %11, i32 0, i32 0, i32 0)
  %183 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %182, <256 x i1> %19, i32 0)
  %184 = call { <64 x float>, <64 x float> } @llvm.hivm.vdintlv.v64f32(<64 x float> %183, <64 x float> %183)
  %185 = extractvalue { <64 x float>, <64 x float> } %184, 0
  %186 = extractvalue { <64 x float>, <64 x float> } %184, 1
  %187 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %185, <64 x float> %72, <256 x i1> %12)
  %188 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %185, <64 x float> %75, <256 x i1> %12)
  %189 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %186, <64 x float> %75, <256 x i1> %12)
  %190 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %186, <64 x float> %72, <256 x i1> %12)
  %191 = call <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float> %187, <64 x float> %189, <256 x i1> %12)
  %192 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %188, <64 x float> %190, <256 x i1> %12)
  %193 = call { <64 x float>, <64 x float> } @llvm.hivm.vintlv.v64f32(<64 x float> %191, <64 x float> %192)
  %194 = extractvalue { <64 x float>, <64 x float> } %193, 0
  %195 = extractvalue { <64 x float>, <64 x float> } %193, 1
  %196 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %194, <256 x i1> %11, i32 0, i32 0, i32 0)
  %197 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %196, <256 x i1> %19, i32 0)
  %198 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %197, <256 x i1> %11)
  %199 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %198, <256 x i1> %12)
  %200 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %198, <256 x i1> %29)
  %201 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %199, <256 x i1> %11, i32 1)
  %202 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %200, <256 x i1> %11, i32 1)
  %203 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %201, <64 x float> %202, <256 x i1> %12)
  %204 = call <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float> %203, float 0x3F1A36E2E0000000, <256 x i1> %11)
  %205 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %204, float 0x3F624924A0000000, <256 x i1> %11)
  %206 = bitcast <64 x float> %205 to <64 x i32>
  %207 = call <64 x i32> @llvm.hivm.vadds.s.x.v64i32(<64 x i32> %206, i32 8388607, <256 x i1> %11)
  %208 = call <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32> %207, i16 23, <256 x i1> %11)
  %209 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %208, i16 23, <256 x i1> %11)
  %210 = bitcast <64 x i32> %209 to <64 x float>
  %211 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %197, <64 x float> %210, <256 x i1> %11)
  %212 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %209, <64 x i32> %23)
  %213 = call <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32> %209, i16 16, <256 x i1> %11)
  %214 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %213, <64 x i32> %212, <256 x i1> %11)
  %215 = call <256 x float8e4m3> @llvm.hivm.vcvtff.f322f8e4m3.x(<64 x float> %211, <256 x i1> %11, i32 0, i32 0, i32 0)
  %216 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %214, <64 x i32> %22)
  %217 = bitcast <256 x float8e4m3> %215 to <256 x i8>
  %218 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v256i8(<256 x i8> %217, ptr addrspace(6) %36, i32 512, i32 12, i32 1, <256 x i1> %11)
  %219 = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 %171)
  %220 = extractvalue { <256 x i1>, i32 } %219, 0
  %221 = extractvalue { <256 x i1>, i32 } %219, 1
  %222 = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 %177)
  %223 = extractvalue { <256 x i1>, i32 } %222, 0
  %224 = extractvalue { <256 x i1>, i32 } %222, 1
  %225 = call <256 x i1> @llvm.hivm.pxor.z(<256 x i1> %220, <256 x i1> %223, <256 x i1> %11)
  %226 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %216, <64 x i32> %100, <256 x i1> %225)
  %227 = add i16 %31, 1
  br label %30

228:                                              ; preds = %30
  call void @llvm.hivm.vstsx1.v64i32(<64 x i32> %32, ptr addrspace(6) %6, i32 0, i32 2, i32 0, <256 x i1> %11)
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %228
  %229 = add i64 %8, 1
  br label %7, !llvm.loop !1

230:                                              ; preds = %7
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
