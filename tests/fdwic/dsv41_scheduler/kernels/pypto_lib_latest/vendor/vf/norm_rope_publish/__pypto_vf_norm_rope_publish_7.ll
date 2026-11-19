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

declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float>, <64 x float>, <64 x float>, <256 x i1>)

declare void @llvm.hivm.mem.bar.vld.vst()

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

declare void @llvm.hivm.mem.bar.vv.all()

declare void @llvm.hivm.mem.bar.vst.vld()

declare { <64 x float>, <64 x float> } @llvm.hivm.vdintlv.v64f32(<64 x float>, <64 x float>)

declare <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare { <64 x float>, <64 x float> } @llvm.hivm.vintlv.v64f32(<64 x float>, <64 x float>)

declare void @llvm.hivm.vstsx1.v64i32(<64 x i32>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

define void @__pypto_vf_norm_rope_publish_7(ptr addrspace(6) %0, i64 %1) #0 {
  %3 = getelementptr bfloat, ptr addrspace(6) %0, i64 2688
  %4 = getelementptr float, ptr addrspace(6) %0, i64 3392
  %5 = getelementptr float, ptr addrspace(6) %0, i64 3648
  %6 = getelementptr bfloat, ptr addrspace(6) %0, i64 2176
  %7 = getelementptr i32, ptr addrspace(6) %0, i64 1024
  br label %8

8:                                                ; preds = %aivscope.latch, %2
  %9 = phi i64 [ %223, %aivscope.latch ], [ 0, %2 ]
  %10 = icmp slt i64 %9, 1
  br i1 %10, label %11, label %224

11:                                               ; preds = %8
  %12 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %13 = call <256 x i1> @llvm.hivm.pset.b32(i32 7)
  %14 = getelementptr bfloat, ptr addrspace(6) %6, i64 0
  %15 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %16 = call <256 x i1> @llvm.hivm.pset.b16(i32 0)
  %17 = call <256 x i1> @llvm.hivm.pset.b32(i32 4)
  %18 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 2139095040, <256 x i1> %12, i32 1)
  %19 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 0, <256 x i1> %12, i32 1)
  %20 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 63, <256 x i1> %12, i32 1)
  %21 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.000000e+00, <256 x i1> %12, i32 1)
  %22 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %12, i32 1)
  %23 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.500000e+00, <256 x i1> %12, i32 1)
  %24 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 5.000000e-01, <256 x i1> %12, i32 1)
  %25 = bitcast <64 x i32> %18 to <64 x float>
  %26 = call <256 x i1> @llvm.hivm.pnot.z(<256 x i1> %13, <256 x i1> %12)
  br label %27

27:                                               ; preds = %154, %11
  %28 = phi i64 [ %221, %154 ], [ 0, %11 ]
  %29 = phi <64 x i32> [ %220, %154 ], [ %19, %11 ]
  %30 = icmp slt i64 %28, %1
  br i1 %30, label %31, label %222

31:                                               ; preds = %27
  %32 = mul i64 %28, 512
  br label %33

33:                                               ; preds = %37, %31
  %34 = phi i16 [ %59, %37 ], [ 0, %31 ]
  %35 = phi <64 x float> [ %58, %37 ], [ %22, %31 ]
  %36 = icmp slt i16 %34, 4
  br i1 %36, label %37, label %60

37:                                               ; preds = %33
  %38 = sext i16 %34 to i64
  %39 = mul i64 %38, 64
  %40 = add i64 %32, %39
  %41 = add i64 %40, 256
  %42 = mul i64 %40, 2
  %43 = getelementptr i8, ptr addrspace(6) %3, i64 %42
  %44 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %43, i32 0, i32 14, i32 0)
  %45 = mul i64 %41, 2
  %46 = getelementptr i8, ptr addrspace(6) %3, i64 %45
  %47 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %46, i32 0, i32 14, i32 0)
  %48 = trunc i64 %38 to i32
  %49 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 %48, <256 x i1> %12, i32 1)
  %50 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %44, <256 x i1> %16, i32 0)
  %51 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %47, <256 x i1> %16, i32 0)
  %52 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %50, <64 x float> %50, <256 x i1> %12)
  %53 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %51, <64 x float> %51, <256 x i1> %12)
  %54 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %52, <64 x float> %53, <256 x i1> %12)
  %55 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %54, <256 x i1> %12)
  %56 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %55, <256 x i1> %12, i32 1)
  %57 = call <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32> %15, <64 x i32> %49, <256 x i1> %12)
  %58 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %56, <64 x float> %35, <256 x i1> %57)
  %59 = add i16 %34, 1
  br label %33

60:                                               ; preds = %33
  %61 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> %35, <256 x i1> %17)
  %62 = mul i64 %28, 32
  %63 = mul i64 %62, 4
  %64 = getelementptr i8, ptr addrspace(6) %4, i64 %63
  %65 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %64, i32 0, i32 0, i32 0)
  %66 = mul i64 %62, 4
  %67 = getelementptr i8, ptr addrspace(6) %5, i64 %66
  %68 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %67, i32 0, i32 0, i32 0)
  %69 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %61, <256 x i1> %12, i32 1)
  %70 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %69, float 0x3F60000000000000, <256 x i1> %12)
  %71 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %70, float 0x3EB0C6F7A0000000, <256 x i1> %12)
  %72 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %21, <64 x float> %71, <256 x i1> %12)
  %73 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %71, float -5.000000e-01, <256 x i1> %12)
  %74 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %71, float -1.000000e+00, <256 x i1> %12)
  %75 = bitcast <64 x float> %71 to <64 x i32>
  %76 = call <64 x float> @llvm.hivm.vsqrt.x.v64f32(<64 x float> %72, <256 x i1> %12)
  %77 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %21, <64 x float> %74, <64 x float> %72, <256 x i1> %12)
  %78 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %73, <64 x float> %76, <256 x i1> %12)
  %79 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %23, <64 x float> %78, <64 x float> %76, <256 x i1> %12)
  %80 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %76, <64 x float> %79, <256 x i1> %12)
  %81 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %80, float -1.000000e+00, <256 x i1> %12)
  %82 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %72, <64 x float> %81, <64 x float> %80, <256 x i1> %12)
  %83 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %77, <64 x float> %71, <64 x float> %82, <256 x i1> %12)
  %84 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %83, <64 x float> %80, <256 x i1> %12)
  %85 = call <64 x float> @llvm.hivm.vmula.s.m.v64f32(<64 x float> %80, <64 x float> %84, <64 x float> %24, <256 x i1> %12)
  %86 = call <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32> %75, <64 x i32> %18, <256 x i1> %12)
  %87 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %22, <64 x float> %85, <256 x i1> %86)
  %88 = call <256 x i1> @llvm.hivm.vcmp.eq.u.z.v64i32(<64 x i32> %75, <64 x i32> %19, <256 x i1> %12)
  %89 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %25, <64 x float> %87, <256 x i1> %88)
  call void @llvm.hivm.mem.bar.vld.vst()
  %90 = mul i64 %28, 8
  br label %91

91:                                               ; preds = %96, %60
  %92 = phi i16 [ %153, %96 ], [ 0, %60 ]
  %93 = phi <64 x i32> [ %152, %96 ], [ %29, %60 ]
  %94 = phi ptr addrspace(6) [ %105, %96 ], [ %14, %60 ]
  %95 = icmp slt i16 %92, 7
  br i1 %95, label %96, label %154

96:                                               ; preds = %91
  %97 = sext i16 %92 to i64
  %98 = mul i64 %97, 64
  %99 = add i64 %32, %98
  %100 = mul i64 %99, 2
  %101 = getelementptr i8, ptr addrspace(6) %3, i64 %100
  %102 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %101, i32 0, i32 14, i32 0)
  %103 = call { <128 x bfloat>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v128bf16(ptr addrspace(6) %94, i32 128, i32 14, i32 1)
  %104 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %103, 0
  %105 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %103, 1
  %106 = add i64 %90, %97
  %107 = trunc i64 %106 to i32
  %108 = icmp sgt i32 %107, 0
  %109 = select i1 %108, i32 %107, i32 0
  %110 = icmp ult i32 %109, 64
  %111 = select i1 %110, i32 %109, i32 64
  %112 = add i64 %106, 1
  %113 = trunc i64 %112 to i32
  %114 = icmp sgt i32 %113, 0
  %115 = select i1 %114, i32 %113, i32 0
  %116 = icmp ult i32 %115, 64
  %117 = select i1 %116, i32 %115, i32 64
  %118 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %102, <256 x i1> %16, i32 0)
  %119 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %104, <256 x i1> %16, i32 0)
  %120 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %118, <64 x float> %89, <256 x i1> %12)
  %121 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %120, <64 x float> %119, <256 x i1> %12)
  %122 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %121, <256 x i1> %12, i32 0, i32 0, i32 0)
  %123 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %122, <256 x i1> %16, i32 0)
  %124 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %123, <256 x i1> %12)
  %125 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %124, <256 x i1> %13)
  %126 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %124, <256 x i1> %26)
  %127 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %125, <256 x i1> %12, i32 1)
  %128 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %126, <256 x i1> %12, i32 1)
  %129 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %127, <64 x float> %128, <256 x i1> %13)
  %130 = call <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float> %129, float 0x3F1A36E2E0000000, <256 x i1> %12)
  %131 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %130, float 0x3F624924A0000000, <256 x i1> %12)
  %132 = bitcast <64 x float> %131 to <64 x i32>
  %133 = call <64 x i32> @llvm.hivm.vadds.s.x.v64i32(<64 x i32> %132, i32 8388607, <256 x i1> %12)
  %134 = call <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32> %133, i16 23, <256 x i1> %12)
  %135 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %134, i16 23, <256 x i1> %12)
  %136 = bitcast <64 x i32> %135 to <64 x float>
  %137 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %123, <64 x float> %136, <256 x i1> %12)
  %138 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %135, <64 x i32> %20)
  %139 = call <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32> %135, i16 16, <256 x i1> %12)
  %140 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %139, <64 x i32> %138, <256 x i1> %12)
  %141 = call <256 x float8e4m3> @llvm.hivm.vcvtff.f322f8e4m3.x(<64 x float> %137, <256 x i1> %12, i32 0, i32 0, i32 0)
  %142 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %140, <64 x i32> %19)
  %143 = bitcast <256 x float8e4m3> %141 to <256 x i8>
  %144 = getelementptr i8, ptr addrspace(6) %0, i64 %99
  call void @llvm.hivm.vstsx1.v256i8(<256 x i8> %143, ptr addrspace(6) %144, i32 0, i32 12, i32 0, <256 x i1> %12)
  %145 = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 %111)
  %146 = extractvalue { <256 x i1>, i32 } %145, 0
  %147 = extractvalue { <256 x i1>, i32 } %145, 1
  %148 = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 %117)
  %149 = extractvalue { <256 x i1>, i32 } %148, 0
  %150 = extractvalue { <256 x i1>, i32 } %148, 1
  %151 = call <256 x i1> @llvm.hivm.pxor.z(<256 x i1> %146, <256 x i1> %149, <256 x i1> %12)
  %152 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %142, <64 x i32> %93, <256 x i1> %151)
  call void @llvm.hivm.mem.bar.vv.all()
  %153 = add i16 %92, 1
  br label %91

154:                                              ; preds = %91
  %155 = add i64 %32, 448
  call void @llvm.hivm.mem.bar.vst.vld()
  %156 = mul i64 %155, 2
  %157 = getelementptr i8, ptr addrspace(6) %3, i64 %156
  %158 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %157, i32 0, i32 14, i32 0)
  %159 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %6, i32 896, i32 14, i32 0)
  %160 = add i64 %90, 7
  %161 = trunc i64 %160 to i32
  %162 = icmp sgt i32 %161, 0
  %163 = select i1 %162, i32 %161, i32 0
  %164 = icmp ult i32 %163, 64
  %165 = select i1 %164, i32 %163, i32 64
  %166 = add i64 %90, 8
  %167 = trunc i64 %166 to i32
  %168 = icmp sgt i32 %167, 0
  %169 = select i1 %168, i32 %167, i32 0
  %170 = icmp ult i32 %169, 64
  %171 = select i1 %170, i32 %169, i32 64
  %172 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %158, <256 x i1> %16, i32 0)
  %173 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %159, <256 x i1> %16, i32 0)
  %174 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %172, <64 x float> %89, <256 x i1> %12)
  %175 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %174, <64 x float> %173, <256 x i1> %12)
  %176 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %175, <256 x i1> %12, i32 0, i32 0, i32 0)
  %177 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %176, <256 x i1> %16, i32 0)
  %178 = call { <64 x float>, <64 x float> } @llvm.hivm.vdintlv.v64f32(<64 x float> %177, <64 x float> %177)
  %179 = extractvalue { <64 x float>, <64 x float> } %178, 0
  %180 = extractvalue { <64 x float>, <64 x float> } %178, 1
  %181 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %179, <64 x float> %65, <256 x i1> %13)
  %182 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %179, <64 x float> %68, <256 x i1> %13)
  %183 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %180, <64 x float> %68, <256 x i1> %13)
  %184 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %180, <64 x float> %65, <256 x i1> %13)
  %185 = call <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float> %181, <64 x float> %183, <256 x i1> %13)
  %186 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %182, <64 x float> %184, <256 x i1> %13)
  %187 = call { <64 x float>, <64 x float> } @llvm.hivm.vintlv.v64f32(<64 x float> %185, <64 x float> %186)
  %188 = extractvalue { <64 x float>, <64 x float> } %187, 0
  %189 = extractvalue { <64 x float>, <64 x float> } %187, 1
  %190 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %188, <256 x i1> %12, i32 0, i32 0, i32 0)
  %191 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %190, <256 x i1> %16, i32 0)
  %192 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %191, <256 x i1> %12)
  %193 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %192, <256 x i1> %13)
  %194 = call <64 x float> @llvm.hivm.vcmax.s.x.v64f32(<64 x float> %192, <256 x i1> %26)
  %195 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %193, <256 x i1> %12, i32 1)
  %196 = call <64 x float> @llvm.hivm.vdup.z.v64f32(<64 x float> %194, <256 x i1> %12, i32 1)
  %197 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %195, <64 x float> %196, <256 x i1> %13)
  %198 = call <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float> %197, float 0x3F1A36E2E0000000, <256 x i1> %12)
  %199 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %198, float 0x3F624924A0000000, <256 x i1> %12)
  %200 = bitcast <64 x float> %199 to <64 x i32>
  %201 = call <64 x i32> @llvm.hivm.vadds.s.x.v64i32(<64 x i32> %200, i32 8388607, <256 x i1> %12)
  %202 = call <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32> %201, i16 23, <256 x i1> %12)
  %203 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %202, i16 23, <256 x i1> %12)
  %204 = bitcast <64 x i32> %203 to <64 x float>
  %205 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %191, <64 x float> %204, <256 x i1> %12)
  %206 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %203, <64 x i32> %20)
  %207 = call <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32> %203, i16 16, <256 x i1> %12)
  %208 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %207, <64 x i32> %206, <256 x i1> %12)
  %209 = call <256 x float8e4m3> @llvm.hivm.vcvtff.f322f8e4m3.x(<64 x float> %205, <256 x i1> %12, i32 0, i32 0, i32 0)
  %210 = call <64 x i32> @llvm.hivm.vselr.v64i32(<64 x i32> %208, <64 x i32> %19)
  %211 = bitcast <256 x float8e4m3> %209 to <256 x i8>
  %212 = getelementptr i8, ptr addrspace(6) %0, i64 %155
  call void @llvm.hivm.vstsx1.v256i8(<256 x i8> %211, ptr addrspace(6) %212, i32 0, i32 12, i32 0, <256 x i1> %12)
  %213 = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 %165)
  %214 = extractvalue { <256 x i1>, i32 } %213, 0
  %215 = extractvalue { <256 x i1>, i32 } %213, 1
  %216 = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 %171)
  %217 = extractvalue { <256 x i1>, i32 } %216, 0
  %218 = extractvalue { <256 x i1>, i32 } %216, 1
  %219 = call <256 x i1> @llvm.hivm.pxor.z(<256 x i1> %214, <256 x i1> %217, <256 x i1> %12)
  %220 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %210, <64 x i32> %93, <256 x i1> %219)
  call void @llvm.hivm.mem.bar.vv.all()
  %221 = add i64 %28, 1
  br label %27

222:                                              ; preds = %27
  call void @llvm.hivm.mem.bar.vv.all()
  call void @llvm.hivm.vstsx1.v64i32(<64 x i32> %29, ptr addrspace(6) %7, i32 0, i32 2, i32 0, <256 x i1> %12)
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %222
  %223 = add i64 %9, 1
  br label %8, !llvm.loop !1

224:                                              ; preds = %8
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
