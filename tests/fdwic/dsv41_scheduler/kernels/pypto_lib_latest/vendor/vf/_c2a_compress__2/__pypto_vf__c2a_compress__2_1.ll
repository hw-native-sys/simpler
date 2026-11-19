; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x i32> @llvm.hivm.vci.v64i32(i32, i32)

declare <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6), i32, i32, i32)

declare <256 x i1> @llvm.hivm.pset.b16(i32)

declare <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat>, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6), <64 x i32>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsel.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare { <64 x float>, <64 x float> } @llvm.hivm.vdintlv.v64f32(<64 x float>, <64 x float>)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare { <64 x float>, <64 x float> } @llvm.hivm.vintlv.v64f32(<64 x float>, <64 x float>)

declare <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float>, <256 x i1>, i32, i32, i32)

declare void @llvm.hivm.mem.bar.vld.vst()

declare void @llvm.hivm.vstsx1.v128bf16(<128 x bfloat>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare void @llvm.hivm.mem.bar.vst.vld()

declare <128 x i16> @llvm.hivm.vci.v128i16(i16, i32)

declare <128 x i16> @llvm.hivm.vdups.z.v128i16(i16, <256 x i1>, i32)

declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32)

declare <128 x i16> @llvm.hivm.vshrs.u.x.v128i16(<128 x i16>, i16, <256 x i1>)

declare { <128 x bfloat>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v128bf16(ptr addrspace(6), i32, i32, i32)

declare <128 x i16> @llvm.hivm.vand.x.v128i16(<128 x i16>, <128 x i16>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vzunpack.v64i32(<128 x i16>, i32)

declare <128 x i16> @llvm.hivm.vcgmax.u.x.v128i16(<128 x i16>, <256 x i1>)

declare <128 x i16> @llvm.hivm.vselr.v128i16(<128 x i16>, <128 x i16>)

declare <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare ptr addrspace(6) @llvm.hivm.vstsx1.post.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare { <64 x i32>, <64 x i32> } @llvm.hivm.vdintlv.v64i32(<64 x i32>, <64 x i32>)

declare <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare ptr addrspace(6) @llvm.hivm.vstsx1.post.v256i8(<256 x i8>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare <64 x i32> @llvm.hivm.vshls.u.x.v64i32(<64 x i32>, i16, <256 x i1>)

define void @__pypto_vf__c2a_compress__2_1(ptr addrspace(6) %0) #0 {
  %2 = getelementptr bfloat, ptr addrspace(6) %0, i64 7680
  %3 = getelementptr float, ptr addrspace(6) %0, i64 1024
  %4 = getelementptr float, ptr addrspace(6) %0, i64 1536
  %5 = getelementptr i8, ptr addrspace(6) %0, i64 2048
  %6 = getelementptr bfloat, ptr addrspace(6) %0, i64 4096
  br label %7

7:                                                ; preds = %aivscope.latch, %1
  %8 = phi i64 [ %115, %aivscope.latch ], [ 0, %1 ]
  %9 = icmp slt i64 %8, 1
  br i1 %9, label %10, label %116

10:                                               ; preds = %7
  %11 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %12 = call <256 x i1> @llvm.hivm.pset.b32(i32 7)
  %13 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %14 = call <128 x bfloat> @llvm.hivm.vldsx1.v128bf16(ptr addrspace(6) %2, i32 896, i32 14, i32 0)
  %15 = call <256 x i1> @llvm.hivm.pset.b16(i32 0)
  %16 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %14, <256 x i1> %15, i32 0)
  %17 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %11, i32 1)
  %18 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %3, <64 x i32> %13, <256 x i1> %12)
  %19 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %4, <64 x i32> %13, <256 x i1> %12)
  %20 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %18, <64 x float> %17, <256 x i1> %12)
  %21 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %19, <64 x float> %17, <256 x i1> %12)
  %22 = call { <64 x float>, <64 x float> } @llvm.hivm.vdintlv.v64f32(<64 x float> %16, <64 x float> %16)
  %23 = extractvalue { <64 x float>, <64 x float> } %22, 0
  %24 = extractvalue { <64 x float>, <64 x float> } %22, 1
  %25 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %23, <64 x float> %20, <256 x i1> %12)
  %26 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %23, <64 x float> %21, <256 x i1> %12)
  %27 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %24, <64 x float> %21, <256 x i1> %12)
  %28 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %24, <64 x float> %20, <256 x i1> %12)
  %29 = call <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float> %25, <64 x float> %27, <256 x i1> %12)
  %30 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %26, <64 x float> %28, <256 x i1> %12)
  %31 = call { <64 x float>, <64 x float> } @llvm.hivm.vintlv.v64f32(<64 x float> %29, <64 x float> %30)
  %32 = extractvalue { <64 x float>, <64 x float> } %31, 0
  %33 = extractvalue { <64 x float>, <64 x float> } %31, 1
  %34 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %32, <256 x i1> %11, i32 0, i32 0, i32 0)
  call void @llvm.hivm.mem.bar.vld.vst()
  call void @llvm.hivm.vstsx1.v128bf16(<128 x bfloat> %34, ptr addrspace(6) %2, i32 896, i32 7, i32 0, <256 x i1> %11)
  call void @llvm.hivm.mem.bar.vst.vld()
  %35 = call <128 x i16> @llvm.hivm.vci.v128i16(i16 0, i32 0)
  %36 = getelementptr bfloat, ptr addrspace(6) %2, i64 0
  %37 = getelementptr float, ptr addrspace(6) %0, i64 0
  %38 = getelementptr i8, ptr addrspace(6) %5, i64 0
  %39 = call <128 x i16> @llvm.hivm.vdups.z.v128i16(i16 32767, <256 x i1> %15, i32 1)
  %40 = call <256 x i1> @llvm.hivm.pset.b16(i32 8)
  %41 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 0, <256 x i1> %11, i32 1)
  %42 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 2.500000e-01, <256 x i1> %11, i32 1)
  %43 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 1, <256 x i1> %11, i32 1)
  %44 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 7.500000e-01, <256 x i1> %11, i32 1)
  %45 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 2, <256 x i1> %11, i32 1)
  %46 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.250000e+00, <256 x i1> %11, i32 1)
  %47 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 3, <256 x i1> %11, i32 1)
  %48 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.750000e+00, <256 x i1> %11, i32 1)
  %49 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 4, <256 x i1> %11, i32 1)
  %50 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 2.500000e+00, <256 x i1> %11, i32 1)
  %51 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 5, <256 x i1> %11, i32 1)
  %52 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 3.500000e+00, <256 x i1> %11, i32 1)
  %53 = call <128 x i16> @llvm.hivm.vshrs.u.x.v128i16(<128 x i16> %35, i16 5, <256 x i1> %15)
  %54 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 6, <256 x i1> %11, i32 1)
  %55 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 5.000000e+00, <256 x i1> %11, i32 1)
  %56 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 7, <256 x i1> %11, i32 1)
  %57 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 8, <256 x i1> %11, i32 1)
  br label %58

58:                                               ; preds = %64, %10
  %59 = phi i16 [ %109, %64 ], [ 0, %10 ]
  %60 = phi ptr addrspace(6) [ %67, %64 ], [ %36, %10 ]
  %61 = phi ptr addrspace(6) [ %82, %64 ], [ %37, %10 ]
  %62 = phi ptr addrspace(6) [ %108, %64 ], [ %38, %10 ]
  %63 = icmp slt i16 %59, 8
  br i1 %63, label %64, label %110

64:                                               ; preds = %58
  %65 = call { <128 x bfloat>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v128bf16(ptr addrspace(6) %60, i32 128, i32 0, i32 1)
  %66 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %65, 0
  %67 = extractvalue { <128 x bfloat>, ptr addrspace(6) } %65, 1
  %68 = bitcast <128 x bfloat> %66 to <128 x i16>
  %69 = call <128 x i16> @llvm.hivm.vand.x.v128i16(<128 x i16> %68, <128 x i16> %39, <256 x i1> %40)
  %70 = call <64 x i32> @llvm.hivm.vzunpack.v64i32(<128 x i16> %68, i32 0)
  %71 = call <128 x i16> @llvm.hivm.vcgmax.u.x.v128i16(<128 x i16> %69, <256 x i1> %40)
  %72 = bitcast <64 x i32> %70 to <128 x bfloat>
  %73 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %72, <256 x i1> %15, i32 0)
  %74 = call <128 x i16> @llvm.hivm.vselr.v128i16(<128 x i16> %71, <128 x i16> %53)
  %75 = bitcast <128 x i16> %74 to <128 x bfloat>
  %76 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %75, <256 x i1> %15, i32 0)
  %77 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %76, float 0x3FC5555560000000, <256 x i1> %11)
  %78 = call <64 x float> @llvm.hivm.vmaxs.s.x.v64f32(<64 x float> %77, float 0x3F60000000000000, <256 x i1> %11)
  %79 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %78, <256 x i1> %11, i32 0, i32 0, i32 0)
  %80 = call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %79, <256 x i1> %15, i32 0)
  %81 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %73, <64 x float> %80, <256 x i1> %11)
  %82 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v64f32(<64 x float> %80, ptr addrspace(6) %61, i32 256, i32 2, i32 1, <256 x i1> %11)
  %83 = bitcast <64 x float> %81 to <64 x i32>
  %84 = call <64 x float> @llvm.hivm.vabs.x.v64f32(<64 x float> %81, <256 x i1> %11)
  %85 = call <64 x i32> @llvm.hivm.vshrs.s.x.v64i32(<64 x i32> %83, i16 28, <256 x i1> %11)
  %86 = call <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32> %85, <64 x i32> %57, <256 x i1> %11)
  %87 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %84, <64 x float> %42, <256 x i1> %11)
  %88 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %43, <64 x i32> %41, <256 x i1> %87)
  %89 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %84, <64 x float> %44, <256 x i1> %11)
  %90 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %45, <64 x i32> %88, <256 x i1> %89)
  %91 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %84, <64 x float> %46, <256 x i1> %11)
  %92 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %47, <64 x i32> %90, <256 x i1> %91)
  %93 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %84, <64 x float> %48, <256 x i1> %11)
  %94 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %49, <64 x i32> %92, <256 x i1> %93)
  %95 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %84, <64 x float> %50, <256 x i1> %11)
  %96 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %51, <64 x i32> %94, <256 x i1> %95)
  %97 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %84, <64 x float> %52, <256 x i1> %11)
  %98 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %54, <64 x i32> %96, <256 x i1> %97)
  %99 = call <256 x i1> @llvm.hivm.vcmp.gt.s.z.v64f32(<64 x float> %84, <64 x float> %55, <256 x i1> %11)
  %100 = call <64 x i32> @llvm.hivm.vsel.v64i32(<64 x i32> %56, <64 x i32> %98, <256 x i1> %99)
  %101 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %100, <64 x i32> %86, <256 x i1> %11)
  %102 = call { <64 x i32>, <64 x i32> } @llvm.hivm.vdintlv.v64i32(<64 x i32> %101, <64 x i32> %101)
  %103 = extractvalue { <64 x i32>, <64 x i32> } %102, 0
  %104 = extractvalue { <64 x i32>, <64 x i32> } %102, 1
  %105 = call <64 x i32> @llvm.hivm.vshls.s.x.v64i32(<64 x i32> %104, i16 4, <256 x i1> %12)
  %106 = call <64 x i32> @llvm.hivm.vor.x.v64i32(<64 x i32> %103, <64 x i32> %105, <256 x i1> %12)
  %107 = bitcast <64 x i32> %106 to <256 x i8>
  %108 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v256i8(<256 x i8> %107, ptr addrspace(6) %62, i32 32, i32 12, i32 1, <256 x i1> %12)
  %109 = add i16 %59, 1
  br label %58

110:                                              ; preds = %58
  %111 = call <64 x i32> @llvm.hivm.vshls.u.x.v64i32(<64 x i32> %13, i16 4, <256 x i1> %11)
  call void @llvm.hivm.mem.bar.vst.vld()
  %112 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %0, <64 x i32> %111, <256 x i1> %12)
  %113 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %112, <64 x float> %17, <256 x i1> %12)
  %114 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %113, <256 x i1> %11, i32 0, i32 0, i32 0)
  call void @llvm.hivm.mem.bar.vld.vst()
  call void @llvm.hivm.vstsx1.v128bf16(<128 x bfloat> %114, ptr addrspace(6) %6, i32 0, i32 7, i32 0, <256 x i1> %12)
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %110
  %115 = add i64 %8, 1
  br label %7, !llvm.loop !1

116:                                              ; preds = %7
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
