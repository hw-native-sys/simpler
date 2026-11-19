; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6), i32, i32, i32)

declare void @llvm.hivm.mem.bar.vld.vst()

declare { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float>, <256 x i1>, i32, i32, i32)

declare ptr addrspace(6) @llvm.hivm.vstsx1.post.v128bf16(<128 x bfloat>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

define void @__pypto_vf_hc_collapse_norm_2(ptr addrspace(6) %0, i64 %1) #0 {
  %3 = getelementptr float, ptr addrspace(6) %0, i64 30720
  %4 = getelementptr bfloat, ptr addrspace(6) %0, i64 40960
  br label %5

5:                                                ; preds = %aivscope.latch, %2
  %6 = phi i64 [ %64, %aivscope.latch ], [ 0, %2 ]
  %7 = icmp slt i64 %6, 1
  br i1 %7, label %8, label %65

8:                                                ; preds = %5
  %9 = mul i64 %1, 4
  %10 = add i64 %9, 4
  %11 = add i64 %9, 5
  %12 = add i64 %9, 6
  %13 = add i64 %9, 7
  %14 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %15 = mul i64 %10, 4
  %16 = getelementptr i8, ptr addrspace(6) %3, i64 %15
  %17 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %16, i32 0, i32 3, i32 0)
  %18 = mul i64 %11, 4
  %19 = getelementptr i8, ptr addrspace(6) %3, i64 %18
  %20 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %19, i32 0, i32 3, i32 0)
  %21 = mul i64 %12, 4
  %22 = getelementptr i8, ptr addrspace(6) %3, i64 %21
  %23 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %22, i32 0, i32 3, i32 0)
  %24 = mul i64 %13, 4
  %25 = getelementptr i8, ptr addrspace(6) %3, i64 %24
  %26 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %25, i32 0, i32 3, i32 0)
  call void @llvm.hivm.mem.bar.vld.vst()
  %27 = getelementptr float, ptr addrspace(6) %0, i64 0
  %28 = getelementptr float, ptr addrspace(6) %0, i64 5120
  %29 = getelementptr float, ptr addrspace(6) %0, i64 10240
  %30 = getelementptr float, ptr addrspace(6) %0, i64 15360
  %31 = getelementptr bfloat, ptr addrspace(6) %4, i64 0
  br label %32

32:                                               ; preds = %40, %8
  %33 = phi i16 [ %62, %40 ], [ 0, %8 ]
  %34 = phi ptr addrspace(6) [ %43, %40 ], [ %27, %8 ]
  %35 = phi ptr addrspace(6) [ %46, %40 ], [ %28, %8 ]
  %36 = phi ptr addrspace(6) [ %49, %40 ], [ %29, %8 ]
  %37 = phi ptr addrspace(6) [ %52, %40 ], [ %30, %8 ]
  %38 = phi ptr addrspace(6) [ %61, %40 ], [ %31, %8 ]
  %39 = icmp slt i16 %33, 80
  br i1 %39, label %40, label %63

40:                                               ; preds = %32
  %41 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %34, i32 256, i32 0, i32 1)
  %42 = extractvalue { <64 x float>, ptr addrspace(6) } %41, 0
  %43 = extractvalue { <64 x float>, ptr addrspace(6) } %41, 1
  %44 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %35, i32 256, i32 0, i32 1)
  %45 = extractvalue { <64 x float>, ptr addrspace(6) } %44, 0
  %46 = extractvalue { <64 x float>, ptr addrspace(6) } %44, 1
  %47 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %36, i32 256, i32 0, i32 1)
  %48 = extractvalue { <64 x float>, ptr addrspace(6) } %47, 0
  %49 = extractvalue { <64 x float>, ptr addrspace(6) } %47, 1
  %50 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %37, i32 256, i32 0, i32 1)
  %51 = extractvalue { <64 x float>, ptr addrspace(6) } %50, 0
  %52 = extractvalue { <64 x float>, ptr addrspace(6) } %50, 1
  %53 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %42, <64 x float> %17, <256 x i1> %14)
  %54 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %45, <64 x float> %20, <256 x i1> %14)
  %55 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %48, <64 x float> %23, <256 x i1> %14)
  %56 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %51, <64 x float> %26, <256 x i1> %14)
  %57 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %53, <64 x float> %54, <256 x i1> %14)
  %58 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %57, <64 x float> %55, <256 x i1> %14)
  %59 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %58, <64 x float> %56, <256 x i1> %14)
  %60 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %59, <256 x i1> %14, i32 0, i32 0, i32 0)
  %61 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v128bf16(<128 x bfloat> %60, ptr addrspace(6) %38, i32 128, i32 7, i32 1, <256 x i1> %14)
  %62 = add i16 %33, 1
  br label %32

63:                                               ; preds = %32
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %63
  %64 = add i64 %6, 1
  br label %5, !llvm.loop !1

65:                                               ; preds = %5
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
