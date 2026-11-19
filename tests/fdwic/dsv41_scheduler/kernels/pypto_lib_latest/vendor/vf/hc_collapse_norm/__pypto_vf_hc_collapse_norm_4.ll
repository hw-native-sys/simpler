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

define void @__pypto_vf_hc_collapse_norm_4(ptr addrspace(6) %0, i64 %1) #0 {
  %3 = getelementptr float, ptr addrspace(6) %0, i64 30720
  %4 = getelementptr float, ptr addrspace(6) %0, i64 30752
  br label %5

5:                                                ; preds = %aivscope.latch, %2
  %6 = phi i64 [ %63, %aivscope.latch ], [ 0, %2 ]
  %7 = icmp slt i64 %6, 1
  br i1 %7, label %8, label %64

8:                                                ; preds = %5
  %9 = mul i64 %1, 4
  %10 = add i64 %9, 1
  %11 = add i64 %9, 2
  %12 = add i64 %9, 3
  %13 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %14 = mul i64 %9, 4
  %15 = getelementptr i8, ptr addrspace(6) %3, i64 %14
  %16 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %15, i32 0, i32 3, i32 0)
  %17 = mul i64 %10, 4
  %18 = getelementptr i8, ptr addrspace(6) %3, i64 %17
  %19 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %18, i32 0, i32 3, i32 0)
  %20 = mul i64 %11, 4
  %21 = getelementptr i8, ptr addrspace(6) %3, i64 %20
  %22 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %21, i32 0, i32 3, i32 0)
  %23 = mul i64 %12, 4
  %24 = getelementptr i8, ptr addrspace(6) %3, i64 %23
  %25 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %24, i32 0, i32 3, i32 0)
  call void @llvm.hivm.mem.bar.vld.vst()
  %26 = getelementptr float, ptr addrspace(6) %4, i64 0
  %27 = getelementptr float, ptr addrspace(6) %4, i64 5120
  %28 = getelementptr float, ptr addrspace(6) %4, i64 10240
  %29 = getelementptr float, ptr addrspace(6) %4, i64 15360
  %30 = getelementptr bfloat, ptr addrspace(6) %0, i64 0
  br label %31

31:                                               ; preds = %39, %8
  %32 = phi i16 [ %61, %39 ], [ 0, %8 ]
  %33 = phi ptr addrspace(6) [ %42, %39 ], [ %26, %8 ]
  %34 = phi ptr addrspace(6) [ %45, %39 ], [ %27, %8 ]
  %35 = phi ptr addrspace(6) [ %48, %39 ], [ %28, %8 ]
  %36 = phi ptr addrspace(6) [ %51, %39 ], [ %29, %8 ]
  %37 = phi ptr addrspace(6) [ %60, %39 ], [ %30, %8 ]
  %38 = icmp slt i16 %32, 80
  br i1 %38, label %39, label %62

39:                                               ; preds = %31
  %40 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %33, i32 256, i32 0, i32 1)
  %41 = extractvalue { <64 x float>, ptr addrspace(6) } %40, 0
  %42 = extractvalue { <64 x float>, ptr addrspace(6) } %40, 1
  %43 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %34, i32 256, i32 0, i32 1)
  %44 = extractvalue { <64 x float>, ptr addrspace(6) } %43, 0
  %45 = extractvalue { <64 x float>, ptr addrspace(6) } %43, 1
  %46 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %35, i32 256, i32 0, i32 1)
  %47 = extractvalue { <64 x float>, ptr addrspace(6) } %46, 0
  %48 = extractvalue { <64 x float>, ptr addrspace(6) } %46, 1
  %49 = call { <64 x float>, ptr addrspace(6) } @llvm.hivm.vldsx1.post.v64f32(ptr addrspace(6) %36, i32 256, i32 0, i32 1)
  %50 = extractvalue { <64 x float>, ptr addrspace(6) } %49, 0
  %51 = extractvalue { <64 x float>, ptr addrspace(6) } %49, 1
  %52 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %41, <64 x float> %16, <256 x i1> %13)
  %53 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %44, <64 x float> %19, <256 x i1> %13)
  %54 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %47, <64 x float> %22, <256 x i1> %13)
  %55 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %50, <64 x float> %25, <256 x i1> %13)
  %56 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %52, <64 x float> %53, <256 x i1> %13)
  %57 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %56, <64 x float> %54, <256 x i1> %13)
  %58 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %57, <64 x float> %55, <256 x i1> %13)
  %59 = call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %58, <256 x i1> %13, i32 0, i32 0, i32 0)
  %60 = call ptr addrspace(6) @llvm.hivm.vstsx1.post.v128bf16(<128 x bfloat> %59, ptr addrspace(6) %37, i32 128, i32 7, i32 1, <256 x i1> %13)
  %61 = add i16 %32, 1
  br label %31

62:                                               ; preds = %31
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %62
  %63 = add i64 %6, 1
  br label %5, !llvm.loop !1

64:                                               ; preds = %5
  ret void
}

attributes #0 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
