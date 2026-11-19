; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x i32> @llvm.hivm.vci.v64i32(i32, i32)

declare { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32)

declare <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x i32> @llvm.hivm.vdups.z.v64i32(i32, <256 x i1>, i32)

declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32)

declare <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vshls.u.x.v64i32(<64 x i32>, i16, <256 x i1>)

declare <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6), <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vxor.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vadd.u.x.v64i32(<64 x i32>, <64 x i32>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsel.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32>, i32, <256 x i1>)

declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float>, float, <256 x i1>)

declare <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare void @llvm.hivm.mem.bar.vld.vst()

; Function Attrs: nounwind writeonly argmemonly
declare void @llvm.hivm.vscatter.v64f32.v300(<64 x float>, ptr addrspace(6), <64 x i32>, <256 x i1>) #0

declare void @llvm.hivm.mem.bar.vst.vst()

declare <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32>, <64 x i32>)

declare <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

declare <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>)

define void @__pypto_vf_mhc_sinkhorn_0(ptr addrspace(6) %0, i64 %1, float %2, float %3, float %4) #1 {
  %6 = getelementptr float, ptr addrspace(6) %0, i64 192
  %7 = getelementptr float, ptr addrspace(6) %0, i64 448
  %8 = getelementptr float, ptr addrspace(6) %0, i64 640
  %9 = getelementptr float, ptr addrspace(6) %0, i64 832
  %10 = getelementptr float, ptr addrspace(6) %0, i64 224
  %11 = getelementptr float, ptr addrspace(6) %0, i64 256
  %12 = getelementptr float, ptr addrspace(6) %0, i64 304
  %13 = getelementptr float, ptr addrspace(6) %0, i64 352
  br label %14

14:                                               ; preds = %aivscope.latch, %5
  %15 = phi i64 [ %314, %aivscope.latch ], [ 0, %5 ]
  %16 = icmp slt i64 %15, 1
  br i1 %16, label %17, label %315

17:                                               ; preds = %14
  %18 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %19 = call <64 x i32> @llvm.hivm.vci.v64i32(i32 0, i32 0)
  %20 = mul i64 %1, 4
  %21 = trunc i64 %20 to i32
  %22 = icmp sgt i32 %21, 0
  %23 = select i1 %22, i32 %21, i32 0
  %24 = icmp ult i32 %23, 64
  %25 = select i1 %24, i32 %23, i32 64
  %26 = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 %25)
  %27 = extractvalue { <256 x i1>, i32 } %26, 0
  %28 = extractvalue { <256 x i1>, i32 } %26, 1
  %29 = call <64 x i32> @llvm.hivm.vshrs.u.x.v64i32(<64 x i32> %19, i16 2, <256 x i1> %18)
  %30 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 3, <256 x i1> %18, i32 1)
  %31 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %18, i32 1)
  %32 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float %2, <256 x i1> %18, i32 1)
  %33 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float %3, <256 x i1> %18, i32 1)
  %34 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float %4, <256 x i1> %18, i32 1)
  %35 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 1.000000e+00, <256 x i1> %18, i32 1)
  %36 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 1, <256 x i1> %18, i32 1)
  %37 = call <64 x i32> @llvm.hivm.vdups.z.v64i32(i32 2, <256 x i1> %18, i32 1)
  %38 = call <64 x i32> @llvm.hivm.vand.x.v64i32(<64 x i32> %19, <64 x i32> %30, <256 x i1> %18)
  %39 = call <64 x i32> @llvm.hivm.vshls.u.x.v64i32(<64 x i32> %29, i16 5, <256 x i1> %18)
  %40 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %6, <64 x i32> %29, <256 x i1> %27)
  %41 = call <64 x i32> @llvm.hivm.vshls.u.x.v64i32(<64 x i32> %29, i16 4, <256 x i1> %18)
  %42 = call <64 x i32> @llvm.hivm.vshls.u.x.v64i32(<64 x i32> %29, i16 3, <256 x i1> %18)
  %43 = call <64 x i32> @llvm.hivm.vxor.x.v64i32(<64 x i32> %19, <64 x i32> %36, <256 x i1> %18)
  %44 = call <64 x i32> @llvm.hivm.vxor.x.v64i32(<64 x i32> %19, <64 x i32> %37, <256 x i1> %18)
  %45 = call <64 x i32> @llvm.hivm.vadd.u.x.v64i32(<64 x i32> %39, <64 x i32> %38, <256 x i1> %18)
  %46 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %40, <64 x float> %31, <256 x i1> %27)
  %47 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %10, <64 x i32> %38, <256 x i1> %27)
  %48 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %38, i32 4, <256 x i1> %18)
  %49 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %38, i32 8, <256 x i1> %18)
  %50 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %38, i32 12, <256 x i1> %18)
  %51 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %10, <64 x i32> %48, <256 x i1> %27)
  %52 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %10, <64 x i32> %49, <256 x i1> %27)
  %53 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %10, <64 x i32> %50, <256 x i1> %27)
  %54 = call <64 x i32> @llvm.hivm.vadd.u.x.v64i32(<64 x i32> %42, <64 x i32> %38, <256 x i1> %18)
  %55 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %7, <64 x i32> %45, <256 x i1> %27)
  %56 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %8, <64 x i32> %45, <256 x i1> %27)
  %57 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %9, <64 x i32> %45, <256 x i1> %27)
  %58 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %47, <64 x float> %31, <256 x i1> %27)
  %59 = call <64 x i32> @llvm.hivm.vadd.u.x.v64i32(<64 x i32> %41, <64 x i32> %38, <256 x i1> %18)
  %60 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %55, <64 x float> %31, <256 x i1> %27)
  %61 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %56, <64 x float> %31, <256 x i1> %27)
  %62 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %57, <64 x float> %31, <256 x i1> %27)
  %63 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %51, <64 x float> %31, <256 x i1> %27)
  %64 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %52, <64 x float> %31, <256 x i1> %27)
  %65 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %53, <64 x float> %31, <256 x i1> %27)
  %66 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %60, <64 x float> %61, <256 x i1> %27)
  %67 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %66, <64 x float> %62, <256 x i1> %27)
  %68 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %0, <64 x i32> %45, <256 x i1> %27)
  %69 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %68, <64 x float> %31, <256 x i1> %27)
  %70 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %67, <64 x float> %69, <256 x i1> %27)
  %71 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %45, i32 4, <256 x i1> %18)
  %72 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %70, <64 x float> %46, <256 x i1> %27)
  %73 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %72, <64 x float> %32, <256 x i1> %27)
  %74 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %73, <64 x float> %58, <256 x i1> %27)
  %75 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %74, float -1.000000e+00, <256 x i1> %27)
  %76 = call <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float> %75, <256 x i1> %27)
  %77 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %76, float 1.000000e+00, <256 x i1> %27)
  %78 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %35, <64 x float> %77, <256 x i1> %27)
  %79 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %78, float 0x3EB0C6F7A0000000, <256 x i1> %27)
  %80 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %45, i32 8, <256 x i1> %18)
  %81 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %45, i32 12, <256 x i1> %18)
  %82 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %45, i32 16, <256 x i1> %18)
  %83 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %45, i32 20, <256 x i1> %18)
  %84 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %7, <64 x i32> %71, <256 x i1> %27)
  %85 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %84, <64 x float> %31, <256 x i1> %27)
  %86 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %8, <64 x i32> %71, <256 x i1> %27)
  %87 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %86, <64 x float> %31, <256 x i1> %27)
  %88 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %85, <64 x float> %87, <256 x i1> %27)
  %89 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %9, <64 x i32> %71, <256 x i1> %27)
  %90 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %0, <64 x i32> %71, <256 x i1> %27)
  %91 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %7, <64 x i32> %80, <256 x i1> %27)
  %92 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %8, <64 x i32> %80, <256 x i1> %27)
  %93 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %92, <64 x float> %31, <256 x i1> %27)
  %94 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %91, <64 x float> %31, <256 x i1> %27)
  %95 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %89, <64 x float> %31, <256 x i1> %27)
  %96 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %90, <64 x float> %31, <256 x i1> %27)
  %97 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %88, <64 x float> %95, <256 x i1> %27)
  %98 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %94, <64 x float> %93, <256 x i1> %27)
  %99 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %97, <64 x float> %96, <256 x i1> %27)
  %100 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %99, <64 x float> %46, <256 x i1> %27)
  %101 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %100, <64 x float> %33, <256 x i1> %27)
  %102 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %101, <64 x float> %63, <256 x i1> %27)
  %103 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %9, <64 x i32> %80, <256 x i1> %27)
  %104 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %0, <64 x i32> %80, <256 x i1> %27)
  %105 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %7, <64 x i32> %81, <256 x i1> %27)
  %106 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %8, <64 x i32> %81, <256 x i1> %27)
  %107 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %106, <64 x float> %31, <256 x i1> %27)
  %108 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %105, <64 x float> %31, <256 x i1> %27)
  %109 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %108, <64 x float> %107, <256 x i1> %27)
  %110 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %9, <64 x i32> %81, <256 x i1> %27)
  %111 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %110, <64 x float> %31, <256 x i1> %27)
  %112 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %109, <64 x float> %111, <256 x i1> %27)
  %113 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %0, <64 x i32> %81, <256 x i1> %27)
  %114 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %7, <64 x i32> %82, <256 x i1> %27)
  %115 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %8, <64 x i32> %82, <256 x i1> %27)
  %116 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %103, <64 x float> %31, <256 x i1> %27)
  %117 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %104, <64 x float> %31, <256 x i1> %27)
  %118 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %98, <64 x float> %116, <256 x i1> %27)
  %119 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %114, <64 x float> %31, <256 x i1> %27)
  %120 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %115, <64 x float> %31, <256 x i1> %27)
  %121 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %113, <64 x float> %31, <256 x i1> %27)
  %122 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %119, <64 x float> %120, <256 x i1> %27)
  %123 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %118, <64 x float> %117, <256 x i1> %27)
  %124 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %112, <64 x float> %121, <256 x i1> %27)
  %125 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %123, <64 x float> %46, <256 x i1> %27)
  %126 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %124, <64 x float> %46, <256 x i1> %27)
  %127 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %125, <64 x float> %34, <256 x i1> %27)
  %128 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %126, <64 x float> %34, <256 x i1> %27)
  %129 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %127, <64 x float> %64, <256 x i1> %27)
  %130 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %128, <64 x float> %65, <256 x i1> %27)
  %131 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %9, <64 x i32> %82, <256 x i1> %27)
  %132 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %0, <64 x i32> %82, <256 x i1> %27)
  %133 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %7, <64 x i32> %83, <256 x i1> %27)
  %134 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %8, <64 x i32> %83, <256 x i1> %27)
  %135 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %9, <64 x i32> %83, <256 x i1> %27)
  %136 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %0, <64 x i32> %83, <256 x i1> %27)
  %137 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %133, <64 x float> %31, <256 x i1> %27)
  %138 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %134, <64 x float> %31, <256 x i1> %27)
  %139 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %131, <64 x float> %31, <256 x i1> %27)
  %140 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %135, <64 x float> %31, <256 x i1> %27)
  %141 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %132, <64 x float> %31, <256 x i1> %27)
  %142 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %136, <64 x float> %31, <256 x i1> %27)
  %143 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %137, <64 x float> %138, <256 x i1> %27)
  %144 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %122, <64 x float> %139, <256 x i1> %27)
  %145 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %143, <64 x float> %140, <256 x i1> %27)
  %146 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %144, <64 x float> %141, <256 x i1> %27)
  %147 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %145, <64 x float> %142, <256 x i1> %27)
  %148 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %38, i32 16, <256 x i1> %18)
  %149 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %38, i32 20, <256 x i1> %18)
  %150 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %146, <64 x float> %46, <256 x i1> %27)
  %151 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %147, <64 x float> %46, <256 x i1> %27)
  %152 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %10, <64 x i32> %148, <256 x i1> %27)
  %153 = call <64 x float> @llvm.hivm.vgather2.bc.v64f32(ptr addrspace(6) %10, <64 x i32> %149, <256 x i1> %27)
  %154 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %150, <64 x float> %34, <256 x i1> %27)
  %155 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %59, i32 4, <256 x i1> %18)
  %156 = call <64 x float> @llvm.hivm.vmul.s.x.v64f32(<64 x float> %151, <64 x float> %34, <256 x i1> %27)
  %157 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %152, <64 x float> %31, <256 x i1> %27)
  %158 = call <64 x float> @llvm.hivm.vsel.v64f32(<64 x float> %153, <64 x float> %31, <256 x i1> %27)
  %159 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %59, i32 8, <256 x i1> %18)
  %160 = call <64 x i32> @llvm.hivm.vadds.u.x.v64i32(<64 x i32> %59, i32 12, <256 x i1> %18)
  %161 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %154, <64 x float> %157, <256 x i1> %27)
  %162 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %156, <64 x float> %158, <256 x i1> %27)
  call void @llvm.hivm.mem.bar.vld.vst()
  %163 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %102, float -1.000000e+00, <256 x i1> %27)
  call void @llvm.hivm.vscatter.v64f32.v300(<64 x float> %79, ptr addrspace(6) %11, <64 x i32> %54, <256 x i1> %27)
  %164 = call <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float> %163, <256 x i1> %27)
  %165 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %164, float 1.000000e+00, <256 x i1> %27)
  %166 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %35, <64 x float> %165, <256 x i1> %27)
  %167 = call <64 x float> @llvm.hivm.vmuls.s.x.v64f32(<64 x float> %166, float 2.000000e+00, <256 x i1> %27)
  call void @llvm.hivm.mem.bar.vst.vst()
  %168 = bitcast <64 x float> %129 to <64 x i32>
  %169 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %168, <64 x i32> %43)
  %170 = bitcast <64 x i32> %169 to <64 x float>
  %171 = bitcast <64 x float> %130 to <64 x i32>
  %172 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %171, <64 x i32> %43)
  %173 = bitcast <64 x i32> %172 to <64 x float>
  %174 = bitcast <64 x float> %161 to <64 x i32>
  %175 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %174, <64 x i32> %43)
  %176 = bitcast <64 x i32> %175 to <64 x float>
  %177 = bitcast <64 x float> %162 to <64 x i32>
  %178 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %177, <64 x i32> %43)
  %179 = bitcast <64 x i32> %178 to <64 x float>
  call void @llvm.hivm.vscatter.v64f32.v300(<64 x float> %167, ptr addrspace(6) %12, <64 x i32> %54, <256 x i1> %27)
  %180 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %129, <64 x float> %170, <256 x i1> %18)
  %181 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %130, <64 x float> %173, <256 x i1> %18)
  %182 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %161, <64 x float> %176, <256 x i1> %18)
  %183 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %162, <64 x float> %179, <256 x i1> %18)
  %184 = bitcast <64 x float> %180 to <64 x i32>
  %185 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %184, <64 x i32> %44)
  %186 = bitcast <64 x i32> %185 to <64 x float>
  %187 = bitcast <64 x float> %181 to <64 x i32>
  %188 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %187, <64 x i32> %44)
  %189 = bitcast <64 x i32> %188 to <64 x float>
  %190 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %180, <64 x float> %186, <256 x i1> %18)
  %191 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %181, <64 x float> %189, <256 x i1> %18)
  %192 = bitcast <64 x float> %182 to <64 x i32>
  %193 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %192, <64 x i32> %44)
  %194 = bitcast <64 x i32> %193 to <64 x float>
  %195 = bitcast <64 x float> %183 to <64 x i32>
  %196 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %195, <64 x i32> %44)
  %197 = bitcast <64 x i32> %196 to <64 x float>
  %198 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %182, <64 x float> %194, <256 x i1> %18)
  %199 = call <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float> %129, <64 x float> %190, <256 x i1> %27)
  %200 = call <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float> %130, <64 x float> %191, <256 x i1> %27)
  %201 = call <64 x float> @llvm.hivm.vmax.s.x.v64f32(<64 x float> %183, <64 x float> %197, <256 x i1> %18)
  %202 = call <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float> %161, <64 x float> %198, <256 x i1> %27)
  %203 = call <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float> %199, <256 x i1> %27)
  %204 = call <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float> %200, <256 x i1> %27)
  %205 = call <64 x float> @llvm.hivm.vsub.s.x.v64f32(<64 x float> %162, <64 x float> %201, <256 x i1> %27)
  %206 = call <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float> %202, <256 x i1> %27)
  %207 = bitcast <64 x float> %203 to <64 x i32>
  %208 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %207, <64 x i32> %43)
  %209 = bitcast <64 x i32> %208 to <64 x float>
  %210 = bitcast <64 x float> %204 to <64 x i32>
  %211 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %210, <64 x i32> %43)
  %212 = bitcast <64 x i32> %211 to <64 x float>
  %213 = call <64 x float> @llvm.hivm.vexp.x.v64f32(<64 x float> %205, <256 x i1> %27)
  %214 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %203, <64 x float> %209, <256 x i1> %18)
  %215 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %204, <64 x float> %212, <256 x i1> %18)
  %216 = bitcast <64 x float> %206 to <64 x i32>
  %217 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %216, <64 x i32> %43)
  %218 = bitcast <64 x i32> %217 to <64 x float>
  %219 = bitcast <64 x float> %213 to <64 x i32>
  %220 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %219, <64 x i32> %43)
  %221 = bitcast <64 x i32> %220 to <64 x float>
  %222 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %206, <64 x float> %218, <256 x i1> %18)
  %223 = bitcast <64 x float> %214 to <64 x i32>
  %224 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %223, <64 x i32> %44)
  %225 = bitcast <64 x i32> %224 to <64 x float>
  %226 = bitcast <64 x float> %215 to <64 x i32>
  %227 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %226, <64 x i32> %44)
  %228 = bitcast <64 x i32> %227 to <64 x float>
  %229 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %213, <64 x float> %221, <256 x i1> %18)
  %230 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %214, <64 x float> %225, <256 x i1> %18)
  %231 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %215, <64 x float> %228, <256 x i1> %18)
  %232 = bitcast <64 x float> %222 to <64 x i32>
  %233 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %232, <64 x i32> %44)
  %234 = bitcast <64 x i32> %233 to <64 x float>
  %235 = bitcast <64 x float> %229 to <64 x i32>
  %236 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %235, <64 x i32> %44)
  %237 = bitcast <64 x i32> %236 to <64 x float>
  %238 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %203, <64 x float> %230, <256 x i1> %27)
  %239 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %204, <64 x float> %231, <256 x i1> %27)
  %240 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %222, <64 x float> %234, <256 x i1> %18)
  %241 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %229, <64 x float> %237, <256 x i1> %18)
  %242 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %206, <64 x float> %240, <256 x i1> %27)
  %243 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %238, float 0x3EB0C6F7A0000000, <256 x i1> %27)
  %244 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %239, float 0x3EB0C6F7A0000000, <256 x i1> %27)
  %245 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %213, <64 x float> %241, <256 x i1> %27)
  %246 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %242, float 0x3EB0C6F7A0000000, <256 x i1> %27)
  %247 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %243, <64 x float> %244, <256 x i1> %27)
  %248 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %245, float 0x3EB0C6F7A0000000, <256 x i1> %27)
  %249 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %247, <64 x float> %246, <256 x i1> %27)
  %250 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %249, <64 x float> %248, <256 x i1> %27)
  %251 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %250, float 0x3EB0C6F7A0000000, <256 x i1> %27)
  %252 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %243, <64 x float> %251, <256 x i1> %27)
  %253 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %244, <64 x float> %251, <256 x i1> %27)
  %254 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %246, <64 x float> %251, <256 x i1> %27)
  %255 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %248, <64 x float> %251, <256 x i1> %27)
  br label %256

256:                                              ; preds = %263, %17
  %257 = phi i16 [ %312, %263 ], [ 0, %17 ]
  %258 = phi <64 x float> [ %308, %263 ], [ %252, %17 ]
  %259 = phi <64 x float> [ %309, %263 ], [ %253, %17 ]
  %260 = phi <64 x float> [ %310, %263 ], [ %254, %17 ]
  %261 = phi <64 x float> [ %311, %263 ], [ %255, %17 ]
  %262 = icmp slt i16 %257, 19
  br i1 %262, label %263, label %313

263:                                              ; preds = %256
  %264 = bitcast <64 x float> %258 to <64 x i32>
  %265 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %264, <64 x i32> %43)
  %266 = bitcast <64 x i32> %265 to <64 x float>
  %267 = bitcast <64 x float> %259 to <64 x i32>
  %268 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %267, <64 x i32> %43)
  %269 = bitcast <64 x i32> %268 to <64 x float>
  %270 = bitcast <64 x float> %260 to <64 x i32>
  %271 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %270, <64 x i32> %43)
  %272 = bitcast <64 x i32> %271 to <64 x float>
  %273 = bitcast <64 x float> %261 to <64 x i32>
  %274 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %273, <64 x i32> %43)
  %275 = bitcast <64 x i32> %274 to <64 x float>
  %276 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %258, <64 x float> %266, <256 x i1> %18)
  %277 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %259, <64 x float> %269, <256 x i1> %18)
  %278 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %260, <64 x float> %272, <256 x i1> %18)
  %279 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %261, <64 x float> %275, <256 x i1> %18)
  %280 = bitcast <64 x float> %276 to <64 x i32>
  %281 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %280, <64 x i32> %44)
  %282 = bitcast <64 x i32> %281 to <64 x float>
  %283 = bitcast <64 x float> %277 to <64 x i32>
  %284 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %283, <64 x i32> %44)
  %285 = bitcast <64 x i32> %284 to <64 x float>
  %286 = bitcast <64 x float> %278 to <64 x i32>
  %287 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %286, <64 x i32> %44)
  %288 = bitcast <64 x i32> %287 to <64 x float>
  %289 = bitcast <64 x float> %279 to <64 x i32>
  %290 = call <64 x i32> @llvm.hivm.vselr.v64f32(<64 x i32> %289, <64 x i32> %44)
  %291 = bitcast <64 x i32> %290 to <64 x float>
  %292 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %276, <64 x float> %282, <256 x i1> %18)
  %293 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %277, <64 x float> %285, <256 x i1> %18)
  %294 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %278, <64 x float> %288, <256 x i1> %18)
  %295 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %279, <64 x float> %291, <256 x i1> %18)
  %296 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %292, float 0x3EB0C6F7A0000000, <256 x i1> %27)
  %297 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %293, float 0x3EB0C6F7A0000000, <256 x i1> %27)
  %298 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %294, float 0x3EB0C6F7A0000000, <256 x i1> %27)
  %299 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %295, float 0x3EB0C6F7A0000000, <256 x i1> %27)
  %300 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %258, <64 x float> %296, <256 x i1> %27)
  %301 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %259, <64 x float> %297, <256 x i1> %27)
  %302 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %260, <64 x float> %298, <256 x i1> %27)
  %303 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %261, <64 x float> %299, <256 x i1> %27)
  %304 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %300, <64 x float> %301, <256 x i1> %27)
  %305 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %304, <64 x float> %302, <256 x i1> %27)
  %306 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %305, <64 x float> %303, <256 x i1> %27)
  %307 = call <64 x float> @llvm.hivm.vadds.s.x.v64f32(<64 x float> %306, float 0x3EB0C6F7A0000000, <256 x i1> %27)
  %308 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %300, <64 x float> %307, <256 x i1> %27)
  %309 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %301, <64 x float> %307, <256 x i1> %27)
  %310 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %302, <64 x float> %307, <256 x i1> %27)
  %311 = call <64 x float> @llvm.hivm.vdiv.s.x.v64f32(<64 x float> %303, <64 x float> %307, <256 x i1> %27)
  %312 = add i16 %257, 1
  br label %256

313:                                              ; preds = %256
  call void @llvm.hivm.mem.bar.vst.vst()
  call void @llvm.hivm.vscatter.v64f32.v300(<64 x float> %258, ptr addrspace(6) %13, <64 x i32> %59, <256 x i1> %27)
  call void @llvm.hivm.mem.bar.vst.vst()
  call void @llvm.hivm.vscatter.v64f32.v300(<64 x float> %259, ptr addrspace(6) %13, <64 x i32> %155, <256 x i1> %27)
  call void @llvm.hivm.mem.bar.vst.vst()
  call void @llvm.hivm.vscatter.v64f32.v300(<64 x float> %260, ptr addrspace(6) %13, <64 x i32> %159, <256 x i1> %27)
  call void @llvm.hivm.mem.bar.vst.vst()
  call void @llvm.hivm.vscatter.v64f32.v300(<64 x float> %261, ptr addrspace(6) %13, <64 x i32> %160, <256 x i1> %27)
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %313
  %314 = add i64 %15, 1
  br label %14, !llvm.loop !1

315:                                              ; preds = %14
  ret void
}

attributes #0 = { nounwind writeonly argmemonly }
attributes #1 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.aivector_scope"}
