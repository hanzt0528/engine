; ModuleID = 'cuda_sgemm_shared3.cpp'
source_filename = "cuda_sgemm_shared3.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", %"class.std::basic_ostream"*, i8, i8, %"class.std::basic_streambuf"*, %"class.std::ctype"*, %"class.std::num_put"*, %"class.std::num_get"* }
%"class.std::ios_base" = type { i32 (...)**, i64, i64, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"class.std::locale" }
%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"class.std::ios_base"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words" = type { i8*, i64 }
%"class.std::locale" = type { %"class.std::locale::_Impl"* }
%"class.std::locale::_Impl" = type { i32, %"class.std::locale::facet"**, i64, %"class.std::locale::facet"**, i8** }
%"class.std::locale::facet" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.std::basic_streambuf" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"class.std::locale" }
%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], %struct.__locale_struct*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct = type { [13 x %struct.__locale_data*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data = type opaque
%"class.std::num_put" = type { %"class.std::locale::facet.base", [4 x i8] }
%"class.std::num_get" = type { %"class.std::locale::facet.base", [4 x i8] }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_cuda_sgemm_shared3.cpp, i8* null }]

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #0

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: norecurse nounwind uwtable
define void @_Z9cpu_sgemmPfS_S_iii(float* nocapture readonly, float* nocapture readonly, float* nocapture, i32, i32, i32) local_unnamed_addr #3 {
  %7 = icmp sgt i32 %3, 0
  br i1 %7, label %8, label %85

; <label>:8:                                      ; preds = %6
  %9 = icmp sgt i32 %4, 0
  %10 = icmp sgt i32 %5, 0
  br i1 %9, label %11, label %85

; <label>:11:                                     ; preds = %8
  %12 = zext i32 %4 to i64
  %13 = shl nuw nsw i64 %12, 2
  %14 = sext i32 %4 to i64
  %15 = sext i32 %5 to i64
  %16 = zext i32 %5 to i64
  %17 = zext i32 %3 to i64
  %18 = and i64 %16, 1
  %19 = icmp eq i32 %5, 1
  %20 = sub nsw i64 %16, %18
  %21 = icmp eq i64 %18, 0
  br label %22

; <label>:22:                                     ; preds = %27, %11
  %23 = phi i64 [ %28, %27 ], [ 0, %11 ]
  %24 = mul nsw i64 %23, %14
  %25 = mul nsw i64 %23, %15
  br i1 %10, label %26, label %79

; <label>:26:                                     ; preds = %22
  br label %30

; <label>:27:                                     ; preds = %47, %79
  %28 = add nuw nsw i64 %23, 1
  %29 = icmp eq i64 %28, %17
  br i1 %29, label %85, label %22

; <label>:30:                                     ; preds = %26, %47
  %31 = phi i64 [ %51, %47 ], [ 0, %26 ]
  br i1 %19, label %33, label %32

; <label>:32:                                     ; preds = %30
  br label %53

; <label>:33:                                     ; preds = %53, %30
  %34 = phi float [ undef, %30 ], [ %75, %53 ]
  %35 = phi i64 [ 0, %30 ], [ %76, %53 ]
  %36 = phi float [ 0.000000e+00, %30 ], [ %75, %53 ]
  br i1 %21, label %47, label %37

; <label>:37:                                     ; preds = %33
  %38 = add nsw i64 %35, %25
  %39 = getelementptr inbounds float, float* %0, i64 %38
  %40 = load float, float* %39, align 4, !tbaa !2
  %41 = mul nsw i64 %35, %14
  %42 = add nsw i64 %41, %31
  %43 = getelementptr inbounds float, float* %1, i64 %42
  %44 = load float, float* %43, align 4, !tbaa !2
  %45 = fmul float %40, %44
  %46 = fadd float %36, %45
  br label %47

; <label>:47:                                     ; preds = %33, %37
  %48 = phi float [ %34, %33 ], [ %46, %37 ]
  %49 = add nsw i64 %31, %24
  %50 = getelementptr inbounds float, float* %2, i64 %49
  store float %48, float* %50, align 4, !tbaa !2
  %51 = add nuw nsw i64 %31, 1
  %52 = icmp eq i64 %51, %12
  br i1 %52, label %27, label %30

; <label>:53:                                     ; preds = %53, %32
  %54 = phi i64 [ 0, %32 ], [ %76, %53 ]
  %55 = phi float [ 0.000000e+00, %32 ], [ %75, %53 ]
  %56 = phi i64 [ %20, %32 ], [ %77, %53 ]
  %57 = add nsw i64 %54, %25
  %58 = getelementptr inbounds float, float* %0, i64 %57
  %59 = load float, float* %58, align 4, !tbaa !2
  %60 = mul nsw i64 %54, %14
  %61 = add nsw i64 %60, %31
  %62 = getelementptr inbounds float, float* %1, i64 %61
  %63 = load float, float* %62, align 4, !tbaa !2
  %64 = fmul float %59, %63
  %65 = fadd float %55, %64
  %66 = or i64 %54, 1
  %67 = add nsw i64 %66, %25
  %68 = getelementptr inbounds float, float* %0, i64 %67
  %69 = load float, float* %68, align 4, !tbaa !2
  %70 = mul nsw i64 %66, %14
  %71 = add nsw i64 %70, %31
  %72 = getelementptr inbounds float, float* %1, i64 %71
  %73 = load float, float* %72, align 4, !tbaa !2
  %74 = fmul float %69, %73
  %75 = fadd float %65, %74
  %76 = add nuw nsw i64 %54, 2
  %77 = add i64 %56, -2
  %78 = icmp eq i64 %77, 0
  br i1 %78, label %33, label %53

; <label>:79:                                     ; preds = %22
  %80 = trunc i64 %23 to i32
  %81 = mul i32 %80, %4
  %82 = sext i32 %81 to i64
  %83 = getelementptr float, float* %2, i64 %82
  %84 = bitcast float* %83 to i8*
  call void @llvm.memset.p0i8.i64(i8* %84, i8 0, i64 %13, i32 4, i1 false)
  br label %27

; <label>:85:                                     ; preds = %27, %8, %6
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

; Function Attrs: norecurse uwtable
define i32 @main(i32, i8** nocapture readnone) local_unnamed_addr #5 {
  %3 = alloca [16384 x float], align 16
  %4 = alloca [16384 x float], align 16
  %5 = alloca [16384 x float], align 16
  %6 = bitcast [16384 x float]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 65536, i8* nonnull %6) #2
  call void @llvm.memset.p0i8.i64(i8* nonnull %6, i8 0, i64 65536, i32 16, i1 false)
  %7 = bitcast [16384 x float]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 65536, i8* nonnull %7) #2
  call void @llvm.memset.p0i8.i64(i8* nonnull %7, i8 0, i64 65536, i32 16, i1 false)
  %8 = bitcast [16384 x float]* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 65536, i8* nonnull %8) #2
  call void @llvm.memset.p0i8.i64(i8* nonnull %8, i8 0, i64 65536, i32 16, i1 false)
  br label %9

; <label>:9:                                      ; preds = %9, %2
  %10 = phi i64 [ 0, %2 ], [ %50, %9 ]
  %11 = getelementptr inbounds [16384 x float], [16384 x float]* %3, i64 0, i64 %10
  %12 = bitcast float* %11 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %12, align 16, !tbaa !2
  %13 = getelementptr float, float* %11, i64 4
  %14 = bitcast float* %13 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %14, align 16, !tbaa !2
  %15 = or i64 %10, 8
  %16 = getelementptr inbounds [16384 x float], [16384 x float]* %3, i64 0, i64 %15
  %17 = bitcast float* %16 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %17, align 16, !tbaa !2
  %18 = getelementptr float, float* %16, i64 4
  %19 = bitcast float* %18 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %19, align 16, !tbaa !2
  %20 = or i64 %10, 16
  %21 = getelementptr inbounds [16384 x float], [16384 x float]* %3, i64 0, i64 %20
  %22 = bitcast float* %21 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %22, align 16, !tbaa !2
  %23 = getelementptr float, float* %21, i64 4
  %24 = bitcast float* %23 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %24, align 16, !tbaa !2
  %25 = or i64 %10, 24
  %26 = getelementptr inbounds [16384 x float], [16384 x float]* %3, i64 0, i64 %25
  %27 = bitcast float* %26 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %27, align 16, !tbaa !2
  %28 = getelementptr float, float* %26, i64 4
  %29 = bitcast float* %28 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %29, align 16, !tbaa !2
  %30 = or i64 %10, 32
  %31 = getelementptr inbounds [16384 x float], [16384 x float]* %3, i64 0, i64 %30
  %32 = bitcast float* %31 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %32, align 16, !tbaa !2
  %33 = getelementptr float, float* %31, i64 4
  %34 = bitcast float* %33 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %34, align 16, !tbaa !2
  %35 = or i64 %10, 40
  %36 = getelementptr inbounds [16384 x float], [16384 x float]* %3, i64 0, i64 %35
  %37 = bitcast float* %36 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %37, align 16, !tbaa !2
  %38 = getelementptr float, float* %36, i64 4
  %39 = bitcast float* %38 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %39, align 16, !tbaa !2
  %40 = or i64 %10, 48
  %41 = getelementptr inbounds [16384 x float], [16384 x float]* %3, i64 0, i64 %40
  %42 = bitcast float* %41 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %42, align 16, !tbaa !2
  %43 = getelementptr float, float* %41, i64 4
  %44 = bitcast float* %43 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %44, align 16, !tbaa !2
  %45 = or i64 %10, 56
  %46 = getelementptr inbounds [16384 x float], [16384 x float]* %3, i64 0, i64 %45
  %47 = bitcast float* %46 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %47, align 16, !tbaa !2
  %48 = getelementptr float, float* %46, i64 4
  %49 = bitcast float* %48 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %49, align 16, !tbaa !2
  %50 = add nuw nsw i64 %10, 64
  %51 = icmp eq i64 %50, 16384
  br i1 %51, label %52, label %9, !llvm.loop !6

; <label>:52:                                     ; preds = %9
  br label %53

; <label>:53:                                     ; preds = %53, %52
  %54 = phi i64 [ 0, %52 ], [ %94, %53 ]
  %55 = getelementptr inbounds [16384 x float], [16384 x float]* %4, i64 0, i64 %54
  %56 = bitcast float* %55 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %56, align 16, !tbaa !2
  %57 = getelementptr float, float* %55, i64 4
  %58 = bitcast float* %57 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %58, align 16, !tbaa !2
  %59 = or i64 %54, 8
  %60 = getelementptr inbounds [16384 x float], [16384 x float]* %4, i64 0, i64 %59
  %61 = bitcast float* %60 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %61, align 16, !tbaa !2
  %62 = getelementptr float, float* %60, i64 4
  %63 = bitcast float* %62 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %63, align 16, !tbaa !2
  %64 = or i64 %54, 16
  %65 = getelementptr inbounds [16384 x float], [16384 x float]* %4, i64 0, i64 %64
  %66 = bitcast float* %65 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %66, align 16, !tbaa !2
  %67 = getelementptr float, float* %65, i64 4
  %68 = bitcast float* %67 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %68, align 16, !tbaa !2
  %69 = or i64 %54, 24
  %70 = getelementptr inbounds [16384 x float], [16384 x float]* %4, i64 0, i64 %69
  %71 = bitcast float* %70 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %71, align 16, !tbaa !2
  %72 = getelementptr float, float* %70, i64 4
  %73 = bitcast float* %72 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %73, align 16, !tbaa !2
  %74 = or i64 %54, 32
  %75 = getelementptr inbounds [16384 x float], [16384 x float]* %4, i64 0, i64 %74
  %76 = bitcast float* %75 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %76, align 16, !tbaa !2
  %77 = getelementptr float, float* %75, i64 4
  %78 = bitcast float* %77 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %78, align 16, !tbaa !2
  %79 = or i64 %54, 40
  %80 = getelementptr inbounds [16384 x float], [16384 x float]* %4, i64 0, i64 %79
  %81 = bitcast float* %80 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %81, align 16, !tbaa !2
  %82 = getelementptr float, float* %80, i64 4
  %83 = bitcast float* %82 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %83, align 16, !tbaa !2
  %84 = or i64 %54, 48
  %85 = getelementptr inbounds [16384 x float], [16384 x float]* %4, i64 0, i64 %84
  %86 = bitcast float* %85 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %86, align 16, !tbaa !2
  %87 = getelementptr float, float* %85, i64 4
  %88 = bitcast float* %87 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %88, align 16, !tbaa !2
  %89 = or i64 %54, 56
  %90 = getelementptr inbounds [16384 x float], [16384 x float]* %4, i64 0, i64 %89
  %91 = bitcast float* %90 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %91, align 16, !tbaa !2
  %92 = getelementptr float, float* %90, i64 4
  %93 = bitcast float* %92 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %93, align 16, !tbaa !2
  %94 = add nuw nsw i64 %54, 64
  %95 = icmp eq i64 %94, 16384
  br i1 %95, label %96, label %53, !llvm.loop !8

; <label>:96:                                     ; preds = %53
  br label %97

; <label>:97:                                     ; preds = %96, %100
  %98 = phi i64 [ %101, %100 ], [ 0, %96 ]
  %99 = shl nsw i64 %98, 7
  br label %103

; <label>:100:                                    ; preds = %105
  %101 = add nuw nsw i64 %98, 1
  %102 = icmp eq i64 %101, 128
  br i1 %102, label %134, label %97

; <label>:103:                                    ; preds = %97, %105
  %104 = phi i64 [ 0, %97 ], [ %108, %105 ]
  br label %110

; <label>:105:                                    ; preds = %110
  %106 = add nuw nsw i64 %104, %99
  %107 = getelementptr inbounds [16384 x float], [16384 x float]* %5, i64 0, i64 %106
  store float %131, float* %107, align 4, !tbaa !2
  %108 = add nuw nsw i64 %104, 1
  %109 = icmp eq i64 %108, 128
  br i1 %109, label %100, label %103

; <label>:110:                                    ; preds = %110, %103
  %111 = phi i64 [ 0, %103 ], [ %132, %110 ]
  %112 = phi float [ 0.000000e+00, %103 ], [ %131, %110 ]
  %113 = add nuw nsw i64 %111, %99
  %114 = getelementptr inbounds [16384 x float], [16384 x float]* %3, i64 0, i64 %113
  %115 = load float, float* %114, align 8, !tbaa !2
  %116 = shl i64 %111, 7
  %117 = add nuw nsw i64 %116, %104
  %118 = getelementptr inbounds [16384 x float], [16384 x float]* %4, i64 0, i64 %117
  %119 = load float, float* %118, align 4, !tbaa !2
  %120 = fmul float %115, %119
  %121 = fadd float %112, %120
  %122 = or i64 %111, 1
  %123 = add nuw nsw i64 %122, %99
  %124 = getelementptr inbounds [16384 x float], [16384 x float]* %3, i64 0, i64 %123
  %125 = load float, float* %124, align 4, !tbaa !2
  %126 = shl i64 %122, 7
  %127 = add nuw nsw i64 %126, %104
  %128 = getelementptr inbounds [16384 x float], [16384 x float]* %4, i64 0, i64 %127
  %129 = load float, float* %128, align 4, !tbaa !2
  %130 = fmul float %125, %129
  %131 = fadd float %121, %130
  %132 = add nuw nsw i64 %111, 2
  %133 = icmp eq i64 %132, 128
  br i1 %133, label %105, label %110

; <label>:134:                                    ; preds = %100
  %135 = getelementptr inbounds [16384 x float], [16384 x float]* %5, i64 0, i64 0
  %136 = load float, float* %135, align 16, !tbaa !2
  %137 = fpext float %136 to double
  %138 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %137)
  %139 = bitcast %"class.std::basic_ostream"* %138 to i8**
  %140 = load i8*, i8** %139, align 8, !tbaa !9
  %141 = getelementptr i8, i8* %140, i64 -24
  %142 = bitcast i8* %141 to i64*
  %143 = load i64, i64* %142, align 8
  %144 = bitcast %"class.std::basic_ostream"* %138 to i8*
  %145 = getelementptr inbounds i8, i8* %144, i64 %143
  %146 = getelementptr inbounds i8, i8* %145, i64 240
  %147 = bitcast i8* %146 to %"class.std::ctype"**
  %148 = load %"class.std::ctype"*, %"class.std::ctype"** %147, align 8, !tbaa !11
  %149 = icmp eq %"class.std::ctype"* %148, null
  br i1 %149, label %150, label %151

; <label>:150:                                    ; preds = %420, %388, %356, %324, %292, %260, %228, %196, %164, %134
  tail call void @_ZSt16__throw_bad_castv() #8
  unreachable

; <label>:151:                                    ; preds = %134
  %152 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %148, i64 0, i32 8
  %153 = load i8, i8* %152, align 8, !tbaa !15
  %154 = icmp eq i8 %153, 0
  br i1 %154, label %158, label %155

; <label>:155:                                    ; preds = %151
  %156 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %148, i64 0, i32 9, i64 10
  %157 = load i8, i8* %156, align 1, !tbaa !17
  br label %164

; <label>:158:                                    ; preds = %151
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %148)
  %159 = bitcast %"class.std::ctype"* %148 to i8 (%"class.std::ctype"*, i8)***
  %160 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %159, align 8, !tbaa !9
  %161 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %160, i64 6
  %162 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %161, align 8
  %163 = tail call signext i8 %162(%"class.std::ctype"* nonnull %148, i8 signext 10)
  br label %164

; <label>:164:                                    ; preds = %155, %158
  %165 = phi i8 [ %157, %155 ], [ %163, %158 ]
  %166 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %138, i8 signext %165)
  %167 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %166)
  %168 = getelementptr inbounds [16384 x float], [16384 x float]* %5, i64 0, i64 1
  %169 = load float, float* %168, align 4, !tbaa !2
  %170 = fpext float %169 to double
  %171 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %170)
  %172 = bitcast %"class.std::basic_ostream"* %171 to i8**
  %173 = load i8*, i8** %172, align 8, !tbaa !9
  %174 = getelementptr i8, i8* %173, i64 -24
  %175 = bitcast i8* %174 to i64*
  %176 = load i64, i64* %175, align 8
  %177 = bitcast %"class.std::basic_ostream"* %171 to i8*
  %178 = getelementptr inbounds i8, i8* %177, i64 %176
  %179 = getelementptr inbounds i8, i8* %178, i64 240
  %180 = bitcast i8* %179 to %"class.std::ctype"**
  %181 = load %"class.std::ctype"*, %"class.std::ctype"** %180, align 8, !tbaa !11
  %182 = icmp eq %"class.std::ctype"* %181, null
  br i1 %182, label %150, label %183

; <label>:183:                                    ; preds = %164
  %184 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %181, i64 0, i32 8
  %185 = load i8, i8* %184, align 8, !tbaa !15
  %186 = icmp eq i8 %185, 0
  br i1 %186, label %190, label %187

; <label>:187:                                    ; preds = %183
  %188 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %181, i64 0, i32 9, i64 10
  %189 = load i8, i8* %188, align 1, !tbaa !17
  br label %196

; <label>:190:                                    ; preds = %183
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %181)
  %191 = bitcast %"class.std::ctype"* %181 to i8 (%"class.std::ctype"*, i8)***
  %192 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %191, align 8, !tbaa !9
  %193 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %192, i64 6
  %194 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %193, align 8
  %195 = tail call signext i8 %194(%"class.std::ctype"* nonnull %181, i8 signext 10)
  br label %196

; <label>:196:                                    ; preds = %190, %187
  %197 = phi i8 [ %189, %187 ], [ %195, %190 ]
  %198 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %171, i8 signext %197)
  %199 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %198)
  %200 = getelementptr inbounds [16384 x float], [16384 x float]* %5, i64 0, i64 2
  %201 = load float, float* %200, align 8, !tbaa !2
  %202 = fpext float %201 to double
  %203 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %202)
  %204 = bitcast %"class.std::basic_ostream"* %203 to i8**
  %205 = load i8*, i8** %204, align 8, !tbaa !9
  %206 = getelementptr i8, i8* %205, i64 -24
  %207 = bitcast i8* %206 to i64*
  %208 = load i64, i64* %207, align 8
  %209 = bitcast %"class.std::basic_ostream"* %203 to i8*
  %210 = getelementptr inbounds i8, i8* %209, i64 %208
  %211 = getelementptr inbounds i8, i8* %210, i64 240
  %212 = bitcast i8* %211 to %"class.std::ctype"**
  %213 = load %"class.std::ctype"*, %"class.std::ctype"** %212, align 8, !tbaa !11
  %214 = icmp eq %"class.std::ctype"* %213, null
  br i1 %214, label %150, label %215

; <label>:215:                                    ; preds = %196
  %216 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %213, i64 0, i32 8
  %217 = load i8, i8* %216, align 8, !tbaa !15
  %218 = icmp eq i8 %217, 0
  br i1 %218, label %222, label %219

; <label>:219:                                    ; preds = %215
  %220 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %213, i64 0, i32 9, i64 10
  %221 = load i8, i8* %220, align 1, !tbaa !17
  br label %228

; <label>:222:                                    ; preds = %215
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %213)
  %223 = bitcast %"class.std::ctype"* %213 to i8 (%"class.std::ctype"*, i8)***
  %224 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %223, align 8, !tbaa !9
  %225 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %224, i64 6
  %226 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %225, align 8
  %227 = tail call signext i8 %226(%"class.std::ctype"* nonnull %213, i8 signext 10)
  br label %228

; <label>:228:                                    ; preds = %222, %219
  %229 = phi i8 [ %221, %219 ], [ %227, %222 ]
  %230 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %203, i8 signext %229)
  %231 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %230)
  %232 = getelementptr inbounds [16384 x float], [16384 x float]* %5, i64 0, i64 3
  %233 = load float, float* %232, align 4, !tbaa !2
  %234 = fpext float %233 to double
  %235 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %234)
  %236 = bitcast %"class.std::basic_ostream"* %235 to i8**
  %237 = load i8*, i8** %236, align 8, !tbaa !9
  %238 = getelementptr i8, i8* %237, i64 -24
  %239 = bitcast i8* %238 to i64*
  %240 = load i64, i64* %239, align 8
  %241 = bitcast %"class.std::basic_ostream"* %235 to i8*
  %242 = getelementptr inbounds i8, i8* %241, i64 %240
  %243 = getelementptr inbounds i8, i8* %242, i64 240
  %244 = bitcast i8* %243 to %"class.std::ctype"**
  %245 = load %"class.std::ctype"*, %"class.std::ctype"** %244, align 8, !tbaa !11
  %246 = icmp eq %"class.std::ctype"* %245, null
  br i1 %246, label %150, label %247

; <label>:247:                                    ; preds = %228
  %248 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %245, i64 0, i32 8
  %249 = load i8, i8* %248, align 8, !tbaa !15
  %250 = icmp eq i8 %249, 0
  br i1 %250, label %254, label %251

; <label>:251:                                    ; preds = %247
  %252 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %245, i64 0, i32 9, i64 10
  %253 = load i8, i8* %252, align 1, !tbaa !17
  br label %260

; <label>:254:                                    ; preds = %247
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %245)
  %255 = bitcast %"class.std::ctype"* %245 to i8 (%"class.std::ctype"*, i8)***
  %256 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %255, align 8, !tbaa !9
  %257 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %256, i64 6
  %258 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %257, align 8
  %259 = tail call signext i8 %258(%"class.std::ctype"* nonnull %245, i8 signext 10)
  br label %260

; <label>:260:                                    ; preds = %254, %251
  %261 = phi i8 [ %253, %251 ], [ %259, %254 ]
  %262 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %235, i8 signext %261)
  %263 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %262)
  %264 = getelementptr inbounds [16384 x float], [16384 x float]* %5, i64 0, i64 4
  %265 = load float, float* %264, align 16, !tbaa !2
  %266 = fpext float %265 to double
  %267 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %266)
  %268 = bitcast %"class.std::basic_ostream"* %267 to i8**
  %269 = load i8*, i8** %268, align 8, !tbaa !9
  %270 = getelementptr i8, i8* %269, i64 -24
  %271 = bitcast i8* %270 to i64*
  %272 = load i64, i64* %271, align 8
  %273 = bitcast %"class.std::basic_ostream"* %267 to i8*
  %274 = getelementptr inbounds i8, i8* %273, i64 %272
  %275 = getelementptr inbounds i8, i8* %274, i64 240
  %276 = bitcast i8* %275 to %"class.std::ctype"**
  %277 = load %"class.std::ctype"*, %"class.std::ctype"** %276, align 8, !tbaa !11
  %278 = icmp eq %"class.std::ctype"* %277, null
  br i1 %278, label %150, label %279

; <label>:279:                                    ; preds = %260
  %280 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %277, i64 0, i32 8
  %281 = load i8, i8* %280, align 8, !tbaa !15
  %282 = icmp eq i8 %281, 0
  br i1 %282, label %286, label %283

; <label>:283:                                    ; preds = %279
  %284 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %277, i64 0, i32 9, i64 10
  %285 = load i8, i8* %284, align 1, !tbaa !17
  br label %292

; <label>:286:                                    ; preds = %279
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %277)
  %287 = bitcast %"class.std::ctype"* %277 to i8 (%"class.std::ctype"*, i8)***
  %288 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %287, align 8, !tbaa !9
  %289 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %288, i64 6
  %290 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %289, align 8
  %291 = tail call signext i8 %290(%"class.std::ctype"* nonnull %277, i8 signext 10)
  br label %292

; <label>:292:                                    ; preds = %286, %283
  %293 = phi i8 [ %285, %283 ], [ %291, %286 ]
  %294 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %267, i8 signext %293)
  %295 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %294)
  %296 = getelementptr inbounds [16384 x float], [16384 x float]* %5, i64 0, i64 5
  %297 = load float, float* %296, align 4, !tbaa !2
  %298 = fpext float %297 to double
  %299 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %298)
  %300 = bitcast %"class.std::basic_ostream"* %299 to i8**
  %301 = load i8*, i8** %300, align 8, !tbaa !9
  %302 = getelementptr i8, i8* %301, i64 -24
  %303 = bitcast i8* %302 to i64*
  %304 = load i64, i64* %303, align 8
  %305 = bitcast %"class.std::basic_ostream"* %299 to i8*
  %306 = getelementptr inbounds i8, i8* %305, i64 %304
  %307 = getelementptr inbounds i8, i8* %306, i64 240
  %308 = bitcast i8* %307 to %"class.std::ctype"**
  %309 = load %"class.std::ctype"*, %"class.std::ctype"** %308, align 8, !tbaa !11
  %310 = icmp eq %"class.std::ctype"* %309, null
  br i1 %310, label %150, label %311

; <label>:311:                                    ; preds = %292
  %312 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %309, i64 0, i32 8
  %313 = load i8, i8* %312, align 8, !tbaa !15
  %314 = icmp eq i8 %313, 0
  br i1 %314, label %318, label %315

; <label>:315:                                    ; preds = %311
  %316 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %309, i64 0, i32 9, i64 10
  %317 = load i8, i8* %316, align 1, !tbaa !17
  br label %324

; <label>:318:                                    ; preds = %311
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %309)
  %319 = bitcast %"class.std::ctype"* %309 to i8 (%"class.std::ctype"*, i8)***
  %320 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %319, align 8, !tbaa !9
  %321 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %320, i64 6
  %322 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %321, align 8
  %323 = tail call signext i8 %322(%"class.std::ctype"* nonnull %309, i8 signext 10)
  br label %324

; <label>:324:                                    ; preds = %318, %315
  %325 = phi i8 [ %317, %315 ], [ %323, %318 ]
  %326 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %299, i8 signext %325)
  %327 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %326)
  %328 = getelementptr inbounds [16384 x float], [16384 x float]* %5, i64 0, i64 6
  %329 = load float, float* %328, align 8, !tbaa !2
  %330 = fpext float %329 to double
  %331 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %330)
  %332 = bitcast %"class.std::basic_ostream"* %331 to i8**
  %333 = load i8*, i8** %332, align 8, !tbaa !9
  %334 = getelementptr i8, i8* %333, i64 -24
  %335 = bitcast i8* %334 to i64*
  %336 = load i64, i64* %335, align 8
  %337 = bitcast %"class.std::basic_ostream"* %331 to i8*
  %338 = getelementptr inbounds i8, i8* %337, i64 %336
  %339 = getelementptr inbounds i8, i8* %338, i64 240
  %340 = bitcast i8* %339 to %"class.std::ctype"**
  %341 = load %"class.std::ctype"*, %"class.std::ctype"** %340, align 8, !tbaa !11
  %342 = icmp eq %"class.std::ctype"* %341, null
  br i1 %342, label %150, label %343

; <label>:343:                                    ; preds = %324
  %344 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %341, i64 0, i32 8
  %345 = load i8, i8* %344, align 8, !tbaa !15
  %346 = icmp eq i8 %345, 0
  br i1 %346, label %350, label %347

; <label>:347:                                    ; preds = %343
  %348 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %341, i64 0, i32 9, i64 10
  %349 = load i8, i8* %348, align 1, !tbaa !17
  br label %356

; <label>:350:                                    ; preds = %343
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %341)
  %351 = bitcast %"class.std::ctype"* %341 to i8 (%"class.std::ctype"*, i8)***
  %352 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %351, align 8, !tbaa !9
  %353 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %352, i64 6
  %354 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %353, align 8
  %355 = tail call signext i8 %354(%"class.std::ctype"* nonnull %341, i8 signext 10)
  br label %356

; <label>:356:                                    ; preds = %350, %347
  %357 = phi i8 [ %349, %347 ], [ %355, %350 ]
  %358 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %331, i8 signext %357)
  %359 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %358)
  %360 = getelementptr inbounds [16384 x float], [16384 x float]* %5, i64 0, i64 7
  %361 = load float, float* %360, align 4, !tbaa !2
  %362 = fpext float %361 to double
  %363 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %362)
  %364 = bitcast %"class.std::basic_ostream"* %363 to i8**
  %365 = load i8*, i8** %364, align 8, !tbaa !9
  %366 = getelementptr i8, i8* %365, i64 -24
  %367 = bitcast i8* %366 to i64*
  %368 = load i64, i64* %367, align 8
  %369 = bitcast %"class.std::basic_ostream"* %363 to i8*
  %370 = getelementptr inbounds i8, i8* %369, i64 %368
  %371 = getelementptr inbounds i8, i8* %370, i64 240
  %372 = bitcast i8* %371 to %"class.std::ctype"**
  %373 = load %"class.std::ctype"*, %"class.std::ctype"** %372, align 8, !tbaa !11
  %374 = icmp eq %"class.std::ctype"* %373, null
  br i1 %374, label %150, label %375

; <label>:375:                                    ; preds = %356
  %376 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %373, i64 0, i32 8
  %377 = load i8, i8* %376, align 8, !tbaa !15
  %378 = icmp eq i8 %377, 0
  br i1 %378, label %382, label %379

; <label>:379:                                    ; preds = %375
  %380 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %373, i64 0, i32 9, i64 10
  %381 = load i8, i8* %380, align 1, !tbaa !17
  br label %388

; <label>:382:                                    ; preds = %375
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %373)
  %383 = bitcast %"class.std::ctype"* %373 to i8 (%"class.std::ctype"*, i8)***
  %384 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %383, align 8, !tbaa !9
  %385 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %384, i64 6
  %386 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %385, align 8
  %387 = tail call signext i8 %386(%"class.std::ctype"* nonnull %373, i8 signext 10)
  br label %388

; <label>:388:                                    ; preds = %382, %379
  %389 = phi i8 [ %381, %379 ], [ %387, %382 ]
  %390 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %363, i8 signext %389)
  %391 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %390)
  %392 = getelementptr inbounds [16384 x float], [16384 x float]* %5, i64 0, i64 8
  %393 = load float, float* %392, align 16, !tbaa !2
  %394 = fpext float %393 to double
  %395 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %394)
  %396 = bitcast %"class.std::basic_ostream"* %395 to i8**
  %397 = load i8*, i8** %396, align 8, !tbaa !9
  %398 = getelementptr i8, i8* %397, i64 -24
  %399 = bitcast i8* %398 to i64*
  %400 = load i64, i64* %399, align 8
  %401 = bitcast %"class.std::basic_ostream"* %395 to i8*
  %402 = getelementptr inbounds i8, i8* %401, i64 %400
  %403 = getelementptr inbounds i8, i8* %402, i64 240
  %404 = bitcast i8* %403 to %"class.std::ctype"**
  %405 = load %"class.std::ctype"*, %"class.std::ctype"** %404, align 8, !tbaa !11
  %406 = icmp eq %"class.std::ctype"* %405, null
  br i1 %406, label %150, label %407

; <label>:407:                                    ; preds = %388
  %408 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %405, i64 0, i32 8
  %409 = load i8, i8* %408, align 8, !tbaa !15
  %410 = icmp eq i8 %409, 0
  br i1 %410, label %414, label %411

; <label>:411:                                    ; preds = %407
  %412 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %405, i64 0, i32 9, i64 10
  %413 = load i8, i8* %412, align 1, !tbaa !17
  br label %420

; <label>:414:                                    ; preds = %407
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %405)
  %415 = bitcast %"class.std::ctype"* %405 to i8 (%"class.std::ctype"*, i8)***
  %416 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %415, align 8, !tbaa !9
  %417 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %416, i64 6
  %418 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %417, align 8
  %419 = tail call signext i8 %418(%"class.std::ctype"* nonnull %405, i8 signext 10)
  br label %420

; <label>:420:                                    ; preds = %414, %411
  %421 = phi i8 [ %413, %411 ], [ %419, %414 ]
  %422 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %395, i8 signext %421)
  %423 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %422)
  %424 = getelementptr inbounds [16384 x float], [16384 x float]* %5, i64 0, i64 9
  %425 = load float, float* %424, align 4, !tbaa !2
  %426 = fpext float %425 to double
  %427 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %426)
  %428 = bitcast %"class.std::basic_ostream"* %427 to i8**
  %429 = load i8*, i8** %428, align 8, !tbaa !9
  %430 = getelementptr i8, i8* %429, i64 -24
  %431 = bitcast i8* %430 to i64*
  %432 = load i64, i64* %431, align 8
  %433 = bitcast %"class.std::basic_ostream"* %427 to i8*
  %434 = getelementptr inbounds i8, i8* %433, i64 %432
  %435 = getelementptr inbounds i8, i8* %434, i64 240
  %436 = bitcast i8* %435 to %"class.std::ctype"**
  %437 = load %"class.std::ctype"*, %"class.std::ctype"** %436, align 8, !tbaa !11
  %438 = icmp eq %"class.std::ctype"* %437, null
  br i1 %438, label %150, label %439

; <label>:439:                                    ; preds = %420
  %440 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %437, i64 0, i32 8
  %441 = load i8, i8* %440, align 8, !tbaa !15
  %442 = icmp eq i8 %441, 0
  br i1 %442, label %446, label %443

; <label>:443:                                    ; preds = %439
  %444 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %437, i64 0, i32 9, i64 10
  %445 = load i8, i8* %444, align 1, !tbaa !17
  br label %452

; <label>:446:                                    ; preds = %439
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %437)
  %447 = bitcast %"class.std::ctype"* %437 to i8 (%"class.std::ctype"*, i8)***
  %448 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %447, align 8, !tbaa !9
  %449 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %448, i64 6
  %450 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %449, align 8
  %451 = tail call signext i8 %450(%"class.std::ctype"* nonnull %437, i8 signext 10)
  br label %452

; <label>:452:                                    ; preds = %446, %443
  %453 = phi i8 [ %445, %443 ], [ %451, %446 ]
  %454 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %427, i8 signext %453)
  %455 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %454)
  call void @llvm.lifetime.end.p0i8(i64 65536, i8* nonnull %8) #2
  call void @llvm.lifetime.end.p0i8(i64 65536, i8* nonnull %7) #2
  call void @llvm.lifetime.end.p0i8(i64 65536, i8* nonnull %6) #2
  ret i32 0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #4

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"*, double) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"*, i8 signext) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"*) local_unnamed_addr #0

; Function Attrs: noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #6

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"*) local_unnamed_addr #0

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_cuda_sgemm_shared3.cpp() #7 section ".text.startup" {
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  %1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #2
  ret void
}

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { noreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"float", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.isvectorized", i32 1}
!8 = distinct !{!8, !7}
!9 = !{!10, !10, i64 0}
!10 = !{!"vtable pointer", !5, i64 0}
!11 = !{!12, !13, i64 240}
!12 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !13, i64 216, !4, i64 224, !14, i64 225, !13, i64 232, !13, i64 240, !13, i64 248, !13, i64 256}
!13 = !{!"any pointer", !4, i64 0}
!14 = !{!"bool", !4, i64 0}
!15 = !{!16, !4, i64 56}
!16 = !{!"_ZTSSt5ctypeIcE", !13, i64 16, !14, i64 24, !13, i64 32, !13, i64 40, !13, i64 48, !4, i64 56, !4, i64 57, !4, i64 313, !4, i64 569}
!17 = !{!4, !4, i64 0}
