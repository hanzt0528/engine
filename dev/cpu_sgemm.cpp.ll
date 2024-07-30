; ModuleID = 'cpu_sgemm.cpp'
source_filename = "cpu_sgemm.cpp"
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
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_cpu_sgemm.cpp, i8* null }]

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #0

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: norecurse nounwind uwtable
define void @_Z8cpuSgemmPfS_S_iii(float* nocapture readonly, float* nocapture readonly, float* nocapture, i32, i32, i32) local_unnamed_addr #3 {
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

; Function Attrs: norecurse uwtable
define i32 @main() local_unnamed_addr #4 {
  %1 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double 3.000000e+00)
  %2 = bitcast %"class.std::basic_ostream"* %1 to i8**
  %3 = load i8*, i8** %2, align 8, !tbaa !6
  %4 = getelementptr i8, i8* %3, i64 -24
  %5 = bitcast i8* %4 to i64*
  %6 = load i64, i64* %5, align 8
  %7 = bitcast %"class.std::basic_ostream"* %1 to i8*
  %8 = getelementptr inbounds i8, i8* %7, i64 %6
  %9 = getelementptr inbounds i8, i8* %8, i64 240
  %10 = bitcast i8* %9 to %"class.std::ctype"**
  %11 = load %"class.std::ctype"*, %"class.std::ctype"** %10, align 8, !tbaa !8
  %12 = icmp eq %"class.std::ctype"* %11, null
  br i1 %12, label %13, label %14

; <label>:13:                                     ; preds = %85, %56, %27, %0
  tail call void @_ZSt16__throw_bad_castv() #8
  unreachable

; <label>:14:                                     ; preds = %0
  %15 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %11, i64 0, i32 8
  %16 = load i8, i8* %15, align 8, !tbaa !12
  %17 = icmp eq i8 %16, 0
  br i1 %17, label %21, label %18

; <label>:18:                                     ; preds = %14
  %19 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %11, i64 0, i32 9, i64 10
  %20 = load i8, i8* %19, align 1, !tbaa !14
  br label %27

; <label>:21:                                     ; preds = %14
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %11)
  %22 = bitcast %"class.std::ctype"* %11 to i8 (%"class.std::ctype"*, i8)***
  %23 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %22, align 8, !tbaa !6
  %24 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %23, i64 6
  %25 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %24, align 8
  %26 = tail call signext i8 %25(%"class.std::ctype"* nonnull %11, i8 signext 10)
  br label %27

; <label>:27:                                     ; preds = %18, %21
  %28 = phi i8 [ %20, %18 ], [ %26, %21 ]
  %29 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %1, i8 signext %28)
  %30 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %29)
  %31 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double 2.000000e+00)
  %32 = bitcast %"class.std::basic_ostream"* %31 to i8**
  %33 = load i8*, i8** %32, align 8, !tbaa !6
  %34 = getelementptr i8, i8* %33, i64 -24
  %35 = bitcast i8* %34 to i64*
  %36 = load i64, i64* %35, align 8
  %37 = bitcast %"class.std::basic_ostream"* %31 to i8*
  %38 = getelementptr inbounds i8, i8* %37, i64 %36
  %39 = getelementptr inbounds i8, i8* %38, i64 240
  %40 = bitcast i8* %39 to %"class.std::ctype"**
  %41 = load %"class.std::ctype"*, %"class.std::ctype"** %40, align 8, !tbaa !8
  %42 = icmp eq %"class.std::ctype"* %41, null
  br i1 %42, label %13, label %43

; <label>:43:                                     ; preds = %27
  %44 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %41, i64 0, i32 8
  %45 = load i8, i8* %44, align 8, !tbaa !12
  %46 = icmp eq i8 %45, 0
  br i1 %46, label %50, label %47

; <label>:47:                                     ; preds = %43
  %48 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %41, i64 0, i32 9, i64 10
  %49 = load i8, i8* %48, align 1, !tbaa !14
  br label %56

; <label>:50:                                     ; preds = %43
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %41)
  %51 = bitcast %"class.std::ctype"* %41 to i8 (%"class.std::ctype"*, i8)***
  %52 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %51, align 8, !tbaa !6
  %53 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %52, i64 6
  %54 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %53, align 8
  %55 = tail call signext i8 %54(%"class.std::ctype"* nonnull %41, i8 signext 10)
  br label %56

; <label>:56:                                     ; preds = %50, %47
  %57 = phi i8 [ %49, %47 ], [ %55, %50 ]
  %58 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %31, i8 signext %57)
  %59 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %58)
  %60 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double 3.000000e+00)
  %61 = bitcast %"class.std::basic_ostream"* %60 to i8**
  %62 = load i8*, i8** %61, align 8, !tbaa !6
  %63 = getelementptr i8, i8* %62, i64 -24
  %64 = bitcast i8* %63 to i64*
  %65 = load i64, i64* %64, align 8
  %66 = bitcast %"class.std::basic_ostream"* %60 to i8*
  %67 = getelementptr inbounds i8, i8* %66, i64 %65
  %68 = getelementptr inbounds i8, i8* %67, i64 240
  %69 = bitcast i8* %68 to %"class.std::ctype"**
  %70 = load %"class.std::ctype"*, %"class.std::ctype"** %69, align 8, !tbaa !8
  %71 = icmp eq %"class.std::ctype"* %70, null
  br i1 %71, label %13, label %72

; <label>:72:                                     ; preds = %56
  %73 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %70, i64 0, i32 8
  %74 = load i8, i8* %73, align 8, !tbaa !12
  %75 = icmp eq i8 %74, 0
  br i1 %75, label %79, label %76

; <label>:76:                                     ; preds = %72
  %77 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %70, i64 0, i32 9, i64 10
  %78 = load i8, i8* %77, align 1, !tbaa !14
  br label %85

; <label>:79:                                     ; preds = %72
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %70)
  %80 = bitcast %"class.std::ctype"* %70 to i8 (%"class.std::ctype"*, i8)***
  %81 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %80, align 8, !tbaa !6
  %82 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %81, i64 6
  %83 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %82, align 8
  %84 = tail call signext i8 %83(%"class.std::ctype"* nonnull %70, i8 signext 10)
  br label %85

; <label>:85:                                     ; preds = %79, %76
  %86 = phi i8 [ %78, %76 ], [ %84, %79 ]
  %87 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %60, i8 signext %86)
  %88 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %87)
  %89 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double 2.000000e+00)
  %90 = bitcast %"class.std::basic_ostream"* %89 to i8**
  %91 = load i8*, i8** %90, align 8, !tbaa !6
  %92 = getelementptr i8, i8* %91, i64 -24
  %93 = bitcast i8* %92 to i64*
  %94 = load i64, i64* %93, align 8
  %95 = bitcast %"class.std::basic_ostream"* %89 to i8*
  %96 = getelementptr inbounds i8, i8* %95, i64 %94
  %97 = getelementptr inbounds i8, i8* %96, i64 240
  %98 = bitcast i8* %97 to %"class.std::ctype"**
  %99 = load %"class.std::ctype"*, %"class.std::ctype"** %98, align 8, !tbaa !8
  %100 = icmp eq %"class.std::ctype"* %99, null
  br i1 %100, label %13, label %101

; <label>:101:                                    ; preds = %85
  %102 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %99, i64 0, i32 8
  %103 = load i8, i8* %102, align 8, !tbaa !12
  %104 = icmp eq i8 %103, 0
  br i1 %104, label %108, label %105

; <label>:105:                                    ; preds = %101
  %106 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %99, i64 0, i32 9, i64 10
  %107 = load i8, i8* %106, align 1, !tbaa !14
  br label %114

; <label>:108:                                    ; preds = %101
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %99)
  %109 = bitcast %"class.std::ctype"* %99 to i8 (%"class.std::ctype"*, i8)***
  %110 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %109, align 8, !tbaa !6
  %111 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %110, i64 6
  %112 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %111, align 8
  %113 = tail call signext i8 %112(%"class.std::ctype"* nonnull %99, i8 signext 10)
  br label %114

; <label>:114:                                    ; preds = %108, %105
  %115 = phi i8 [ %107, %105 ], [ %113, %108 ]
  %116 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %89, i8 signext %115)
  %117 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %116)
  ret i32 0
}

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"*, double) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"*, i8 signext) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"*) local_unnamed_addr #0

; Function Attrs: noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #5

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"*) local_unnamed_addr #0

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_cpu_sgemm.cpp() #6 section ".text.startup" {
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  %1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #7

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { argmemonly nounwind }
attributes #8 = { noreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"float", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"vtable pointer", !5, i64 0}
!8 = !{!9, !10, i64 240}
!9 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !10, i64 216, !4, i64 224, !11, i64 225, !10, i64 232, !10, i64 240, !10, i64 248, !10, i64 256}
!10 = !{!"any pointer", !4, i64 0}
!11 = !{!"bool", !4, i64 0}
!12 = !{!13, !4, i64 56}
!13 = !{!"_ZTSSt5ctypeIcE", !10, i64 16, !11, i64 24, !10, i64 32, !10, i64 40, !10, i64 48, !4, i64 56, !4, i64 57, !4, i64 313, !4, i64 569}
!14 = !{!4, !4, i64 0}
