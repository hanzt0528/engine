; ModuleID = 'input_for_mba.ll'
source_filename = "../input_for_mba.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"ret= %d\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef signext i8 @foo(i8 noundef signext %a, i8 noundef signext %b, i8 noundef signext %c, i8 noundef signext %d) local_unnamed_addr #0 {
entry:
  %0 = and i8 %b, %a
  %1 = mul i8 2, %0
  %2 = xor i8 %b, %a
  %3 = add i8 %2, %1
  %4 = mul i8 39, %3
  %5 = add i8 23, %4
  %6 = mul i8 -105, %5
  %add = add i8 111, %6
  %7 = and i8 %add, %c
  %8 = mul i8 2, %7
  %9 = xor i8 %add, %c
  %10 = add i8 %9, %8
  %11 = mul i8 39, %10
  %12 = add i8 23, %11
  %13 = mul i8 -105, %12
  %add5 = add i8 111, %13
  %14 = and i8 %add5, %d
  %15 = mul i8 2, %14
  %16 = xor i8 %add5, %d
  %17 = add i8 %16, %15
  %18 = mul i8 39, %17
  %19 = add i8 23, %18
  %20 = mul i8 -105, %19
  %add9 = add i8 111, %20
  ret i8 %add9
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %argc, ptr nocapture noundef readonly %argv) local_unnamed_addr #1 {
entry:
  %arrayidx = getelementptr inbounds i8, ptr %argv, i64 8
  %0 = load ptr, ptr %arrayidx, align 8, !tbaa !5
  %call.i = tail call i64 @strtol(ptr nocapture noundef nonnull %0, ptr noundef null, i32 noundef 10) #4
  %arrayidx1 = getelementptr inbounds i8, ptr %argv, i64 16
  %1 = load ptr, ptr %arrayidx1, align 8, !tbaa !5
  %call.i18 = tail call i64 @strtol(ptr nocapture noundef nonnull %1, ptr noundef null, i32 noundef 10) #4
  %arrayidx3 = getelementptr inbounds i8, ptr %argv, i64 24
  %2 = load ptr, ptr %arrayidx3, align 8, !tbaa !5
  %call.i20 = tail call i64 @strtol(ptr nocapture noundef nonnull %2, ptr noundef null, i32 noundef 10) #4
  %arrayidx5 = getelementptr inbounds i8, ptr %argv, i64 32
  %3 = load ptr, ptr %arrayidx5, align 8, !tbaa !5
  %call.i22 = tail call i64 @strtol(ptr nocapture noundef nonnull %3, ptr noundef null, i32 noundef 10) #4
  %conv = trunc i64 %call.i to i32
  %conv7 = trunc i64 %call.i18 to i32
  %conv8 = trunc i64 %call.i20 to i32
  %conv9 = trunc i64 %call.i22 to i32
  %add.i = add i32 %conv7, %conv
  %add5.i = add i32 %add.i, %conv8
  %add9.i = add i32 %add5.i, %conv9
  %sext = shl i32 %add9.i, 24
  %conv11 = ashr exact i32 %sext, 24
  %call12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %conv11)
  ret i32 %conv11
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr nocapture noundef, i32 noundef) local_unnamed_addr #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nofree nounwind willreturn "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 19.0.0git (https://github.com/llvm/llvm-project.git ccc02563f4d620d4d29a1cbd2c463871cc54745b)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
