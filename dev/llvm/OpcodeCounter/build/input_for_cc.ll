; ModuleID = '../input_for_cc.c'
source_filename = "../input_for_cc.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo() #0 {
entry:
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @bar() #0 {
entry:
  call void @foo()
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fez() #0 {
entry:
  call void @bar()
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %ii = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  call void @foo()
  call void @bar()
  call void @fez()
  store i32 0, ptr %ii, align 4
  store i32 0, ptr %ii, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %ii, align 4
  %cmp = icmp slt i32 %0, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  call void @foo()
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %1 = load i32, ptr %ii, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr %ii, align 4
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 19.0.0git (https://github.com/llvm/llvm-project.git ccc02563f4d620d4d29a1cbd2c463871cc54745b)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
