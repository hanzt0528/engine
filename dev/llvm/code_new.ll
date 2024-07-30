; ModuleID = '/data/hanzt1/he/codes/engine/tests/llvm/code.c'
source_filename = "/data/hanzt1/he/codes/engine/tests/llvm/code.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone
define dso_local i64 @f(i64 noundef %a, i64 noundef %b) #0 {
entry:
  %a.addr = alloca i64, align 8
  %b.addr = alloca i64, align 8
  %x = alloca i64, align 8
  store i64 %a, ptr %a.addr, align 8
  store i64 %b, ptr %b.addr, align 8
  %0 = load i64, ptr %a.addr, align 8
  store i64 %0, ptr %x, align 8
  %1 = load i64, ptr %a.addr, align 8
  %2 = load i64, ptr %b.addr, align 8
  %cmp = icmp sgt i64 %1, %2
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %3 = load i64, ptr %x, align 8
  %add = add nsw i64 %3, 20
  store i64 %add, ptr %x, align 8
  br label %if.end

if.else:                                          ; preds = %entry
  %4 = load i64, ptr %b.addr, align 8
  %5 = load i64, ptr %x, align 8
  %add1 = add nsw i64 %5, %4
  store i64 %add1, ptr %x, align 8
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %6 = load i64, ptr %x, align 8
  ret i64 %6
}

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main(i32 noundef %argc, ptr noundef %argv) #0 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  %k = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 %argc, ptr %argc.addr, align 4
  store ptr %argv, ptr %argv.addr, align 8
  %call = call i64 @f(i64 noundef 12, i64 noundef 22)
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %k, align 4
  ret i32 0
}

attributes #0 = { noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 19.0.0git (https://github.com/llvm/llvm-project.git ccc02563f4d620d4d29a1cbd2c463871cc54745b)"}
