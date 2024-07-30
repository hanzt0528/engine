
# MBAAdd
The MBAAdd pass implements a slightly more involved formula that is only valid for 8 bit integers:
```
a + b == (((a ^ b) + 2 * (a & b)) * 39 + 23) * 151 + 111
```
Similarly to MBASub, it replaces all instances of integer add according to the above identity, but only for 8-bit integers. The LIT tests verify that both the formula and the implementation are correct.

#  Install llvm


#  Building & Testing
### Building
```
cd MBAAdd
mkdir build
cd build
cmake ..
make

```

###  Testing
```
cd build
clang -O1 -emit-llvm -S ../input_for_mba.c -o input_for_mba.ll

define dso_local noundef signext i8 @foo(i8 noundef signext %a, i8 noundef signext %b, i8 noundef signext %c, i8 noundef signext %d) local_unnamed_addr #0 {
entry:
  %add = add i8 %b, %a
  %add5 = add i8 %add, %c
  %add9 = add i8 %add5, %d
  ret i8 %add9
}



opt -load-pass-plugin=./libMBAAdd.so -passes="mba-add" -S input_for_mba.ll -o out.ll


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


```
