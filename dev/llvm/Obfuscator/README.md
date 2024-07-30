

#  Install llvm

#  Building & Testing
### Building
```
cd Obfuscator
mkdir build
cd build
cmake ..
make

```

###  Testing
```
cd build
clang -cc1 -load ./libObfuscator.so -plugin Obfuscator ../input_add.cpp

extern void foo(int some_arg);

void bar() {
  foo(/*some_arg=*/123);
}
```


# Reference
```
https://github.com/banach-space/clang-tutor/tree/main
```
