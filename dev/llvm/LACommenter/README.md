

#  Install llvm

#  Building & Testing
### Building
```
cd LACommenter
mkdir build
cd build
cmake ..
make

```

###  Testing
```
cd build
clang -cc1 -load ./libLACommenter.so -plugin LACommenter ../input_file.cpp

extern void foo(int some_arg);

void bar() {
  foo(/*some_arg=*/123);
}
```
