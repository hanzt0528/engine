

#  Install llvm
```
source ~/.bashrc 

```
#  Building & Testing
### Building
```
source ~/.bashrc 
cd InjectFuncCall
mkdir build
cd build
cmake ..
make

```

###  Testing
```
cd build
clang -O0 -emit-llvm -c ../input_for_hello.c -o input_for_hello.bc
opt -load-pass-plugin ./libInjectFuncCall.so --passes="inject-func-call" input_for_hello.bc -o instrumented.bin

lli instrumented.bin
(llvm-tutor) Hello from: main
(llvm-tutor)   number of arguments: 2
(llvm-tutor) Hello from: foo
(llvm-tutor)   number of arguments: 1
(llvm-tutor) Hello from: bar
(llvm-tutor)   number of arguments: 2
(llvm-tutor) Hello from: foo
(llvm-tutor)   number of arguments: 1
(llvm-tutor) Hello from: fez
(llvm-tutor)   number of arguments: 3
(llvm-tutor) Hello from: bar
(llvm-tutor)   number of arguments: 2
(llvm-tutor) Hello from: foo
(llvm-tutor)   number of arguments: 1
```
