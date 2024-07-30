

#  Install llvm

#  Building & Testing
### Building
```
cd HelloWord
mkdir build
cd build
cmake ..
make

```

###  Testing
```
cd build
clang -O1 -S -emit-llvm ../input_for_hello.c -o input_for_hello.ll
opt -load-pass-plugin ./libHelloWorld.so -passes=hello-world -disable-output input_for_hello.ll
(llvm-tutor) Hello from: foo
(llvm-tutor)   number of arguments: 1
(llvm-tutor) Hello from: bar
(llvm-tutor)   number of arguments: 2
(llvm-tutor) Hello from: fez
(llvm-tutor)   number of arguments: 3
(llvm-tutor) Hello from: main
(llvm-tutor)   number of arguments: 2
(base) root@66d923040958:/data/hanzt1/h
```
