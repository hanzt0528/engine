

#  Install llvm
```
source ~/.bashrc 

```
#  Building & Testing
### Building
```
source ~/.bashrc 
cd OpcodeCounter
mkdir build
cd build
cmake ..
make

```

###  Testing
```
cd build
clang -emit-llvm -c ../input_for_cc.c -o input_for_cc.bc
opt -load-pass-plugin ./libOpcodeCounter.so -debug-pass-manager --passes="print<opcode-counter>" -disable-output input_for_cc.bc

=================================================
LLVM-TUTOR: OpcodeCounter results
=================================================
OPCODE               #TIMES USED
-------------------------------------------------
add                  1         
call                 4         
ret                  1         
load                 2         
br                   4         
alloca               2         
store                4         
icmp                 1         
-------------------------------------------------

```
