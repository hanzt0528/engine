
# 编译
```
cd engine
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/data/hanzt1/he/codes/ml-install-path ..
make
make install
```

 切换为clang编译方法
 ```
 cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang ..
 ```
