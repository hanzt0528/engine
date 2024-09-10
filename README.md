
# 编译
```
cd engine
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/data/hanzt1/he/codes/ml-install-path ..
make
make install
```

## Using cuBLAS

```bash
# fix the path to point to your CUDA compiler

cmake -DGGML_CUBLAS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
make
./bin/test-cuda

```



## 切换为clang编译方法
 ```bash
 cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang ..
 ```
