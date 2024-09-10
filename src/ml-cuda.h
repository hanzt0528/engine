#pragma once

#include "ml.h"
#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif
#define GGML_CUDA_MAX_DEVICES       16

// Always success. To check if CUDA is actually loaded, use `ggml_cublas_loaded`.
GGML_API void   ggml_init_cublas(void);

// Returns `true` if there are available CUDA devices and cublas loads successfully; otherwise, it returns `false`.
GGML_API bool   ggml_cublas_loaded(void);

GGML_API void * ggml_cuda_host_malloc(size_t size);
GGML_API void   ggml_cuda_host_free(void * ptr);
#ifdef  __cplusplus
}
#endif
