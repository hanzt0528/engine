#include "ml-cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>

static int g_device_count = -1;
static int g_main_device = 0;
static int g_compute_capabilities[GGML_CUDA_MAX_DEVICES];
static float g_tensor_split[GGML_CUDA_MAX_DEVICES] = {0};


#define MAX_STREAMS 8
static cudaStream_t g_cudaStreams[GGML_CUDA_MAX_DEVICES][MAX_STREAMS] = { nullptr };

static bool g_cublas_loaded = false;
bool ggml_cublas_loaded(void) {
    return g_cublas_loaded;
}


static cublasHandle_t g_cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};



/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }


#if CUDART_VERSION >= 12000
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            int id;                                                                     \
            cudaGetDevice(&id);                                                         \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d: %s\n",                         \
                    err_, __FILE__, __LINE__, cublasGetStatusString(err_));             \
            fprintf(stderr, "current device: %d\n", id);                                \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#else
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            int id;                                                                     \
            cudaGetDevice(&id);                                                         \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);  \
            fprintf(stderr, "current device: %d\n", id);                                \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#endif // CUDART_VERSION >= 11


inline cudaError_t ggml_cuda_set_device(const int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return cudaSuccess;
    }

    return cudaSetDevice(device);
}



void ggml_init_cublas() {
    static bool initialized = false;

    if (!initialized) {

#ifdef __HIP_PLATFORM_AMD__
        // Workaround for a rocBLAS bug when using multiple graphics cards:
        // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
        rocblas_initialize();
        CUDA_CHECK(cudaDeviceSynchronize());
#endif

        if (cudaGetDeviceCount(&g_device_count) != cudaSuccess) {
            initialized = true;
            g_cublas_loaded = false;
            return;
        }

        GGML_ASSERT(g_device_count <= GGML_CUDA_MAX_DEVICES);
        int64_t total_vram = 0;
#if defined(GGML_CUDA_FORCE_MMQ)
        fprintf(stderr, "%s: GGML_CUDA_FORCE_MMQ:   yes\n", __func__);
#else
        fprintf(stderr, "%s: GGML_CUDA_FORCE_MMQ:   no\n", __func__);
#endif
#if defined(CUDA_USE_TENSOR_CORES)
        fprintf(stderr, "%s: CUDA_USE_TENSOR_CORES: yes\n", __func__);
#else
        fprintf(stderr, "%s: CUDA_USE_TENSOR_CORES: no\n", __func__);
#endif
        fprintf(stderr, "%s: found %d " GGML_CUDA_NAME " devices:\n", __func__, g_device_count);
        for (int id = 0; id < g_device_count; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            fprintf(stderr, "  Device %d: %s, compute capability %d.%d\n", id, prop.name, prop.major, prop.minor);

            g_tensor_split[id] = total_vram;
            total_vram += prop.totalGlobalMem;
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
            g_compute_capabilities[id] = 100*prop.major + 10*prop.minor + CC_OFFSET_AMD;
#else
            g_compute_capabilities[id] = 100*prop.major + 10*prop.minor;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        }
        for (int id = 0; id < g_device_count; ++id) {
            g_tensor_split[id] /= total_vram;
        }

        for (int id = 0; id < g_device_count; ++id) {
            CUDA_CHECK(ggml_cuda_set_device(id));

            // create cuda streams
            for (int is = 0; is < MAX_STREAMS; ++is) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams[id][is], cudaStreamNonBlocking));
            }

            // create cublas handle
            CUBLAS_CHECK(cublasCreate(&g_cublas_handles[id]));
            CUBLAS_CHECK(cublasSetMathMode(g_cublas_handles[id], CUBLAS_TF32_TENSOR_OP_MATH));
        }

        // configure logging to stdout
        // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

        initialized = true;
        g_cublas_loaded = true;
    }
}


void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // The allocation error can be bypassed. A null ptr will assigned out of this function.
        // This can fixed the OOM error in WSL.
        cudaGetLastError();
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory: %s\n",
            size/1024.0/1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

void ggml_cuda_host_free(void * ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

