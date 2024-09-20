#include "ml.h"
#include "ml-cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <atomic>



#define MIN_CC_DP4A   610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define CC_VOLTA      700
#define CC_OFFSET_AMD 1000000
#define CC_RDNA2      (CC_OFFSET_AMD + 1030)

#define GGML_CUDA_MAX_NODES 8192

#define WARP_SIZE 32

#define CUDA_ADD_BLOCK_SIZE 256
#define CUDA_MUL_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_CPY_BLOCK_SIZE 32
#define CUDA_SCALE_BLOCK_SIZE 256
#define CUDA_CLAMP_BLOCK_SIZE 256
#define CUDA_ROPE_BLOCK_SIZE 256
#define CUDA_ALIBI_BLOCK_SIZE 32
#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define CUDA_GET_ROWS_BLOCK_SIZE 256

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses


static int g_device_count = -1;
static int g_main_device = 0;
static int g_compute_capabilities[GGML_CUDA_MAX_DEVICES];
static float g_tensor_split[GGML_CUDA_MAX_DEVICES] = {0};


#define MAX_STREAMS 8
static cudaStream_t g_cudaStreams[GGML_CUDA_MAX_DEVICES][MAX_STREAMS] = { nullptr };
struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES][MAX_STREAMS]; // events for synchronizing multiple GPUs
};

static bool g_cublas_loaded = false;
bool ggml_cublas_loaded(void) {
    return g_cublas_loaded;
}


static cublasHandle_t g_cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};


static __global__ void add_f32(const float * x, const float * y, float * dst, const int kx, const int ky) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= kx) {
        return;
    }
    dst[i] = x[i] + y[i%ky];
}

static __global__ void add_f16_f32_f16(const half * x, const float * y, half * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = __hadd(x[i], __float2half(y[i]));
}

static __global__ void add_f16_f32_f32(const half * x, const float * y, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = __half2float(x[i]) + y[i];
}


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

typedef void (*ggml_cuda_func_t)(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
typedef void (*ggml_cuda_op_flatten_t)(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream);


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

bool ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    if (!g_cublas_loaded) return false;

    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
            src1->type == GGML_TYPE_F32 &&
             dst->type == GGML_TYPE_F32 &&
            (ne0 >= 32 && ne1 >= 32 && ne10 >= 32);
}

void ggml_cuda_set_tensor_split(const float * tensor_split) {
    if (tensor_split == nullptr) {
        return;
    }
    bool all_zero = true;
    for (int i = 0; i < g_device_count; ++i) {
        if (tensor_split[i] != 0.0f) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return;
    }
    float split_sum = 0.0f;
    for (int i = 0; i < g_device_count; ++i) {
        g_tensor_split[i] = split_sum;
        split_sum += tensor_split[i];
    }
    for (int i = 0; i < g_device_count; ++i) {
        g_tensor_split[i] /= split_sum;
    }
}
static int64_t get_row_rounding(ggml_type type) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int64_t id = 0; id < g_device_count; ++id) {
        if (g_tensor_split[id] < (id + 1 < g_device_count ? g_tensor_split[id + 1] : 1.0f)) {
            if (min_compute_capability > g_compute_capabilities[id]) {
                min_compute_capability = g_compute_capabilities[id];
            }
            if (max_compute_capability < g_compute_capabilities[id]) {
                max_compute_capability = g_compute_capabilities[id];
            }
        }
    }

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return max_compute_capability >= CC_RDNA2 ? 128 : 64;
        case GGML_TYPE_F16:
            return 1;
        case GGML_TYPE_Q2_K:
            return max_compute_capability >= CC_RDNA2 ? 128 : 32;
        case GGML_TYPE_Q3_K:
            return min_compute_capability < CC_RDNA2 ? 128 : 64;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return max_compute_capability >= CC_RDNA2 ? 128 : 64;
        default:
            GGML_ASSERT(false);
    }
#else
    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= CC_VOLTA ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return max_compute_capability >= CC_VOLTA ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ASSERT(false);
    }
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
}


void ggml_cuda_transform_tensor(void * data, struct ggml_tensor * tensor) {
    const int64_t nrows = ggml_nrows(tensor);

    const int64_t ne0 = tensor->ne[0];

    const size_t nb1 = tensor->nb[1];

    ggml_backend_type backend = tensor->backend;
    ggml_tensor_extra_gpu * extra = new struct ggml_tensor_extra_gpu;
    memset(extra, 0, sizeof(*extra));

    for (int64_t id = 0; id < g_device_count; ++id) {
        if (backend == GGML_BACKEND_GPU && id != g_main_device) {
            continue;
        }

        ggml_cuda_set_device(id);

        int64_t row_low, row_high;
        if (backend == GGML_BACKEND_GPU) {
            row_low = 0;
            row_high = nrows;
        } else if (backend == GGML_BACKEND_GPU_SPLIT) {
            const int64_t rounding = get_row_rounding(tensor->type);

            row_low = id == 0 ? 0 : nrows*g_tensor_split[id];
            row_low -= row_low % rounding;

            if (id == g_device_count - 1) {
                row_high = nrows;
            } else {
                row_high = nrows*g_tensor_split[id + 1];
                row_high -= row_high % rounding;
            }
        } else {
            GGML_ASSERT(false);
        }
        if (row_low == row_high) {
            continue;
        }

        int64_t nrows_split = row_high - row_low;

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += (MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING)
                * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
        }

        char * buf;
        CUDA_CHECK(cudaMalloc(&buf, size));
        char * buf_host = (char*)data + offset_split;

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            CUDA_CHECK(cudaMemset(buf + original_size, 0, size - original_size));
        }

        CUDA_CHECK(cudaMemcpy(buf, buf_host, original_size, cudaMemcpyHostToDevice));

        extra->data_device[id] = buf;

        if (backend == GGML_BACKEND_GPU_SPLIT) {
            for (int64_t is = 0; is < MAX_STREAMS; ++is) {
                CUDA_CHECK(cudaEventCreateWithFlags(&extra->events[id][is], cudaEventDisableTiming));
            }
        }
    }

    tensor->extra = extra;
}


void ggml_cuda_free_data(struct ggml_tensor * tensor) {
    if (!tensor || (tensor->backend != GGML_BACKEND_GPU && tensor->backend != GGML_BACKEND_GPU_SPLIT) ) {
        return;
    }

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int64_t id = 0; id < g_device_count; ++id) {
        if (extra->data_device[id] != nullptr) {
            CUDA_CHECK(ggml_cuda_set_device(id));
            CUDA_CHECK(cudaFree(extra->data_device[id]));
        }

        for (int64_t is = 0; is < MAX_STREAMS; ++is) {
            if (extra->events[id][is] != nullptr) {
                CUDA_CHECK(ggml_cuda_set_device(id));
                CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
            }
        }
    }

    delete extra;
}


static void add_f32_cuda(const float * x, const float * y, float * dst, const int kx, const int ky, cudaStream_t stream) {
    const int num_blocks = (kx + CUDA_ADD_BLOCK_SIZE - 1) / CUDA_ADD_BLOCK_SIZE;
    add_f32<<<num_blocks, CUDA_ADD_BLOCK_SIZE, 0, stream>>>(x, y, dst, kx, ky);
}


// buffer pool for cuda
#define MAX_CUDA_BUFFERS 256

struct scoped_spin_lock {
    std::atomic_flag& lock;
    scoped_spin_lock(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            ; // spin
        }
    }
    ~scoped_spin_lock() {
        lock.clear(std::memory_order_release);
    }
    scoped_spin_lock(const scoped_spin_lock&) = delete;
    scoped_spin_lock& operator=(const scoped_spin_lock&) = delete;
};

struct cuda_buffer {
    void * ptr = nullptr;
    size_t size = 0;
};

static cuda_buffer g_cuda_buffer_pool[GGML_CUDA_MAX_DEVICES][MAX_CUDA_BUFFERS];
static std::atomic_flag g_cuda_pool_lock = ATOMIC_FLAG_INIT;

static void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_cuda_pool_lock);
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
#ifdef DEBUG_CUDA_MALLOC
    int nnz = 0;
    size_t max_size = 0, tot_size = 0;
#endif
    size_t best_diff = 1ull << 36;
    int ibest = -1;
    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[id][i];
        if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
            ++nnz;
            tot_size += b.size;
            if (b.size > max_size) max_size = b.size;
#endif
            if (b.size >= size) {
                size_t diff = b.size - size;
                if (diff < best_diff) {
                    best_diff = diff;
                    ibest = i;
                    if (!best_diff) {
                        void * ptr = b.ptr;
                        *actual_size = b.size;
                        b.ptr = nullptr;
                        b.size = 0;
                        return ptr;
                    }
                }
            }
        }
    }
    if (ibest >= 0) {
        cuda_buffer& b = g_cuda_buffer_pool[id][ibest];
        void * ptr = b.ptr;
        *actual_size = b.size;
        b.ptr = nullptr;
        b.size = 0;
        return ptr;
    }
#ifdef DEBUG_CUDA_MALLOC
    fprintf(stderr, "%s: %d buffers, max_size = %u MB, tot_size = %u MB, requested %u MB\n", __func__, nnz,
            (uint32_t)(max_size/1024/1024), (uint32_t)(tot_size/1024/1024), (uint32_t)(size/1024/1024));
#endif
    void * ptr;
    size_t look_ahead_size = (size_t) (1.05 * size);
    look_ahead_size = 256 * ((look_ahead_size + 255)/256);
    CUDA_CHECK(cudaMalloc((void **) &ptr, look_ahead_size));
    *actual_size = look_ahead_size;
    return ptr;
}


static void ggml_cuda_pool_free(void * ptr, size_t size) {
    scoped_spin_lock lock(g_cuda_pool_lock);
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[id][i];
        if (b.ptr == nullptr) {
            b.ptr = ptr;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
    CUDA_CHECK(cudaFree(ptr));
}

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    cudaMemcpyKind kind;
    char * src_ptr;
    if (src->backend == GGML_BACKEND_CPU) {
        kind = cudaMemcpyHostToDevice;
        src_ptr = (char *) src->data;
    } else if (src->backend == GGML_BACKEND_GPU || src->backend == GGML_BACKEND_GPU_SPLIT) {
        GGML_ASSERT(src->backend != GGML_BACKEND_GPU_SPLIT || (i1_low == 0 && i1_high == src->ne[1]));
        kind = cudaMemcpyDeviceToDevice;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        CUDA_CHECK(cudaGetDevice(&id));
        src_ptr = (char *) extra->data_device[id];
    } else {
        GGML_ASSERT(false);
    }
    char * dst_ptr = (char *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, kind, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, kind, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, kind, stream);
            if (r != cudaSuccess) return r;
        }
        return cudaSuccess;
    }
}


static void ggml_cuda_op_flatten(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const ggml_cuda_op_flatten_t op) {
    const int64_t nrows0 = ggml_nrows(src0);

    const bool use_src1 = src1 != nullptr;
    const int64_t nrows1 = use_src1 ? ggml_nrows(src1) : 1;

    GGML_ASSERT(!use_src1 || src1->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(              dst->backend != GGML_BACKEND_GPU_SPLIT);

    ggml_tensor_extra_gpu * src0_extra =            (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = use_src1 ? (ggml_tensor_extra_gpu *) src1->extra : nullptr;
    ggml_tensor_extra_gpu * dst_extra  =            (ggml_tensor_extra_gpu *)  dst->extra;

    const bool src0_on_device =             src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT;
    const bool src1_on_device = use_src1 && src1->backend == GGML_BACKEND_GPU;
    const bool  dst_on_device =              dst->backend == GGML_BACKEND_GPU;

    const bool src1_stays_on_host = use_src1 && dst->op == GGML_OP_SCALE;

    // dd = data device
    float * src0_ddf = nullptr;
    float * src1_ddf = nullptr;
    float *  dst_ddf = nullptr;

    // as = actual size
    size_t src0_asf = 0;
    size_t src1_asf = 0;
    size_t  dst_asf = 0;

    ggml_cuda_set_device(g_main_device);
    const cudaStream_t main_stream = g_cudaStreams[g_main_device][0];

    if (src0_on_device) {
        src0_ddf = (float *) src0_extra->data_device[g_main_device];
    } else {
        src0_ddf = (float *) ggml_cuda_pool_malloc(ggml_nbytes(src0), &src0_asf);
        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_ddf, src0, 0, 0, 0, nrows0, main_stream));
    }

    if (use_src1 && !src1_stays_on_host) {
        if (src1_on_device) {
            src1_ddf = (float *) src1_extra->data_device[g_main_device];
        } else {
            src1_ddf = (float *) ggml_cuda_pool_malloc(ggml_nbytes(src1), &src1_asf);
            CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src1_ddf, src1, 0, 0, 0, nrows1, main_stream));
        }
    }
    if (dst_on_device) {
        dst_ddf = (float *) dst_extra->data_device[g_main_device];
    } else {
        dst_ddf = (float *) ggml_cuda_pool_malloc(ggml_nbytes(dst), &dst_asf);
    }

    // do the computation
    op(src0, src1, dst, src0_ddf, src1_ddf, dst_ddf, main_stream);
    CUDA_CHECK(cudaGetLastError());

    // copy dst to host if necessary
    if (!dst_on_device) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, dst_ddf, ggml_nbytes(dst), cudaMemcpyDeviceToHost, main_stream));
    }

    if (src0_asf > 0) {
        ggml_cuda_pool_free(src0_ddf, src0_asf);
    }
    if (src1_asf > 0) {
        ggml_cuda_pool_free(src1_ddf, src1_asf);
    }
    if (dst_asf > 0) {
        ggml_cuda_pool_free(dst_ddf, dst_asf);
    }

    if (dst->backend == GGML_BACKEND_CPU) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

inline void ggml_cuda_op_add(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const float * src0_dd, const float * src1_dd, float * dst_dd, const cudaStream_t & main_stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        add_f32_cuda(src0_dd, src1_dd, dst_dd, ggml_nelements(src0), ne10*ne11, main_stream);
    }
    //  else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
    //     add_f16_f32_f16_cuda((const half *) src0_dd, src1_dd, (half *) dst_dd, ggml_nelements(src0), main_stream);
    // } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
    //     add_f16_f32_f32_cuda((const half *) src0_dd, src1_dd, dst_dd, ggml_nelements(src0), main_stream);
    // } 
    else {
        fprintf(stderr, "src0->type: %d  dst->type: %d\n", src0->type, dst->type);
        GGML_ASSERT(false);
    }

    (void) src1;
    (void) dst;
}

static void ggml_cuda_add(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_op_flatten(src0, src1, dst, ggml_cuda_op_add);
}

bool ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    if (!g_cublas_loaded) return false;

    std::cout << "ggml_cuda_compute_forward" << std::endl;

    ggml_cuda_func_t func;
    const bool any_on_device = tensor->backend == GGML_BACKEND_GPU
        || (tensor->src[0] != nullptr && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT))
        || (tensor->src[1] != nullptr && tensor->src[1]->backend == GGML_BACKEND_GPU);

    if (!any_on_device && tensor->op != GGML_OP_MUL_MAT) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT) {
        if (tensor->src[0]->ne[3] != tensor->src[1]->ne[3]) {
#ifndef NDEBUG
            fprintf(stderr, "%s: cannot compute %s: src0->ne[3] = %d, src1->ne[3] = %d - fallback to CPU\n", __func__, tensor->name, tensor->src[0]->ne[3], tensor->src[1]->ne[3]);
#endif
            return false;
        }
    }

    switch (tensor->op) {
        // case GGML_OP_REPEAT:
        //     func = ggml_cuda_repeat;
        //     break;
        // case GGML_OP_GET_ROWS:
        //     func = ggml_cuda_get_rows;
        //     break;
        // case GGML_OP_DUP:
        //     func = ggml_cuda_dup;
        //     break;
        case GGML_OP_ADD:
            func = ggml_cuda_add;
            break;
        // case GGML_OP_MUL:
        //     func = ggml_cuda_mul;
        //     break;
        // case GGML_OP_UNARY:
        //     switch (ggml_get_unary_op(tensor)) {
        //         case GGML_UNARY_OP_GELU:
        //             func = ggml_cuda_gelu;
        //             break;
        //         case GGML_UNARY_OP_SILU:
        //             func = ggml_cuda_silu;
        //             break;
        //         case GGML_UNARY_OP_RELU:
        //             func = ggml_cuda_relu;
        //             break;
        //         default:
        //             return false;
        //     } break;
        // case GGML_OP_NORM:
        //     func = ggml_cuda_norm;
        //     break;
        // case GGML_OP_RMS_NORM:
        //     func = ggml_cuda_rms_norm;
        //     break;
        // case GGML_OP_MUL_MAT:
        //     if (!any_on_device && !ggml_cuda_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
        //         return false;
        //     }
        //     func = ggml_cuda_mul_mat;
        //     break;
        // case GGML_OP_SCALE:
        //     func = ggml_cuda_scale;
        //     break;
        // case GGML_OP_SQR:
        //     func = ggml_cuda_sqr;
        //     break;
        // case GGML_OP_CLAMP:
        //     if (!any_on_device) {
        //         return false;
        //     }
        //     func = ggml_cuda_clamp;
        //     break;
        // case GGML_OP_CPY:
        //     func = ggml_cuda_cpy;
        //     break;
        // case GGML_OP_CONT:
        //     func = ggml_cuda_dup;
        //     break;
        // case GGML_OP_RESHAPE:
        // case GGML_OP_VIEW:
        // case GGML_OP_PERMUTE:
        // case GGML_OP_TRANSPOSE:
        //     func = ggml_cuda_nop;
        //     break;
        // case GGML_OP_DIAG_MASK_INF:
        //     func = ggml_cuda_diag_mask_inf;
        //     break;
        // case GGML_OP_SOFT_MAX:
        //     func = ggml_cuda_soft_max;
        //     break;
        // case GGML_OP_ROPE:
        //     func = ggml_cuda_rope;
        //     break;
        // case GGML_OP_ALIBI:
        //     func = ggml_cuda_alibi;
        //     break;
        // case GGML_OP_IM2COL:
        //     func = ggml_cuda_im2col;
        //     break;
        default:
            return false;
    }

    if (params->ith != 0) {
        return true;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return true;
    }
    func(tensor->src[0], tensor->src[1], tensor);
    return true;
}
