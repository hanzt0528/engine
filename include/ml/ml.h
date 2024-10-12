#ifndef ML_H_

#define ML_H_

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#define GGML_FILE_MAGIC   0x67676d6c // "ggml"
#define GGML_FILE_VERSION 1

#define GGML_QNT_VERSION        2    // bump this on quantization format changes
#define GGML_QNT_VERSION_FACTOR 1000 // do not change this


//#define bool int
#define GGML_EXIT_SUCCESS 0
#define GGML_EXIT_ABORTED 1
#define GGML_UNUSED(x) (void)(x)
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define GGML_MAX_DIMS           4
#define GGML_MAX_PARAMS         1024
#define GGML_MAX_CONTEXTS       64
#define GGML_MAX_SRC            6
#define GGML_MAX_NAME           64
#define GGML_MAX_NAME           64
#define GGML_MAX_OP_PARAMS      64
#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

#define GGML_HASHTABLE_FULL ((size_t)-1)
#define GGML_HASHTABLE_ALREADY_EXISTS ((size_t)-2)

#define GGML_DEFAULT_N_THREADS  4
#define GGML_DEFAULT_GRAPH_SIZE 2048
#if UINTPTR_MAX == 0xFFFFFFFF
    #define GGML_MEM_ALIGN 4
#else
    #define GGML_MEM_ALIGN 16
#endif



#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_API __declspec(dllexport)
#        else
#            define GGML_API __declspec(dllimport)
#        endif
#    else
#        define GGML_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define GGML_API
#endif


// void ggml_print_backtrace(void);


// #if (defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)) && \
//     (!defined(TARGET_OS_TV) && !defined(TARGET_OS_WATCH))

// #include <sys/wait.h>
// #include <unistd.h>



// void ggml_print_backtrace(void) {
//     /*
//     #include <execinfo.h>
//     #include <dlfcn.h>

//     void * trace[100];

//     int nptrs = backtrace(trace, sizeof(trace)/sizeof(trace[0]));

//     backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
//     */

//     // backtrack_symbols does not show line numbers, use gdb instead
//     char attach[32];
//     snprintf(attach, sizeof(attach), "attach %d", getpid());
//     int pid = fork();
//     if (pid == 0) {
//         execlp("gdb", "gdb", "--batch",
//             "-ex", "set style enabled on",
//             "-ex", attach,
//             "-ex", "bt -frame-info source-and-location",
//             "-ex", "detach",
//             "-ex", "quit",
//             NULL);
//     } else {
//         waitpid(pid, NULL, 0);
//     }
// }
// #else
// void ggml_print_backtrace(void) {
//     // platform not supported
// }
// #endif



// #define GGML_ASSERT(x) \
//     do { \
//         if (!(x)) { \
//             fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
//             fflush(stderr); \
//             fflush(stdout); \
//             ggml_print_backtrace(); \
//             exit(1); \
//         } \
//     } while (0)

#define GGML_ASSERT(x) {}

typedef uint16_t ggml_fp16_t;

// convert FP16 <-> FP32
GGML_API float       ggml_fp16_to_fp32(ggml_fp16_t x);
GGML_API ggml_fp16_t ggml_fp32_to_fp16(float x);

GGML_API void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n);
GGML_API void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n);

GGML_API struct ggml_tensor *ggml_im2col(
    struct ggml_context *ctx,
    struct ggml_tensor *a,
    struct ggml_tensor *b,
    int s0,
    int s1,
    int p0,
    int p1,
    int d0,
    int d1,
    bool is_2D);

GGML_API struct ggml_tensor *ggml_conv_2d(
    struct ggml_context *ctx,
    struct ggml_tensor *a,
    struct ggml_tensor *b,
    int s0,
    int s1,
    int p0,
    int p1,
    int d0,
    int d1);

enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 (5) support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    // k-quantizations
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
};
enum ggml_backend_type {
    GGML_BACKEND_CPU = 0,
    GGML_BACKEND_GPU = 10,
    GGML_BACKEND_GPU_SPLIT = 20,
};

// model file types
enum ggml_ftype {
    GGML_FTYPE_UNKNOWN     = -1,
    GGML_FTYPE_ALL_F32     = 0,
    GGML_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
    GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
    GGML_FTYPE_MOSTLY_Q3_K = 11, // except 1d tensors
    GGML_FTYPE_MOSTLY_Q4_K = 12, // except 1d tensors
    GGML_FTYPE_MOSTLY_Q5_K = 13, // except 1d tensors
    GGML_FTYPE_MOSTLY_Q6_K = 14, // except 1d tensors
};

// available tensor operations:
enum ggml_op {
    GGML_OP_NONE = 0,

    GGML_OP_DUP,
    GGML_OP_ADD,
    GGML_OP_ADD1,
    GGML_OP_ACC,
    GGML_OP_SUB,
    GGML_OP_MUL,
    GGML_OP_DIV,
    GGML_OP_SQR,
    GGML_OP_SQRT,
    GGML_OP_LOG,
    GGML_OP_SUM,
    GGML_OP_SUM_ROWS,
    GGML_OP_MEAN,
    GGML_OP_ARGMAX,
    GGML_OP_REPEAT,
    GGML_OP_REPEAT_BACK,
    GGML_OP_CONCAT,
    GGML_OP_SILU_BACK,
    GGML_OP_NORM, // normalize
    GGML_OP_RMS_NORM,
    GGML_OP_RMS_NORM_BACK,
    GGML_OP_GROUP_NORM,

    GGML_OP_MUL_MAT,
    GGML_OP_OUT_PROD,

    GGML_OP_SCALE,
    GGML_OP_SET,
    GGML_OP_CPY,
    GGML_OP_CONT,
    GGML_OP_RESHAPE,
    GGML_OP_VIEW,
    GGML_OP_PERMUTE,
    GGML_OP_TRANSPOSE,
    GGML_OP_GET_ROWS,
    GGML_OP_GET_ROWS_BACK,
    GGML_OP_DIAG,
    GGML_OP_DIAG_MASK_INF,
    GGML_OP_DIAG_MASK_ZERO,
    GGML_OP_SOFT_MAX,
    GGML_OP_SOFT_MAX_BACK,
    GGML_OP_ROPE,
    GGML_OP_ROPE_BACK,
    GGML_OP_ALIBI,
    GGML_OP_CLAMP,
    GGML_OP_CONV_TRANSPOSE_1D,
    GGML_OP_IM2COL,
    GGML_OP_CONV_TRANSPOSE_2D,
    GGML_OP_POOL_1D,
    GGML_OP_POOL_2D,

    GGML_OP_UPSCALE, // nearest interpolate

    GGML_OP_FLASH_ATTN,
    GGML_OP_FLASH_FF,
    GGML_OP_FLASH_ATTN_BACK,
    GGML_OP_WIN_PART,
    GGML_OP_WIN_UNPART,
    GGML_OP_GET_REL_POS,
    GGML_OP_ADD_REL_POS,

    GGML_OP_UNARY,

    GGML_OP_MAP_UNARY,
    GGML_OP_MAP_BINARY,

    GGML_OP_MAP_CUSTOM1_F32,
    GGML_OP_MAP_CUSTOM2_F32,
    GGML_OP_MAP_CUSTOM3_F32,

    GGML_OP_MAP_CUSTOM1,
    GGML_OP_MAP_CUSTOM2,
    GGML_OP_MAP_CUSTOM3,

    GGML_OP_CROSS_ENTROPY_LOSS,
    GGML_OP_CROSS_ENTROPY_LOSS_BACK,
    
    GGML_OP_BATCH_NORM,
    GGML_OP_BATCH_REPEAT,

    GGML_OP_COUNT,
};

    enum ggml_unary_op {
        GGML_UNARY_OP_ABS,
        GGML_UNARY_OP_SGN,
        GGML_UNARY_OP_NEG,
        GGML_UNARY_OP_STEP,
        GGML_UNARY_OP_TANH,
        GGML_UNARY_OP_ELU,
        GGML_UNARY_OP_RELU,
        GGML_UNARY_OP_GELU,
        GGML_UNARY_OP_GELU_QUICK,
        GGML_UNARY_OP_SILU,
        GGML_UNARY_OP_LEAKY
    };

enum ggml_object_type {
    GGML_OBJECT_TENSOR,
    GGML_OBJECT_GRAPH,
    GGML_OBJECT_WORK_BUFFER
};

enum ggml_log_level {
    GGML_LOG_LEVEL_ERROR = 2,
    GGML_LOG_LEVEL_WARN = 3,
    GGML_LOG_LEVEL_INFO = 4
};

// ggml object
struct ggml_object {
    size_t offs;
    size_t size;

    struct ggml_object * next;

    enum ggml_object_type type;

    char padding[4];
};

static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);
// n-dimensional tensor
struct ggml_tensor {
    enum ggml_type         type;
    enum ggml_backend_type backend;

    struct ggml_backend_buffer * buffer;

    int     n_dims;
    int64_t ne[GGML_MAX_DIMS]; // number of elements
    size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                // nb[0] = ggml_type_size(type)
                                // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    enum ggml_op op;

    // op params - allocated as int32_t for alignment
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

    bool is_param;

    struct ggml_tensor * grad;
    struct ggml_tensor * src[GGML_MAX_SRC];

    // performance
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;

    struct ggml_tensor * view_src;
    size_t               view_offs;

    void * data;

    char name[GGML_MAX_NAME];

    void * extra; // extra things e.g. for ggml-cuda.cu

    char padding[12];
};

static const size_t GGML_TENSOR_SIZE = sizeof(struct ggml_tensor);


struct ggml_hash_set {
    size_t size;
    struct ggml_tensor ** keys;
};
enum ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
    GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
    GGML_CGRAPH_EVAL_ORDER_COUNT
};

// computation graph
struct ggml_cgraph {
    int size;
    int n_nodes;
    int n_leafs;

    struct ggml_tensor ** nodes;
    struct ggml_tensor ** grads;
    struct ggml_tensor ** leafs;

    struct ggml_hash_set visited_hash_table;

    enum ggml_cgraph_eval_order order;

    // performance
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;
};


struct ggml_cplan {
    size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
    uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

    int n_threads;

    // abort ggml_graph_compute when true
    bool (*abort_callback)(void * data);
    void * abort_callback_data;
};

// scratch buffer
struct ggml_scratch {
    size_t offs;
    size_t size;
    void * data;
};

struct ggml_init_params {
    // memory pool
    size_t mem_size;   // bytes
    void * mem_buffer; // if NULL, memory will be allocated internally
    bool   no_alloc;   // don't allocate memory for the tensor data
};


struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;
    bool   no_alloc_save; // this is used to save the no_alloc state when using scratch buffers

    int    n_objects;

    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;

    struct ggml_scratch scratch;
    struct ggml_scratch scratch_save;
    };


//
// NUMA support
//


struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
};
//
// ggml state
//
struct ggml_context_container {
    bool used;

    struct ggml_context context;
};

struct ggml_state {
    struct ggml_context_container contexts[GGML_MAX_CONTEXTS];
    struct ggml_numa_nodes numa;
};


// global state
//static struct ggml_state g_state;

// use uniquelock
// static atomic_int g_state_barrier = 0;

// // barrier via spin lock
// inline static void ggml_critical_section_start(void) {
//     int processing = atomic_fetch_add(&g_state_barrier, 1);

//     while (processing > 0) {
//         // wait for other threads to finish
//         atomic_fetch_sub(&g_state_barrier, 1);
//         sched_yield(); // TODO: reconsider this
//         processing = atomic_fetch_add(&g_state_barrier, 1);
//     }
// }

// // TODO: make this somehow automatically executed
// //       some sort of "sentry" mechanism
// inline static void ggml_critical_section_end(void) {
//     atomic_fetch_sub(&g_state_barrier, 1);
// }


enum ggml_task_type {
    GGML_TASK_INIT = 0,
    GGML_TASK_COMPUTE,
    GGML_TASK_FINALIZE,
};

struct ggml_compute_params {
    enum ggml_task_type type;

    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;
};

    // misc

GGML_API void    ggml_time_init(void); // call this once at the beginning of the program
GGML_API int64_t ggml_time_ms(void);
GGML_API int64_t ggml_time_us(void);
GGML_API int64_t ggml_cycles(void);
GGML_API int64_t ggml_cycles_per_ms(void);

int ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
// GGML_API void    ggml_numa_init(void); // call once for better performance on NUMA systems
// GGML_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node

 void    ggml_numa_init(void); // call once for better performance on NUMA systems
 int    ggml_is_numa(void); // true if init detected that system has >1 NUMA node

struct ggml_tensor * ggml_new_tensor(
        struct ggml_context * ctx,
        enum   ggml_type      type,
        int                   n_dims,
        const int64_t       * ne) ;

GGML_API struct ggml_tensor * ggml_new_tensor_1d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0);

GGML_API struct ggml_tensor * ggml_new_tensor_2d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1);

GGML_API struct ggml_tensor * ggml_new_tensor_3d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2);

GGML_API struct ggml_tensor * ggml_new_tensor_4d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3);

GGML_API struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);


GGML_API struct ggml_tensor * ggml_add(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

GGML_API struct ggml_tensor * ggml_sub(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

GGML_API struct ggml_tensor * ggml_sub_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

GGML_API struct ggml_tensor * ggml_sqrt(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

GGML_API struct ggml_tensor * ggml_sqrt_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);
        

// A: k columns, n rows => [ne03, ne02, n, k]
// B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
// result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
GGML_API struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

GGML_API struct ggml_tensor * ggml_div(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

GGML_API struct ggml_tensor * ggml_div_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);
        
GGML_API struct ggml_tensor * ggml_get_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

GGML_API void ggml_set_param(
            struct ggml_context * ctx,
            struct ggml_tensor  * tensor);
GGML_API struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);

GGML_API struct ggml_tensor * ggml_format_name(      struct ggml_tensor * tensor, const char * fmt, ...);

// graph allocation in a context
GGML_API struct ggml_cgraph * ggml_new_graph         (struct ggml_context * ctx); // size = GGML_DEFAULT_GRAPH_SIZE, grads = false

GGML_API struct ggml_cgraph * ggml_new_graph_custom(struct ggml_context * ctx, size_t size, bool grads);
GGML_API struct ggml_cgraph * ggml_graph_dup         (struct ggml_context * ctx, struct ggml_cgraph * cgraph);
GGML_API struct ggml_cgraph * ggml_graph_view        (struct ggml_context * ctx, struct ggml_cgraph * cgraph, int i0, int i1);
GGML_API void                 ggml_graph_cpy         (struct ggml_cgraph * src, struct ggml_cgraph * dst);
GGML_API void                 ggml_graph_reset       (struct ggml_cgraph * cgraph);  // zero grads
GGML_API void                 ggml_graph_clear       (struct ggml_cgraph * cgraph);

GGML_API size_t ggml_graph_overhead(void);
GGML_API size_t ggml_graph_overhead_custom(size_t size, bool grads);



GGML_API void ggml_build_forward_expand (struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
GGML_API void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * gf, struct ggml_cgraph * gb, bool keep);
GGML_API void ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
GGML_API struct ggml_cplan ggml_graph_plan   (struct ggml_cgraph * cgraph, int n_threads /*= GGML_DEFAULT_N_THREADS*/);
GGML_API struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);

GGML_API int64_t ggml_nelements   (const struct ggml_tensor * tensor);
GGML_API int64_t ggml_nrows       (const struct ggml_tensor * tensor);
   
GGML_API size_t  ggml_nbytes      (const struct ggml_tensor * tensor);    
    //
    // Internal types and functions exposed for tests and benchmarks
    //


#ifdef  __cplusplus
// restrict not standard in C++
#define GGML_RESTRICT
#else
#define GGML_RESTRICT restrict
#endif


typedef void (*ggml_to_float_t)  (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
typedef void (*ggml_from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int k);
typedef void (*ggml_vec_dot_t)   (const int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT x, const void * GGML_RESTRICT y);

typedef struct {
    const char      * type_name;
    int               blck_size;
    size_t            type_size;
    bool              is_quantized;
    ggml_to_float_t   to_float;
    ggml_from_float_t from_float;
    ggml_from_float_t from_float_reference;
    ggml_vec_dot_t    vec_dot;
    enum ggml_type    vec_dot_type;
} ggml_type_traits_t;




GGML_API ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type);
void init_types();
// #ifdef  __cplusplus
// }
// #endif


    // return view(a), b specifies the new shape
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_API struct ggml_tensor * ggml_reshape(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_API struct ggml_tensor * ggml_reshape_1d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0);

    GGML_API struct ggml_tensor * ggml_reshape_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_API struct ggml_tensor * ggml_reshape_3d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    GGML_API struct ggml_tensor * ggml_reshape_4d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    GGML_API int     ggml_blck_size (enum ggml_type type);
    GGML_API size_t  ggml_type_size (enum ggml_type type); // size in bytes for all elements in a block
    GGML_API float   ggml_type_sizef(enum ggml_type type); // ggml_type_size()/ggml_blck_size() as float

    GGML_API const char * ggml_type_name(enum ggml_type type);
    GGML_API const char * ggml_op_name  (enum ggml_op   op);
    GGML_API const char * ggml_op_symbol(enum ggml_op   op);


    // use this to compute the memory overhead of a tensor
    GGML_API size_t ggml_tensor_overhead(void);
    GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);
    GGML_API void                  ggml_free(struct ggml_context * ctx);

    GGML_API const char *         ggml_get_name   (const struct ggml_tensor * tensor);
    GGML_API struct ggml_tensor * ggml_set_name   (      struct ggml_tensor * tensor, const char * name);
    
    GGML_API enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);

    GGML_API bool ggml_is_transposed(const struct ggml_tensor * tensor);
    GGML_API bool ggml_is_contiguous(const struct ggml_tensor * tensor);
    GGML_API bool ggml_is_permuted  (const struct ggml_tensor * tensor);
    GGML_API struct ggml_tensor * ggml_unary(
            struct ggml_context * ctx,
             struct ggml_tensor * a,
             enum ggml_unary_op op);

    GGML_API struct ggml_tensor * ggml_relu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_soft_max(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_soft_max_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_soft_max_back(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_soft_max_back_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API void *  ggml_get_data    (const struct ggml_tensor * tensor);
    GGML_API float * ggml_get_data_f32(const struct ggml_tensor * tensor);
    GGML_API void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);
    GGML_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    GGML_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    GGML_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    GGML_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    GGML_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    GGML_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    GGML_API void ggml_unravel_index(const struct ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);

    enum ggml_op_pool {
        GGML_OP_POOL_MAX,
        GGML_OP_POOL_AVG,
        GGML_OP_POOL_COUNT,
    };

    // the result will have 2*p0 padding for the first dimension
    // and 2*p1 padding for the second dimension
    GGML_API struct ggml_tensor * ggml_pool_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            enum ggml_op_pool     op,
            int                   k0,
            int                   k1,
            int                   s0,
            int                   s1,
            float                 p0,
            float                 p1);

    // if a is the same shape as b, and a is not parameter, return a
    // otherwise, return a new tensor: repeat(a) to fit in b
    GGML_API struct ggml_tensor * ggml_batch_repeat(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // if a is the same shape as b, and a is not parameter, return a
    // otherwise, return a new tensor: repeat(a) to fit in b
    GGML_API struct ggml_tensor * ggml_repeat(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // sums repetitions in a into shape of b
    GGML_API struct ggml_tensor * ggml_repeat_back(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_batch_norm(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_mul(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);
            
    GGML_API bool    ggml_is_quantized(enum ggml_type type);
    GGML_API size_t  ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split);
    
    // TODO: temporary until model loading of ggml examples is refactored
    GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);

#endif//ML_H_