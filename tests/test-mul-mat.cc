//#include <iostream>
#include "ml.h"
#include <string.h>
#include <cstdio>


static float frand(void) {
    return (float)rand()/(float)RAND_MAX;
}

static struct ggml_tensor * get_random_tensor(
    struct ggml_context * ctx0, int ndims, int64_t ne[], float fmin, float fmax
) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)result->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)result->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        }
                    }
                }
            }
            break;
        default:
             return result;
    }

    return result;
}

int main()
{
   // std::cout << "call main!"<<std::endl;
    ggml_numa_init();

    init_types();
     struct ggml_cgraph cgraph;
     struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

     cplan.n_threads = 1;



    int64_t ne1[4] = {4000, 1280, 1, 1};
    int64_t ne2[4] = {4000, 1280, 1, 1};
    int64_t ne3[4] = {4000, 1280, 1, 1};
    struct ggml_context * ctx = (struct ggml_context * )malloc(sizeof(struct ggml_context));

   *ctx = (struct ggml_context) {
        /*.mem_size           =*/ 1024*1024*1000,
        /*.mem_buffer         =*/ malloc(1024*1024*1000),
        /*.mem_buffer_owned   =*/ 0,
        /*.no_alloc           =*/ 0,
        /*.no_alloc_save      =*/ 0,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ NULL,
        /*.objects_end        =*/ NULL,
        /*.scratch            =*/ { 0, 0, NULL, },
        /*.scratch_save       =*/ { 0, 0, NULL, },
    };


    struct ggml_tensor * a = get_random_tensor(ctx, 2, ne1, -1, +1);
    struct ggml_tensor * b = get_random_tensor(ctx, 2, ne2, -1, +1);
    ggml_set_param(ctx, a);
    ggml_set_param(ctx, b);

    struct ggml_tensor * c = get_random_tensor(ctx, 2, ne3, -1, +1);

    // struct ggml_tensor * ab = ggml_mul_mat(ctx, a, b);
    // struct ggml_tensor * d  = ggml_sub(ctx, c, ab);
    // struct ggml_tensor * e  = ggml_sum(ctx, ggml_sqr(ctx, d));
    //struct ggml_tensor * d  = ggml_add(ctx, a, b);
    struct ggml_tensor * d = ggml_mul_mat(ctx, a, b);

    struct ggml_cgraph * ge = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);

    ggml_build_forward_expand(ge, d);
    ggml_graph_reset(ge);

    for(int i = 0; i < 1; i++)
    {
        ggml_graph_compute_with_ctx(ctx, ge, /*n_threads*/ 1);

    }
    

    printf("a[0] = %.4f\n",((float *)(a->data))[128]);
    printf("b[0] = %.4f\n",((float *)(b->data))[128]);
    printf("c[0] = %.4f\n",((float *)(d->data))[128]);

    return 0;
};