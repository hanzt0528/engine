//#include <iostream>
#include "ml.h"
#include <string.h>
#include <cstdio>
#include <iostream>
#include <vector>

static void print_shape(int layer, const ggml_tensor * t)
{
    printf("Layer %2d output shape:  %3d x %3d x %4d x %3d\n", layer, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
}

int main()
{
   std::cout << "conv2d main!"<<std::endl;
    ggml_numa_init();

    init_types();
   
      static size_t buf_size = 2000000 * sizeof(float) * 4;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);


    std::vector<float> digit;

    digit.resize(28*28);

    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            digit[row*28 + col] = 1.0;
        }
    }

    struct ggml_tensor * input = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 28,28,1,1);
    memcpy(input->data, digit.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");


  
    int32_t ne_weight[4] = {3,3,1,1};
       
    struct ggml_tensor *weight = ggml_new_tensor_4d(ctx0,GGML_TYPE_F32,ne_weight[0],ne_weight[1],ne_weight[2],ne_weight[3]);

    std::vector<float> kernel;

    kernel.resize(3*3);
        for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            kernel[row*3 + col] = 1.0;
        }
    }
    memcpy(weight->data, kernel.data(), ggml_nbytes(weight));
    ggml_set_name(weight, "weight");
    struct ggml_tensor * result = ggml_conv_2d(ctx0, weight, input, 1, 1, 1, 1, 1, 1);
 

    // build / export / run the computation graph
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx0, gf, 1);

    //ggml_graph_print   (&gf);
    ggml_graph_dump_dot(gf, NULL, "op_conv2d.dot");


    {
        struct ggml_tensor * log = result;
        const float *data= ggml_get_data_f32(log);


        std::cout << "conv2d data:"<<std::endl;
        for(int i = 0; i < ggml_nelements(log); i++)
        {
           std::cout << data[i]<<std::endl;
        }

    }

    ggml_free(ctx0);

    return 0;
};