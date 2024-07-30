#include "ml.h"
#include <string.h>
#include <cstdio>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

struct conv2d_layer {
    struct ggml_tensor * weight;
    struct ggml_tensor * bias;
    int padding = 0;
    int index = 0;
    struct ggml_tensor * add_result;
    struct ggml_tensor * relu_result;
};

struct fc_layer {
    struct ggml_tensor * weight;
    struct ggml_tensor * bias;
};


struct Model{

    std::vector<conv2d_layer> conv2d_layers;
    std::vector<fc_layer> fc_layers;
    struct ggml_context *ctx;
};

bool model_load(const std::string &fname,Model &model)
{
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());
    auto fin = std::ifstream(fname,std::ios::binary);
    if(!fin){
        fprintf(stderr,"%s: failed to open '%s'\n",__func__,fname.c_str());
        return false;
    }

    //verify magic
    {
        uint32_t magic;
        fin.read((char*)&magic,sizeof(magic));
        if(magic != GGML_FILE_MAGIC)
        {
            fprintf(stderr,"%s: invalid model file format '%s'(bad magic)",__func__,fname.c_str());
            return false;
        }
    }

    size_t ctx_size = 0;
    {
        ctx_size += 6*5*5*ggml_type_sizef(GGML_TYPE_F32);//conv1.weight
        ctx_size += 6*ggml_type_sizef(GGML_TYPE_F32);//conv1.bias
        ctx_size +=16*6*5*5*ggml_type_sizef(GGML_TYPE_F32);//conv2.weight
        ctx_size +=16*ggml_type_sizef(GGML_TYPE_F32);//conv2.bias
        ctx_size +=120*256*ggml_type_sizef(GGML_TYPE_F32);//fc1.weight
        ctx_size +=120*ggml_type_sizef(GGML_TYPE_F32);//fc1.bias
        ctx_size +=84*120*ggml_type_sizef(GGML_TYPE_F32);//fc2.weight
        ctx_size +=84*ggml_type_sizef(GGML_TYPE_F32);//fc2.bias
        ctx_size +=10*84*ggml_type_sizef(GGML_TYPE_F32);//fc3.weight
        ctx_size +=10*ggml_type_sizef(GGML_TYPE_F32);//fc3.bias;
    }

        // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size + 1024*1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

        
    model.conv2d_layers.resize(2);
    
    // Read conv1 
    {//weight
        // Read dimensions
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "read conv1 n_dims = "<<n_dims<<std::endl;
        int32_t ne_weight[3] = {1,1,1};
        for(int i = 0; i < n_dims; i++)
        {
            fin.read(reinterpret_cast<char*>(&ne_weight[i]),sizeof(ne_weight[i]));
            std::cout << "n_dims["<<i<<"]= "<<ne_weight[i]<<std::endl;
        }
        model.conv2d_layers[0].weight = ggml_new_tensor_4d(model.ctx,GGML_TYPE_F32,ne_weight[0],ne_weight[1],1,ne_weight[2]);

        std::cout << "conv1 data size = " << ggml_nbytes(model.conv2d_layers[0].weight)<<std::endl;
        fin.read(reinterpret_cast<char*>(model.conv2d_layers[0].weight->data),ggml_nbytes(model.conv2d_layers[0].weight));

        model.conv2d_layers[0].padding = 0;
        model.conv2d_layers[0].index = 0;
    }

    {//bias
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "read conv1 bias dims = "<<n_dims<<std::endl;
        
        int32_t ne_bias[1] = { 1 };
        for (int i = 0; i < 1; ++i) {
            fin.read(reinterpret_cast<char *>(&ne_bias[i]), sizeof(ne_bias[i]));
        }
        std::cout << "read conv1 bias dim value = "<<ne_bias[0]<<std::endl;
        model.conv2d_layers[0].bias = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_bias[0]);
        fin.read(reinterpret_cast<char*>(model.conv2d_layers[0].bias->data),ggml_nbytes(model.conv2d_layers[0].bias));
    }

    // Read conv2
    {//weight
        // Read dimensions
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "read conv2 weight n_dims = "<<n_dims<<std::endl;
        int32_t ne_weight[4] = {1,1,1,1};
        for(int i = 0; i < n_dims; i++)
        {
            fin.read(reinterpret_cast<char*>(&ne_weight[i]),sizeof(ne_weight[i]));
            std::cout << "n_dims["<<i<<"]= "<<ne_weight[i]<<std::endl;
        }
        model.conv2d_layers[1].weight = ggml_new_tensor_4d(model.ctx,GGML_TYPE_F32,ne_weight[0],ne_weight[1],ne_weight[2],ne_weight[3]);

        std::cout << "conv2 weight data size = " << ggml_nbytes(model.conv2d_layers[1].weight)<<std::endl;
        fin.read(reinterpret_cast<char*>(model.conv2d_layers[1].weight->data),ggml_nbytes(model.conv2d_layers[1].weight));
        model.conv2d_layers[1].padding = 0;
        model.conv2d_layers[1].index = 1;

    }

    {//bias
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "read conv2 bias dims = "<<n_dims<<std::endl;
        
        int32_t ne_bias[1] = { 1 };
        for (int i = 0; i < 1; ++i) {
            fin.read(reinterpret_cast<char *>(&ne_bias[i]), sizeof(ne_bias[i]));
        }
        std::cout << "read conv2 bias dim value = "<<ne_bias[0]<<std::endl;
        model.conv2d_layers[1].bias = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_bias[0]);
        fin.read(reinterpret_cast<char*>(model.conv2d_layers[1].bias->data),ggml_nbytes(model.conv2d_layers[1].bias));
    }
    // Read fc1-fc3
    model.fc_layers.resize(3);
    for(int i = 0; i < 3; i++)
    {
        {//weight
                // Read dimensions
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read fc"<<i+1<<" weight n_dims = "<<n_dims<<std::endl;
                int32_t ne_weight[2] = {1,1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }
                model.fc_layers[i].weight = ggml_new_tensor_2d(model.ctx,GGML_TYPE_F32,ne_weight[0],ne_weight[1]);

                std::cout << "fc"<<i+1<<" weight data size = " << ggml_nbytes(model.fc_layers[i].weight)<<std::endl;
                fin.read(reinterpret_cast<char*>(model.fc_layers[i].weight->data),ggml_nbytes(model.fc_layers[i].weight));
            }

            {//bias
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read fc"<<i+1<<" bias dims = "<<n_dims<<std::endl;
                
                int32_t ne_bias[1] = { 1 };
                for (int j = 0; j < 1; ++j) {
                    fin.read(reinterpret_cast<char *>(&ne_bias[j]), sizeof(ne_bias[j]));
                } 
                std::cout << "read fc"<<i+1<<" bias dim value = "<<ne_bias[0]<<std::endl;
                model.fc_layers[i].bias = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_bias[0]);
                fin.read(reinterpret_cast<char*>(model.fc_layers[i].bias->data),ggml_nbytes(model.fc_layers[i].bias));
            }
        
    }
    return true;
}

struct ggml_tensor * add_result;
struct ggml_tensor * relu_result;
struct ggml_tensor * conv2d_result;
static ggml_tensor * apply_conv2d(ggml_context * ctx, ggml_tensor * input, const conv2d_layer & layer)
{
    struct ggml_tensor * result = ggml_conv_2d(ctx, layer.weight, input, 1, 1, layer.padding, layer.padding, 1, 1);
    if(layer.index ==0)
    {
        conv2d_result = result;

    }


    result = ggml_add(ctx, result, ggml_repeat(ctx, layer.bias, result));
    
    if(layer.index ==0)
    {
        add_result = result;

    }
    result = ggml_relu(ctx, result);


    if(layer.index ==0)
    {
        relu_result = result;

    }
    

    return result;
}


// evaluate the model
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - digit:     784 pixel values
//
// returns 0 - 9 prediction


static void print_shape(int layer, const ggml_tensor * t)
{
    printf("Layer %2d output shape:  %3d x %3d x %4d x %3d\n", layer, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
}

int model_eval(
        const Model & model,
        const int n_threads,
        std::vector<float> digit,
        const char * fname_cgraph
        ) {


    // for(int i = 0; i < 28; i++)
    // {
    //     for(int j = 0; j < 28; j++)
    //         {
    //             std::cout << " " << digit[i*28+j];
    //         }
    //         std::cout << std::endl;
    // }
    static size_t buf_size = 2000000 * sizeof(float) * 4;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * input = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 28,28,1,1);
    memcpy(input->data, digit.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");


    struct ggml_tensor * result1 = apply_conv2d(ctx0, input, model.conv2d_layers[0]);
    print_shape(0, result1);
    struct ggml_tensor * result2 = ggml_pool_2d(ctx0, result1, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  
    struct ggml_tensor * result3 = apply_conv2d(ctx0, result2, model.conv2d_layers[1]);
    print_shape(1, result3);
    struct ggml_tensor * result4 = ggml_pool_2d(ctx0, result3, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);   

    struct ggml_tensor * view = ggml_reshape_1d(ctx0,result4, result4->ne[0]*result4->ne[1]*result4->ne[2]*result4->ne[3]);

    //memcpy(view->data,reinterpret_cast<char*>(result->data), ggml_nbytes(view));

    // fc1 MLP = Ax + b
    ggml_tensor * fc1 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc_layers[0].weight, view),                model.fc_layers[0].bias);
    print_shape(2, fc1);
    ggml_tensor * fc2 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc_layers[1].weight, ggml_relu(ctx0, fc1)), model.fc_layers[1].bias);
    print_shape(3, fc2);
    ggml_tensor * fc3 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc_layers[2].weight, ggml_relu(ctx0, fc2)), model.fc_layers[2].bias);
    print_shape(4, fc3);

    // soft max
    ggml_tensor * probs = ggml_soft_max(ctx0, fc3);
    print_shape(5, probs);
    ggml_set_name(probs, "probs");

    // build / export / run the computation graph
    ggml_build_forward_expand(gf, probs);
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    //ggml_graph_print   (&gf);
    ggml_graph_dump_dot(gf, NULL, "lenet.dot");

    if (fname_cgraph) {
        // export the compute graph for later use
        // see the "mnist-cpu" example
        //ggml_graph_export(gf, "mnist.ggml");

        fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
    }

        {
        struct ggml_tensor * log = model.conv2d_layers[0].bias;
        const float *data= ggml_get_data_f32(log);


        std::cout << "conv1 bias data:"<<std::endl;
        for(int i = 0; i < ggml_nelements(log); i++)
        {
           //std::cout << data[i]<<std::endl;
        }

    }

    {
        struct ggml_tensor * log = conv2d_result;
        const float *data= ggml_get_data_f32(log);


        std::cout << "conv2d_result data:"<<std::endl;
        for(int i = 0; i < ggml_nelements(log); i++)
        {
           //std::cout << data[i]<<std::endl;
        }

    }
    {
        struct ggml_tensor * log = add_result;
        const float *data= ggml_get_data_f32(log);


        std::cout << "add_result data:"<<std::endl;
        for(int i = 0; i < ggml_nelements(log); i++)
        {
           // std::cout << data[i]<<std::endl;
        }

    }
    
    {
        struct ggml_tensor * log = relu_result;
        const float *data= ggml_get_data_f32(log);


        std::cout << "relu_result data:"<<std::endl;
        for(int i = 0; i < ggml_nelements(log); i++)
        {
            //std::cout << data[i]<<std::endl;
        }

    }
    
    const float * probs_data = ggml_get_data_f32(probs);

    std::cout <<"probs_data:"<<std::endl;
    for(int i = 0; i < 10; i ++)
    {
        std::cout << probs_data[i]<< std::endl;
    }

    const int prediction = std::max_element(probs_data, probs_data + 10) - probs_data;

    ggml_free(ctx0);

    return prediction;
}


int main(int argc,char* argv[])
{
    std::cout << "lenet main:"<<std::endl;

    ggml_numa_init();
    srand(time(NULL));
    ggml_time_init();

    init_types();

        if (argc != 3) {
        fprintf(stderr, "Usage: %s models/mnist/lenet-ggml-model-f32.bin ./test/raw/t10k-images.idx3-ubyte\n", argv[0]);
        exit(0);
    }

    uint8_t buf[784];
    Model model;
    std::vector<float> digit;
    if (!model_load(argv[1], model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, "models/ggml-model-f32.bin");
        return 1;
    }

          // read a random digit from the test set
    {
        std::ifstream fin(argv[2], std::ios::binary);
        if (!fin) {
            fprintf(stderr, "%s: failed to open '%s'\n", __func__, argv[2]);
            return 1;
        }

        // seek to a random digit: 16-byte header + 28*28 * (random 0 - 10000)
        fin.seekg(16 + 784 * (rand() % 10000));
        fin.read((char *) &buf, sizeof(buf));
    }
    // {
    //     std::ifstream fin(argv[2], std::ios::binary);
    //     if (!fin) {
    //         fprintf(stderr, "%s: failed to open '%s'\n", __func__, argv[2]);
    //         return 1;
    //     }


    //     fin.read((char *) &buf, sizeof(buf));
    // }
        // render the digit in ASCII
    {
        digit.resize(sizeof(buf));

        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                fprintf(stderr, "%c ", (float)buf[row*28 + col] > 0 ? '*' : '_');
                //fprintf(stderr, "%d", buf[row*28 + col]);
                digit[row*28 + col] = ((float)buf[row*28 + col])/255;
            }

            fprintf(stderr, "\n");
        }

        fprintf(stderr, "\n");
    }

    const int prediction = model_eval(model, 1, digit, "lenet.ggml");

    fprintf(stdout, "%s: predicted digit is %d\n", __func__, prediction);

    ggml_free(model.ctx);
    return 1;
}