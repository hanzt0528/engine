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
    int stride = 0;
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
        ctx_size += 64*3*11*11*ggml_type_sizef(GGML_TYPE_F32);//features.0.weight
        ctx_size += 64*ggml_type_sizef(GGML_TYPE_F32);//features.0.bias
        
        ctx_size +=192*64*5*5*ggml_type_sizef(GGML_TYPE_F32);//features.3.weight 
        ctx_size +=192*ggml_type_sizef(GGML_TYPE_F32);//features.3.bias
        
        ctx_size +=384*192*3*3*ggml_type_sizef(GGML_TYPE_F32);//features.6.weight
        ctx_size +=384*ggml_type_sizef(GGML_TYPE_F32);//features.6.bias 
        
        ctx_size +=256*384*3*3*ggml_type_sizef(GGML_TYPE_F32);//features.8.weight
        ctx_size +=256*ggml_type_sizef(GGML_TYPE_F32);//features.8.bias
        
        ctx_size +=256*256*3*3*ggml_type_sizef(GGML_TYPE_F32);//features.10.weight
        ctx_size +=256*ggml_type_sizef(GGML_TYPE_F32);//features.10.bias 

        ctx_size +=4096*9216*ggml_type_sizef(GGML_TYPE_F32);//classifier.1.weight 
        ctx_size +=4096*ggml_type_sizef(GGML_TYPE_F32);//classifier.1.bias
        
        ctx_size +=4096*4096*ggml_type_sizef(GGML_TYPE_F32);//classifier.4.weight
        ctx_size +=4096*ggml_type_sizef(GGML_TYPE_F32);//classifier.4.bias
        
        ctx_size +=1000*4096*ggml_type_sizef(GGML_TYPE_F32);//classifier.6.weight
        ctx_size +=1000*ggml_type_sizef(GGML_TYPE_F32);//classifier.6.bias

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

        
    model.conv2d_layers.resize(5);
    model.conv2d_layers[0].stride = 4;
    model.conv2d_layers[0].padding = 2;

    model.conv2d_layers[1].stride = 1;
    model.conv2d_layers[1].padding = 2;

    model.conv2d_layers[2].stride = 1;
    model.conv2d_layers[2].padding = 1;

    model.conv2d_layers[3].stride = 1;
    model.conv2d_layers[3].padding = 1;

    model.conv2d_layers[4].stride = 1;
    model.conv2d_layers[4].padding = 1;

    for(int i = 0; i < 5; i++)
    {
        {//weight
            // Read dimensions
            int32_t n_dims;
            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            std::cout << "read conv2d n_dims = "<<n_dims<<std::endl;
            int32_t ne_weight[4] = {1,1,1,1};
            for(int j = 0; j < n_dims; j++)
            {
                fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
            }
            model.conv2d_layers[i].weight = ggml_new_tensor_4d(model.ctx,GGML_TYPE_F32,ne_weight[0],ne_weight[1],ne_weight[2],ne_weight[3]);

            //std::cout << "conv1 data size = " << ggml_nbytes(model.conv2d_layers[i].weight)<<std::endl;
            fin.read(reinterpret_cast<char*>(model.conv2d_layers[i].weight->data),ggml_nbytes(model.conv2d_layers[i].weight));
 
        }

        {//bias
            int32_t n_dims;
            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            std::cout << "read conv2d bias dims = "<<n_dims<<std::endl;
            
            int32_t ne_bias[1] = { 1 };
            for (int j = 0; j < 1; j++) {
                fin.read(reinterpret_cast<char *>(&ne_bias[j]), sizeof(ne_bias[j]));
                 std::cout << "n_dims["<<i<<"]= "<<ne_bias[j]<<std::endl;
            }
            //std::cout << "read conv bias dim value = "<<ne_bias[0]<<std::endl;
            model.conv2d_layers[i].bias = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_bias[0]);
            fin.read(reinterpret_cast<char*>(model.conv2d_layers[i].bias->data),ggml_nbytes(model.conv2d_layers[i].bias));
        }
    }

    std::cout << "-------------------------"<<std::endl;
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

            //std::cout << "fc"<<i+1<<" weight data size = " << ggml_nbytes(model.fc_layers[i].weight)<<std::endl;
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


static ggml_tensor * apply_conv2d(ggml_context * ctx, ggml_tensor * input, const conv2d_layer & layer)
{
    struct ggml_tensor * result = ggml_conv_2d(ctx, layer.weight, input, layer.stride, layer.stride, layer.padding, layer.padding, 1, 1);
    result = ggml_add(ctx, result, ggml_repeat(ctx, layer.bias, result));
    result = ggml_relu(ctx, result);

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

std::vector<float> topKElements(std::vector<float>& nums, int k) {
    std::sort(nums.begin(), nums.end(), std::greater<float>()); // 降序排序
    return std::vector<float>(nums.begin(), nums.begin() + k); // 取前K个元素
}

int model_eval(
        const Model & model,
        const int n_threads,
        std::vector<float> digit,
        const char * fname_cgraph
        ) {



    static size_t buf_size = 6000000 * sizeof(float) * 4;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * input = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 224,224,3,1);
    memcpy(input->data, digit.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");


    struct ggml_tensor * result1 = apply_conv2d(ctx0, input, model.conv2d_layers[0]);
    print_shape(0, result1);
    struct ggml_tensor * result2 = ggml_pool_2d(ctx0, result1, GGML_OP_POOL_MAX, 3, 3, 2, 2, 0, 0);
  
    struct ggml_tensor * result3 = apply_conv2d(ctx0, result2, model.conv2d_layers[1]);
    print_shape(1, result3);
    struct ggml_tensor * result4 = ggml_pool_2d(ctx0, result3, GGML_OP_POOL_MAX, 3, 3, 2, 2, 0, 0);   

    struct ggml_tensor * result5 = apply_conv2d(ctx0, result4, model.conv2d_layers[2]);
    print_shape(2, result5);

    struct ggml_tensor * result6 = apply_conv2d(ctx0, result5, model.conv2d_layers[3]);
    print_shape(3, result6);

    struct ggml_tensor * result7 = apply_conv2d(ctx0, result6, model.conv2d_layers[4]);
    print_shape(4, result7);
    struct ggml_tensor * result8 = ggml_pool_2d(ctx0, result7, GGML_OP_POOL_MAX, 3, 3, 2, 2, 0, 0);   
    print_shape(5, result8);

    // struct ggml_tensor * result9 = ggml_pool_2d(ctx0, result8, GGML_OP_POOL_AVG, 2, 2, 1, 1, 1, 1);   
    // print_shape(6, result9);

    struct ggml_tensor * reshape =  result8;

    struct ggml_tensor * view = ggml_reshape_1d(ctx0,reshape, reshape->ne[0]*reshape->ne[1]*reshape->ne[2]*reshape->ne[3]);

    // //memcpy(view->data,reinterpret_cast<char*>(result->data), ggml_nbytes(view));
    print_shape(100, view);
    // // fc1 MLP = Ax + b
    ggml_tensor * fc1 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc_layers[0].weight, view),                model.fc_layers[0].bias);
    print_shape(6, fc1);
    ggml_tensor * fc2 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc_layers[1].weight, ggml_relu(ctx0, fc1)), model.fc_layers[1].bias);
    print_shape(7, fc2);
    ggml_tensor * fc3 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc_layers[2].weight, ggml_relu(ctx0, fc2)), model.fc_layers[2].bias);
    print_shape(8, fc3);

    // soft max
    ggml_tensor * probs = ggml_soft_max(ctx0, fc3);
    print_shape(9, probs);
    ggml_set_name(probs, "probs");

    //build / export / run the computation graph
    ggml_build_forward_expand(gf, probs);
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    //ggml_graph_print   (&gf);
    ggml_graph_dump_dot(gf, NULL, "alexnet.dot");



    {
        struct ggml_tensor * log = probs;
        const float *data= ggml_get_data_f32(log);


        std::cout << "conv1 bias data:"<<std::endl;
        for(int i = 0; i < ggml_nelements(log); i++)
        {
            std::cout << data[i]<<std::endl;
        }

    }


    const float * probs_data = ggml_get_data_f32(probs);



    const int prediction = std::max_element(probs_data, probs_data + 1000) - probs_data;


    std::cout <<"probs_data:"<<std::endl;

    std::vector<float> vprobs;
    for(int i = 0; i < 1000; i ++)
    {
        //std::cout << probs_data[i]<< std::endl;
        vprobs.push_back(probs_data[i]);

    }

    std::vector<float> top5 = topKElements(vprobs,5);

    for(int j = 0; j < 5; j++)
    {
        std::cout << "top["<<j<<"]="<<top5[j]<<std::endl;
    }


    std::string filename = "/data/hanzt1/he/codes/engine/examples/alexnet/imagenet_classes.txt"; // 标签文件的路径
    std::ifstream file(filename); // 创建ifstream对象
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return 1;
    }

    std::vector<std::string> labels; // 存储标签的向量
    std::string label;
    while (std::getline(file, label)) { // 逐行读取
        labels.push_back(label); // 将读取的标签添加到向量中
    }

    file.close(); // 关闭文件


    for(int i = 0; i < 1000; i ++)
    {
        for(int j = 0; j < 5; j++)
        {
            if(top5[j] == probs_data[i])
            {
                std::cout << "top["<<j<<"]="<<top5[j]<<" index = "<<i<<" label = "<<labels[i]<<std::endl;
            }
        }

    }

    ggml_free(ctx0);

    std::cout <<*std::max_element(probs_data, probs_data + 1000)<<std::endl;

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

    float buf[224*224*3];
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
  
        fin.read(reinterpret_cast<char*>(buf),sizeof(buf));

    }

    digit.resize(224*224*3);
    std::cout << "digit="<<std::endl;
    for(int i = 0; i < 224*224*3; i++)
    {
        digit[i] = buf[i];
        if(i < 5)
        {
            std::cout <<digit[i]<<std::endl;
        }
    }
  
    const int prediction = model_eval(model, 1, digit, "alexnet.ggml");

    fprintf(stdout, "%s: predicted digit is %d\n", __func__, prediction);

    // ggml_free(model.ctx);
    return 1;
}