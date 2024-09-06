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
    bool have_bias=false;
};

struct bn_layer {
    struct ggml_tensor * weight;
    struct ggml_tensor * bias;
    struct ggml_tensor * mean;
    struct ggml_tensor * var;

};

struct downsample_layer {
    struct conv2d_layer conv;
    struct bn_layer bn;

};

struct basic_block{
    struct conv2d_layer conv1;
    struct bn_layer     bn1;
    struct conv2d_layer conv2;
    struct bn_layer     bn2;
    struct conv2d_layer conv3;
    struct bn_layer     bn3;

    downsample_layer dowsample;
    bool have_dowsample= false;

};

struct resnet_layer {
    std::vector<basic_block> blocks;
};


struct fc_layer {
    struct ggml_tensor * weight;
    struct ggml_tensor * bias;
};

//ResNet50
struct Model{
    struct conv2d_layer conv1;
    struct bn_layer bn1;
    std::vector<resnet_layer> layers;
    fc_layer fc;

    struct ggml_context *ctx;
};
struct BottleneckConfig
{
    int kerner_size;
    int bn_size;
};

struct ModelLayer
{
    std::vector<BottleneckConfig> bottle_configs;
};
struct ModelConfig{
    public:
    std::vector<ModelLayer> layers;
    std::vector<int> blocks;
};

ModelConfig config;

void init_config_resnet18()
{
    config.blocks.resize(4);
    config.blocks[0]=2;
    config.blocks[1]=2;
    config.blocks[2]=2;
    config.blocks[3]=2;
}


void init_config_resnet34()
{
    config.blocks.resize(4);
    config.blocks[0]=3;
    config.blocks[1]=4;
    config.blocks[2]=6;
    config.blocks[3]=3;
}
void init_config_resnet50()
{
    config.blocks.resize(4);
    config.blocks[0]=3;
    config.blocks[1]=4;
    config.blocks[2]=6;
    config.blocks[3]=3;

    config.layers.resize(4);

    int bn_size = 64;
    for(int layer = 0; layer< 4;layer++)
    {
        config.layers[layer].bottle_configs.resize(3);
        
        if(layer == 0)
        {
            bn_size = 64;
        }
        else
        {
            bn_size = bn_size*2;
        }
        config.layers[layer].bottle_configs[0].kerner_size=1;
        config.layers[layer].bottle_configs[0].bn_size = bn_size;
        config.layers[layer].bottle_configs[1].kerner_size=3;
        config.layers[layer].bottle_configs[1].bn_size = bn_size;
        config.layers[layer].bottle_configs[2].kerner_size=1;
        config.layers[layer].bottle_configs[2].bn_size = bn_size*4;
    }
}

void init_config_resnet101()
{
    config.blocks.resize(4);
    config.blocks[0]=3;
    config.blocks[1]=4;
    config.blocks[2]=23;
    config.blocks[3]=3;

    config.layers.resize(4);

    int bn_size = 64;
    for(int layer = 0; layer< 4;layer++)
    {
        config.layers[layer].bottle_configs.resize(3);
        
        if(layer == 0)
        {
            bn_size = 64;
        }
        else
        {
            bn_size = bn_size*2;
        }
        config.layers[layer].bottle_configs[0].kerner_size=1;
        config.layers[layer].bottle_configs[0].bn_size = bn_size;
        config.layers[layer].bottle_configs[1].kerner_size=3;
        config.layers[layer].bottle_configs[1].bn_size = bn_size;
        config.layers[layer].bottle_configs[2].kerner_size=1;
        config.layers[layer].bottle_configs[2].bn_size = bn_size*4;
    }
}


void log_tensor(char * desc,struct ggml_tensor * t)
{
    printf("%s  shape:  %3d x %3d x %4d x %3d\n", desc,(int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
}

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
        ctx_size += 64*3*7*7*ggml_type_sizef(GGML_TYPE_F32);//conv1.weight
        ctx_size += 64*ggml_type_sizef(GGML_TYPE_F32);//bn1.weight
        ctx_size += 64*ggml_type_sizef(GGML_TYPE_F32);//bn1.bias
        ctx_size += 64*ggml_type_sizef(GGML_TYPE_F32);//bn1.running_mean 
        ctx_size += 64*ggml_type_sizef(GGML_TYPE_F32);//bn1.running_var

        //layer1
        int last_bn_size = 64;
        for(int layer = 0; layer < config.layers.size(); layer++)
        {
            for(int b = 0; b < config.blocks[layer];b++)
            {
                if(b == 0 )
                {
                    ctx_size += config.layers[layer].bottle_configs[0].bn_size*last_bn_size*config.layers[layer].bottle_configs[0].kerner_size*config.layers[layer].bottle_configs[0].kerner_size*ggml_type_sizef(GGML_TYPE_F32);
                }
                else
                {
                    ctx_size += config.layers[layer].bottle_configs[0].bn_size*config.layers[layer].bottle_configs[0].bn_size*config.layers[layer].bottle_configs[0].kerner_size*config.layers[layer].bottle_configs[0].kerner_size*ggml_type_sizef(GGML_TYPE_F32);
                }
                
                ctx_size += config.layers[layer].bottle_configs[0].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.weight 
                ctx_size += config.layers[layer].bottle_configs[0].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.bias
                ctx_size += config.layers[layer].bottle_configs[0].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.running_mean
                ctx_size += config.layers[layer].bottle_configs[0].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.running_var
        
                ctx_size += config.layers[layer].bottle_configs[1].bn_size*config.layers[layer].bottle_configs[1].bn_size*config.layers[layer].bottle_configs[1].kerner_size*config.layers[layer].bottle_configs[1].kerner_size*ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += config.layers[layer].bottle_configs[1].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.weight 
                ctx_size += config.layers[layer].bottle_configs[1].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.bias
                ctx_size += config.layers[layer].bottle_configs[1].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.running_mean
                ctx_size += config.layers[layer].bottle_configs[1].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.running_var
        
                ctx_size += config.layers[layer].bottle_configs[2].bn_size*config.layers[layer].bottle_configs[1].bn_size*config.layers[layer].bottle_configs[2].kerner_size*config.layers[layer].bottle_configs[2].kerner_size*ggml_type_sizef(GGML_TYPE_F32);
                ctx_size += config.layers[layer].bottle_configs[2].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.weight 
                ctx_size += config.layers[layer].bottle_configs[2].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.bias
                ctx_size += config.layers[layer].bottle_configs[2].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.running_mean
                ctx_size += config.layers[layer].bottle_configs[2].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.bn1.running_var


                if(b == 0)
                {
                    ctx_size += config.layers[layer].bottle_configs[2].bn_size*config.layers[layer].bottle_configs[1].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.downsample.0.weight
                    ctx_size += config.layers[layer].bottle_configs[2].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.downsample.1.weight
                    ctx_size += config.layers[layer].bottle_configs[2].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.downsample.1.bias
                    ctx_size += config.layers[layer].bottle_configs[2].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.downsample.1.running_mean 
                    ctx_size += config.layers[layer].bottle_configs[2].bn_size*ggml_type_sizef(GGML_TYPE_F32);//layer1.0.downsample.1.running_var 
                }
            }

            last_bn_size = config.layers[layer].bottle_configs[2].bn_size;         
        }
       
        ctx_size +=1000*2048*ggml_type_sizef(GGML_TYPE_F32);//fc.weight
        ctx_size +=1000*ggml_type_sizef(GGML_TYPE_F32);//fc.bias

    }   
    ctx_size = 25610205*4;
    std::cout << "parameters = "<<ctx_size/4<<std::endl;
    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size + 1024*1024*1000,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    int layer_count = config.blocks.size();
    model.layers.resize(layer_count);


    for(int i = 0; i < layer_count; i++)
    {
        model.layers[i].blocks.resize(config.blocks[i]);
        model.layers[i].blocks[0].have_dowsample = true;
    }

    //read conv1
   {
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "read conv1 n_dims = "<<n_dims<<std::endl;
        int32_t ne_weight[4] = {1,1,1,1};
        for(int j = 0; j < n_dims; j++)
        {
            fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
            std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
        }
        model.conv1.weight = ggml_new_tensor_4d(model.ctx,GGML_TYPE_F32,ne_weight[0],ne_weight[1],ne_weight[2],ne_weight[3]);

        //std::cout << "conv1 data size = " << ggml_nbytes(model.conv2d_layers[i].weight)<<std::endl;
        fin.read(reinterpret_cast<char*>(model.conv1.weight->data),ggml_nbytes(model.conv1.weight));

        model.conv1.stride=2;
        model.conv1.padding=3;
   }

    //read bn1
    
    {
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "read bn1.weight n_dims = "<<n_dims<<std::endl;
        int32_t ne_weight[1] = {1};
        for(int j = 0; j < n_dims; j++)
        {
            fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
            std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
        }

        model.bn1.weight = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
        fin.read(reinterpret_cast<char*>(model.bn1.weight->data),ggml_nbytes(model.bn1.weight));
    }

    {
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "read bn1.bias n_dims = "<<n_dims<<std::endl;
        int32_t ne_weight[1] = {1};
        for(int j = 0; j < n_dims; j++)
        {
            fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
            std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
        }

        model.bn1.bias = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
        fin.read(reinterpret_cast<char*>(model.bn1.bias->data),ggml_nbytes(model.bn1.bias));

    }
    {
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "read bn1.mean n_dims = "<<n_dims<<std::endl;
        int32_t ne_weight[1] = {1};
        for(int j = 0; j < n_dims; j++)
        {
            fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
            std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
        }

        model.bn1.mean = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
        fin.read(reinterpret_cast<char*>(model.bn1.mean->data),ggml_nbytes(model.bn1.mean));

    }
    {
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "read bn1.var n_dims = "<<n_dims<<std::endl;
        int32_t ne_weight[1] = {1};
        for(int j = 0; j < n_dims; j++)
        {
            fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
            std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
        }

        model.bn1.var = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
        fin.read(reinterpret_cast<char*>(model.bn1.var->data),ggml_nbytes(model.bn1.var));
    }

    //bn1.num_batches_tracked 
    {
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "bn1.num_batches_tracked n_dims = "<<n_dims<<std::endl;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    }
    for(int i = 0; i < layer_count; i++)
    {
        std::cout << "*****read layer "<<i+1<<" ******"<<std::endl;
        for(int b = 0; b < config.blocks[i]; b++)
        {
            //conv1
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" conv1.weight n_dims = "<<n_dims<<std::endl;
                int32_t ne_weight[4] = {1,1,1,1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }
                model.layers[i].blocks[b].conv1.weight = ggml_new_tensor_4d(model.ctx,GGML_TYPE_F32,1,1,ne_weight[0],ne_weight[1]);

                std::cout << "conv1 data size = " << ggml_nbytes(model.layers[i].blocks[b].conv1.weight)<<std::endl;
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].conv1.weight->data),ggml_nbytes(model.layers[i].blocks[b].conv1.weight));

                // if(b == 0 && i>0)
                // {
                //     model.layers[i].blocks[b].conv1.stride = 2;
                //     model.layers[i].blocks[b].conv1.padding = 0;
                // }
                // else
                // {
                //     model.layers[i].blocks[b].conv1.stride = 1;
                //     model.layers[i].blocks[b].conv1.padding = 0;
                // }

                model.layers[i].blocks[b].conv1.stride = 1;
                model.layers[i].blocks[b].conv1.padding = 0;
                
            }

            //bn1
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn1.weight n_dims = "<<n_dims<<std::endl;
                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn1.weight = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn1.weight->data),ggml_nbytes(model.layers[i].blocks[b].bn1.weight));
            }

            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn1.bias n_dims = "<<n_dims<<std::endl;
                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn1.bias = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn1.bias ->data),ggml_nbytes(model.layers[i].blocks[b].bn1.bias ));

            }
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn1.mean n_dims = "<<n_dims<<std::endl;

                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn1.mean = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn1.mean->data),ggml_nbytes(model.layers[i].blocks[b].bn1.mean));

            }
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn1.var n_dims = "<<n_dims<<std::endl;

                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn1.var = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn1.var->data),ggml_nbytes(model.layers[i].blocks[b].bn1.var));
            }

            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "bn1.num_batches_tracked n_dims = "<<n_dims<<std::endl;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            }
            
            //conv2
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" conv2.weight n_dims = "<<n_dims<<std::endl;
                int32_t ne_weight[4] = {1,1,1,1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }
                model.layers[i].blocks[b].conv2.weight = ggml_new_tensor_4d(model.ctx,GGML_TYPE_F32,ne_weight[0],ne_weight[1],ne_weight[2],ne_weight[3]);

                //std::cout << "conv1 data size = " << ggml_nbytes(model.conv2d_layers[i].weight)<<std::endl;
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].conv2.weight->data),ggml_nbytes(model.layers[i].blocks[b].conv2.weight));
                

                if(b == 0 && i>0)
                {
                    model.layers[i].blocks[b].conv2.stride = 2;
                    model.layers[i].blocks[b].conv2.padding = 1;
                }
                else
                {
                    model.layers[i].blocks[b].conv2.stride = 1;
                    model.layers[i].blocks[b].conv2.padding = 1;
                }


                // model.layers[i].blocks[b].conv2.stride = 1;
                // model.layers[i].blocks[b].conv2.padding = 1;

            }

            //bn2
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn2.weight n_dims = "<<n_dims<<std::endl;
                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn2.weight = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn2.weight->data),ggml_nbytes(model.layers[i].blocks[b].bn2.weight));
            }

            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn2.bias n_dims = "<<n_dims<<std::endl;
                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn2.bias = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn2.bias ->data),ggml_nbytes(model.layers[i].blocks[b].bn2.bias ));

            }
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn2.mean n_dims = "<<n_dims<<std::endl;

                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn2.mean = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn2.mean->data),ggml_nbytes(model.layers[i].blocks[b].bn2.mean));

            }
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn2.var n_dims = "<<n_dims<<std::endl;

                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn2.var = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn2.var->data),ggml_nbytes(model.layers[i].blocks[b].bn2.var));
            }


            {
                int32_t n_dims;
                int32_t value;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "bn2.num_batches_tracked n_dims = "<<n_dims<<std::endl;
                fin.read(reinterpret_cast<char *>(&value), sizeof(value));
            }


            //conv3
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" conv3.weight n_dims = "<<n_dims<<std::endl;
                int32_t ne_weight[4] = {1,1,1,1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }
                model.layers[i].blocks[b].conv3.weight = ggml_new_tensor_4d(model.ctx,GGML_TYPE_F32,1,1,ne_weight[0],ne_weight[1]);

                //std::cout << "conv1 data size = " << ggml_nbytes(model.conv2d_layers[i].weight)<<std::endl;
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].conv3.weight->data),ggml_nbytes(model.layers[i].blocks[b].conv3.weight));
                model.layers[i].blocks[b].conv3.stride = 1;
                model.layers[i].blocks[b].conv3.padding = 0;
            }

            //bn3
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn3.weight n_dims = "<<n_dims<<std::endl;
                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn3.weight = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn3.weight->data),ggml_nbytes(model.layers[i].blocks[b].bn3.weight));
            }

            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn3.bias n_dims = "<<n_dims<<std::endl;
                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn3.bias = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn3.bias ->data),ggml_nbytes(model.layers[i].blocks[b].bn3.bias ));

            }
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn3.mean n_dims = "<<n_dims<<std::endl;

                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn3.mean = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn3.mean->data),ggml_nbytes(model.layers[i].blocks[b].bn3.mean));

            }
            {
                int32_t n_dims;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "read layer "<<i+1<<" block "<<b<<" bn3.var n_dims = "<<n_dims<<std::endl;

                int32_t ne_weight[1] = {1};
                for(int j = 0; j < n_dims; j++)
                {
                    fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                    std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                }

                model.layers[i].blocks[b].bn3.var = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].bn3.var->data),ggml_nbytes(model.layers[i].blocks[b].bn3.var));
            }


            {
                int32_t n_dims;
                int32_t value;
                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                std::cout << "bn3.num_batches_tracked n_dims = "<<n_dims<<std::endl;
                fin.read(reinterpret_cast<char *>(&value), sizeof(value));
            }


            std::cout << "have_dowsample = "<<model.layers[i].blocks[b].have_dowsample<<std::endl;

            if(model.layers[i].blocks[b].have_dowsample)
            {
                //conv1
                {
                    int32_t n_dims;
                    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                    std::cout << "read layer "<<i+1<<" block "<<b<<" downsample.0.weight n_dims = "<<n_dims<<std::endl;
                    int32_t ne_weight[2] = {1,1};
                    for(int j = 0; j < n_dims; j++)
                    {
                        fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                        std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                    }
                    model.layers[i].blocks[b].dowsample.conv.weight = ggml_new_tensor_4d(model.ctx,GGML_TYPE_F32,1,1,ne_weight[0],ne_weight[1]);

                    //std::cout << "conv1 data size = " << ggml_nbytes(model.layers[i].blocks[b].conv1.weight)<<std::endl;
                    fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].dowsample.conv.weight ->data),ggml_nbytes(model.layers[i].blocks[b].dowsample.conv.weight));
                    if(i == 0)
                        model.layers[i].blocks[b].dowsample.conv.stride = 1;
                    else
                        model.layers[i].blocks[b].dowsample.conv.stride = 2;

                    model.layers[i].blocks[b].dowsample.conv.padding = 0;
                }

                //bn
                {
                    int32_t n_dims;
                    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                    std::cout << "read layer "<<i+1<<" block "<<b<<" downsample.1.weight n_dims = "<<n_dims<<std::endl;
                    int32_t ne_weight[1] = {1};
                    for(int j = 0; j < n_dims; j++)
                    {
                        fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                        std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                    }

                    model.layers[i].blocks[b].dowsample.bn.weight = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                    fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].dowsample.bn.weight->data),ggml_nbytes(model.layers[i].blocks[b].dowsample.bn.weight));
                }

                {
                    int32_t n_dims;
                    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                    std::cout << "read layer "<<i+1<<" block "<<b<<" downsample.1.bias n_dims = "<<n_dims<<std::endl;
                    int32_t ne_weight[1] = {1};
                    for(int j = 0; j < n_dims; j++)
                    {
                        fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                        std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                    }

                    model.layers[i].blocks[b].dowsample.bn.bias = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                    fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].dowsample.bn.bias ->data),ggml_nbytes(model.layers[i].blocks[b].dowsample.bn.bias ));

                }
                {
                    int32_t n_dims;
                    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                    std::cout << "read layer "<<i+1<<" block "<<b<<" downsample.1.mean n_dims = "<<n_dims<<std::endl;

                    int32_t ne_weight[1] = {1};
                    for(int j = 0; j < n_dims; j++)
                    {
                        fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                        std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                    }

                    model.layers[i].blocks[b].dowsample.bn.mean = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                    fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].dowsample.bn.mean->data),ggml_nbytes(model.layers[i].blocks[b].dowsample.bn.mean));

                }
                {
                    int32_t n_dims;
                    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                    std::cout << "read layer "<<i+1<<" block "<<b<<" downsample.1.var n_dims = "<<n_dims<<std::endl;

                    int32_t ne_weight[1] = {1};
                    for(int j = 0; j < n_dims; j++)
                    {
                        fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
                        std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
                    }

                    model.layers[i].blocks[b].dowsample.bn.var = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_weight[0]);
                    fin.read(reinterpret_cast<char*>(model.layers[i].blocks[b].dowsample.bn.var->data),ggml_nbytes(model.layers[i].blocks[b].dowsample.bn.var));
                }


                {
                    int32_t n_dims;
                    int32_t value;
                    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                    std::cout << "bn1.num_batches_tracked n_dims = "<<n_dims<<std::endl;
                    fin.read(reinterpret_cast<char *>(&value), sizeof(value));
                }

            }
        }
        std::cout << std::endl;

    }

    //fc

    {//weight
        // Read dimensions
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "read fc  weight n_dims = "<<n_dims<<std::endl;
        int32_t ne_weight[2] = {1,1};
        for(int j = 0; j < n_dims; j++)
        {
            fin.read(reinterpret_cast<char*>(&ne_weight[j]),sizeof(ne_weight[j]));
            std::cout << "n_dims["<<j<<"]= "<<ne_weight[j]<<std::endl;
        }
        model.fc.weight = ggml_new_tensor_2d(model.ctx,GGML_TYPE_F32,ne_weight[0],ne_weight[1]);

        //std::cout << "fc"<<i+1<<" weight data size = " << ggml_nbytes(model.fc_layers[i].weight)<<std::endl;
        fin.read(reinterpret_cast<char*>(model.fc.weight->data),ggml_nbytes(model.fc.weight));
    }

    {//bias
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        std::cout << "read fc bias dims = "<<n_dims<<std::endl;
        
        int32_t ne_bias[1] = { 1 };
        for (int j = 0; j < 1; ++j) {
            fin.read(reinterpret_cast<char *>(&ne_bias[j]), sizeof(ne_bias[j]));
        } 
        std::cout << "read fc bias dim value = "<<ne_bias[0]<<std::endl;
        model.fc.bias = ggml_new_tensor_1d(model.ctx,GGML_TYPE_F32,ne_bias[0]);
        fin.read(reinterpret_cast<char*>(model.fc.bias->data),ggml_nbytes(model.fc.bias));
    }
    return true;
}


static ggml_tensor * apply_conv2d(ggml_context * ctx, ggml_tensor * input, const conv2d_layer & layer)
{
    struct ggml_tensor * result = ggml_conv_2d(ctx, layer.weight, input, layer.stride, layer.stride, layer.padding, layer.padding, 1, 1);
    if(layer.have_bias)
    {
        result = ggml_add(ctx, result, ggml_repeat(ctx, layer.bias, result));
    }
    
    return result;
}

static ggml_tensor * apply_bn2d(ggml_context * ctx, ggml_tensor * input, const bn_layer & layer)
{
    struct ggml_tensor * batch_mean = ggml_batch_repeat(ctx, layer.mean, input);
    //log_tensor("model.bn1.mean batched ",batch_mean);

    struct ggml_tensor * sub_result = ggml_sub(ctx, input, batch_mean);

    struct ggml_tensor * sqrt_result = ggml_sqrt(ctx,layer.var);

    struct ggml_tensor * div_result = ggml_div(ctx,sub_result,ggml_batch_repeat(ctx, sqrt_result, sub_result));
    
    struct ggml_tensor * result = ggml_mul(ctx,div_result,ggml_batch_repeat(ctx,layer.weight, div_result));

    result = ggml_add(ctx, result, ggml_batch_repeat(ctx, layer.bias, result));

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



    static size_t buf_size = 60000000 * sizeof(float) * 4*2;
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
    log_tensor("model.input",input);


    struct ggml_tensor * result = apply_conv2d(ctx0, input, model.conv1);
    //print_shape(0, result);
    log_tensor("model.conv1",result);



    result = apply_bn2d(ctx0, result, model.bn1);
    log_tensor("model.bn1.mean",model.bn1.mean);
    log_tensor("model.bn1",result);



    result = ggml_relu(ctx0, result);
    log_tensor("model.relu",result);
    
  
    
    //print_shape(0, result);
    result = ggml_pool_2d(ctx0, result, GGML_OP_POOL_MAX, 3, 3, 2, 2, 1, 1); 
    log_tensor("model.maxpool",result);
    

    struct ggml_tensor * result_log = nullptr;    
    struct ggml_tensor * result_layer1; 
    for(int layer = 0; layer < config.blocks.size(); layer++)
    {
        // if(layer ==3)
        //     break;

        for(int block = 0; block < config.blocks[layer]; block++)
        {
            struct ggml_tensor * x = result;

            result = apply_conv2d(ctx0, result, model.layers[layer].blocks[block].conv1);


            result = apply_bn2d(ctx0, result, model.layers[layer].blocks[block].bn1);

            result = ggml_relu(ctx0, result);

            result = apply_conv2d(ctx0, result, model.layers[layer].blocks[block].conv2);

            result = apply_bn2d(ctx0, result, model.layers[layer].blocks[block].bn2);

            result = ggml_relu(ctx0, result);

            result = apply_conv2d(ctx0, result, model.layers[layer].blocks[block].conv3);

            result = apply_bn2d(ctx0, result, model.layers[layer].blocks[block].bn3);


            // if(layer ==1 && block ==0)
            // {
            //     result_log = result;
            //     break;
            // }
            struct ggml_tensor *t = result;
            printf("Layer %d block %d bn2 output shape:  %3d x %3d x %4d x %3d\n", layer,block, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);

            struct ggml_tensor *downsample_result;
            if(model.layers[layer].blocks[block].have_dowsample)
            {
                downsample_result = apply_conv2d(ctx0, x, model.layers[layer].blocks[block].dowsample.conv);
                downsample_result = apply_bn2d(ctx0, downsample_result, model.layers[layer].blocks[block].dowsample.bn);
                //printf("Layer %d block %d bn2 output shape:  %3d x %3d x %4d x %3d\n", layer,block, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
            }
            else
            {
                downsample_result = x;
            }

            result  = ggml_add(ctx0,result,downsample_result);

            result = ggml_relu(ctx0,result);
        }

        // if(layer == 3)
        // {
        //     result_layer1 = result;
        // }

    }

        

    // result_log = result;
    // ggml_build_forward_expand(gf, result);
    // ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    // {
    //     struct ggml_tensor * log = result_log;
    //     const float *data= ggml_get_data_f32(log);


    //     std::cout << "result_log data:"<<std::endl;
    //     for(int i = 0; i < ggml_nelements(log); i++)
    //     {
    //         std::cout << data[i]<<std::endl;
    //     }

    // }




    result = ggml_pool_2d(ctx0, result, GGML_OP_POOL_AVG, 7, 7, 7,7, 0, 0);   

    log_tensor("model.avgpool",result);
    struct ggml_tensor * view = ggml_reshape_1d(ctx0,result, result->ne[0]*result->ne[1]*result->ne[2]*result->ne[3]);


    result = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc.weight, view),                model.fc.bias);

    log_tensor("model.fc",result);




    struct ggml_tensor *t = result;

    // // soft max
    ggml_tensor * probs = ggml_soft_max(ctx0, result);
    log_tensor("model.softmax",probs);
    ggml_set_name(probs, "probs");

    // //build / export / run the computation graph
    ggml_build_forward_expand(gf, probs);
    //ggml_build_forward_expand(gf, view);
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    // //ggml_graph_print   (&gf);
    // ggml_graph_dump_dot(gf, NULL, "alexnet.dot");

    // {
    //     struct ggml_tensor * log = result_log;
    //     const float *data= ggml_get_data_f32(log);


    //     std::cout << "result_log data:"<<std::endl;
    //     for(int i = 0; i < ggml_nelements(log); i++)
    //     {
    //         std::cout << data[i]<<std::endl;
    //     }

    // }


    const float * probs_data = ggml_get_data_f32(probs);



    const int prediction = std::max_element(probs_data, probs_data + 1000) - probs_data;


    // std::cout <<"probs_data:"<<std::endl;

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


    std::string filename = "/data/hanzt1/he/codes/engine/examples/resnet50/imagenet_classes.txt"; // 标签文件的路径
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

    // ggml_free(ctx0);

    // std::cout <<*std::max_element(probs_data, probs_data + 1000)<<std::endl;

    return prediction;
    
}


int main(int argc,char* argv[])
{
    std::cout << "resnet50 main:"<<std::endl;
    init_config_resnet50();
    init_config_resnet101();
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