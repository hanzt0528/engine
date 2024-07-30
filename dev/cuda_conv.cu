// 在人工智能的深度学习领域，特别是在卷积神经网络（CNN）中，2D卷积核（也称为滤波器或权重矩阵）是用来从输入数据（通常是图像）中提取特征的。2D卷积核的维度对于理解其如何工作至关重要。以下是2D卷积核维度的组成部分：

// 高度（Height）：这是卷积核在垂直方向上的尺寸。例如，如果一个卷积核的高度是3，那么它在垂直方向上会覆盖输入图像的3行像素。

// 宽度（Width）：这是卷积核在水平方向上的尺寸。与高度类似，如果卷积核的宽度是3，它在水平方向上会覆盖输入图像的3列像素。

// 输入通道数（Input Channels）：这是卷积核需要匹配的输入数据的通道数。对于彩色图像，通常有3个通道（红色、绿色和蓝色），因此对应的卷积核也需要有3个通道。

// 输出通道数（Output Channels）：这是卷积核生成的特征图的数量。每个输出通道都有自己的一组卷积核，用于提取不同的特征。

// 将这些维度组合起来，我们可以得到一个完整的2D卷积核的维度表示，通常写作：[height, width, input_channels, output_channels]。

// 例如，一个常见的卷积核维度可能是[3, 3, 3, 64]，这意味着：

// 卷积核的高度和宽度都是3。
// 输入图像有3个通道（例如，RGB彩色图像）。
// 这个卷积核会生成64个不同的特征图。
// 卷积核维度的作用：
// 特征提取：卷积核对输入数据进行卷积操作，生成特征图，每个特征图对应一个输出通道，捕捉输入数据中的不同特征。

// 参数共享：由于卷积核在输入图像上滑动时使用相同的权重，这减少了模型的参数数量，使得网络更易于训练。

// 局部感知：卷积核的尺寸（高度和宽度）决定了它在输入图像上覆盖的局部区域的大小，从而决定了它能够捕捉的特征的尺度。

// 深度：通过堆叠多个卷积层，每个层级可以捕捉更复杂的特征，构建起一个从简单到复杂的特征提取网络。

// 选择卷积核尺寸的考虑因素：
// 感受野：卷积核尺寸越大，其感受野也越大，能够捕捉更大规模的特征，但同时参数数量也会增加。

// 计算复杂度：较大的卷积核会增加计算量，可能导致训练时间变长。

// 过拟合风险：较大的卷积核和更多的输出通道可能会增加过拟合的风险，尤其是在数据量较小的情况下。

// 内存和硬件限制：卷积核的尺寸和数量会影响模型的内存占用和计算需求，需要根据可用的硬件资源进行权衡。

// 在设计CNN时，卷积核的尺寸选择是一个重要的决策，需要根据具体任务和数据集的特点进行调整。
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define OFFSET(row,col,ld) ((row)*(ld) + (col))

void convolution(float *in,float *k,float *out,const int in_c,const int in_h,const int in_w,const int out_c,const int out_h,const int out_w,const int k_h,const int k_w)
{
    float val = 0.0;
    int out_pos = 0,in_pos=0,k_pos = 0;
    for(int oc = 0; oc < out_c; oc++)
    {
        for(int i = 0; i < out_h; i++)
        {
            for(int j = 0; j < out_w;j++)
            {
                val = 0;
                out_pos = oc*out_h*out_w + OFFSET(i,j,out_w);
                for(int ic = 0; ic < in_c; ic++)
                {
                    for(int ki = 0; ki < k_h;ki++)
                    {
                        for(int kj = 0; kj < k_w; kj++)
                        {
                            in_pos = ic*in_w*in_h + OFFSET(i+ki,j+kj,in_w);
                            k_pos = oc*in_c*k_h*k_w + ic*k_h*k_w + OFFSET(ki,kj,k_w);
                            val += in[in_pos]*k[k_pos];
                        }
                    }
                }
                out[out_pos] = val;
            }
        }
    }
}

__global__ void cuda_convolution(float *in,float *k,float *out,const int in_c,const int in_h,const int in_w,const int out_c,const int out_h,const int out_w,const int k_h,const int k_w)
{
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    const int i = blockIdx.y*blockDim.y + threadIdx.y;
      float val = 0.0;
    int out_pos = 0,in_pos=0,k_pos = 0;

    if(j < out_w && i < out_h)
    {
        for(int oc = 0; oc < out_c; oc++)
            {
                //for(int i = 0; i < out_h; i++)
                {
                    //for(int j = 0; j < out_w;j++)
                    {
                        val = 0;
                        out_pos = oc*out_h*out_w + OFFSET(i,j,out_w);
                        for(int ic = 0; ic < in_c; ic++)
                        {
                            for(int ki = 0; ki < k_h;ki++)
                            {
                                for(int kj = 0; kj < k_w; kj++)
                                {
                                    in_pos = ic*in_w*in_h + OFFSET(i+ki,j+kj,in_w);
                                    k_pos = oc*in_c*k_h*k_w + ic*k_h*k_w + OFFSET(ki,kj,k_w);
                                    val += in[in_pos]*k[k_pos];
                                }
                            }
                        }
                        out[out_pos] = val;
                    }
                }
            }

    }
  
}
int main(int argc,char* argv[])
{
    //std::cout << "main:"<<std::endl;
    const int I_C = 3;
    const int I_H = 128;
    const int I_W = 128;

    const int K_C = I_C;
    const int K_H = 3;
    const int K_W = 3;
    const int stride = 3;
    const int padding = 0;

    const int O_C = 2;
    const int O_H = (I_H - K_H + 2*padding)/stride +1;
    const int O_W = (I_W - K_W + 2*padding)/stride +1;

    //std::cout << "O_H:"<<O_H<<std::endl;
    //std::cout << "O_W:"<<O_W<<std::endl;

    float in[I_C*I_H*I_W] = {0};
    float out[O_C*O_H*O_W] = {0};
    float k[O_C*K_C*K_H*K_W] = {0};

    for(int i = 0; i < I_C*I_H*I_W; i++)
    {
        in[i] = 1.0;
    }

    for(int i = 0; i < O_C*K_C*K_H*K_W; i++)
    {
        k[i] = 1.0;
        if(i >=9 && i <=17)
        {
            k[i] = 2.0;
        }

        if(i >=36 && i <= 44)
        {
            k[i] = 3.0;
        }
    }
    //std::cout << "k:"<<std::endl;
    // for(int i = 0; i < O_C*K_C*K_H*K_W; i++)
    // {
    //     std::cout << k[i] << std::endl;
    // }
    {    
        auto start = std::chrono::high_resolution_clock::now();
        convolution(in,k,out,I_C,I_H,I_W,O_C,O_H,O_W,K_H,K_W);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        std::cout << "cpu convolution:"<<duration.count()<<std::endl;
    }
    {
        const int BM = 32;
        const int BN = 32;
        dim3 gridDim((O_W+BN -1)/BN,(O_H+BM-1)/BM);
        dim3 blockDim(BN,BM);
        float *d_in=nullptr,*d_k=nullptr,*d_out=nullptr;
        cudaMalloc((void**)&d_in,sizeof(float)*I_C*I_H*I_W);
        cudaMalloc((void**)&d_k,sizeof(float)*O_C*K_C*K_H*K_W);
        cudaMalloc((void**)&d_out,sizeof(float)*O_C*O_H*O_W);
        auto start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_in,in,sizeof(float)*I_C*I_H*I_W,cudaMemcpyHostToDevice);
        cudaMemcpy(d_k,k,sizeof(float)*O_C*K_C*K_H*K_W,cudaMemcpyHostToDevice);

        cuda_convolution<<<gridDim,blockDim,0>>>(d_in,d_k,d_out,I_C,I_H,I_W,O_C,O_H,O_W,K_H,K_W);
        
        cudaMemcpy(out,d_out,sizeof(float)*O_C*O_H*O_W,cudaMemcpyDeviceToDevice);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        std::cout << "gpu convolution:"<<duration.count()<<std::endl;

    }

    // for(int i = 0; i < O_C*O_H*O_W; i++)
    // {
    //     std::cout << out[i] << std::endl;
    // }
}
