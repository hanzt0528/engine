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