#include <iostream>
#include <cuda_runtime.h>

#define OFFSET(row,col,ld) ((row)*(ld)+(col))

__global__ void cuda_sgemm(float* a,float *b,float *c,int M,int N,int K)
{
    int n = threadIdx.y + blockIdx.y*blockDim.y;
    int m = threadIdx.x + blockIdx.x*blockDim.x;

    if(n < N && m < M)
    {
        float psum = 0.0;
        for(int k = 0; k < K; k++)
        {
            psum+=a[OFFSET(m,k,K)]*b[OFFSET(k,n,N)];
        }
        c[OFFSET(m,n,N)]= psum;
    }

}

int main(int argc,char* argv[])
{
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;

    float a[M*K] = {1};
    float b[N*K] = {1};
    float c[M*N] = {1};

    for(int i = 0; i < M*K; i ++)
    {
        a[i] = 1;
    }

    for(int i = 0; i < N*K; i ++)
    {
        b[i] = 1;
    }


    float *a_d = nullptr;
    float *b_d = nullptr;
    float *c_d = nullptr;
    cudaMalloc((void**)&a_d,sizeof(float)*M*K);
    cudaMalloc((void**)&b_d,sizeof(float)*N*K);
    cudaMalloc((void**)&c_d,sizeof(float)*N*M);

    cudaMemcpy(a_d,a,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,sizeof(float)*N*K,cudaMemcpyHostToDevice);

    int BM = 32;
    int BN = 32;

    dim3 gridDim((N+BN-1)/BN,(M+BM-1)/BM);
    dim3 blockDim(BN,BM);
    cuda_sgemm<<<gridDim,blockDim,0>>>(a_d,b_d,c_d,M,N,K);

    cudaMemcpy(c,c_d,sizeof(float)*N*M,cudaMemcpyDeviceToHost);

    //log out value
    for(int i = 0; i < 10; i ++)
    {
        std::cout << c[i] << std::endl;
    }
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaDeviceReset();
    std::cout << "main"<<std::endl;
}
