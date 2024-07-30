#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#define OFFSET(row,col,ld) ((row)*(ld) + (col))

#define FLOAT4(pointer)(reinterpret_cast<float4*>(&(pointer))[0])

void cpu_sgemm(float *a,float*b,float*c,const int M,const int N,const int K)
{
    for(int m = 0; m < M;m++)
    {
        for(int n = 0; n < N; n++)
        {
            float psum = 0.0;
            for(int k = 0; k < K;k++)
            {
                psum+=a[OFFSET(m,k,K)]*b[OFFSET(k,n,N)];
            }
            c[OFFSET(m,n,N)] = psum;
        }
        
    }
}

__global__ void cuda_sgemm(float *a,float*b,float*c,const int M,const int N,const int K)
{
    int n = blockIdx.y*blockDim.y+threadIdx.y;
    int m = blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float temp[256];
    if(n < N && m < M)
    {
        float psum = 0.0;
        for(int k = 0; k < K; k++)
        {
            psum+=a[OFFSET(m,k,K)]*b[OFFSET(k,n,N)];
        }
        c[OFFSET(m,n,N)] = psum;
    }
}

__global__ void cuda_sgemmv2(float *a,float*b,float*c,const int M,const int N,const int K)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty*blockDim.x + tx;

    //printf("bx=%d\n",bx);
    //printf("by=%d\n",by);
    //printf("tx=%d\n",tx);
    //printf("ty=%d\n",ty);
    
    const int BM = 128;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;
    const int BK = 8;

    __shared__ __device__ float s_a[BM][BK];
    __shared__ __device__ float s_b[BK][BN];

    float r_c[TM][TN];

    int load_a_smem_m = tid>>1;
    int load_a_smem_k = (tid&1)<<2;
    int load_b_smem_k = tid>>5;
    int load_b_smem_n = (tid&31) << 2;

    int load_a_gmem_m = by*BM+load_a_smem_m;
    int load_b_gmem_n = bx*BN+load_b_smem_n;


    for(int bk = 0; bk < (K+BK-1)/BK; bk++)
    {
        int load_a_gmem_k = bk*BK+load_a_smem_k;
        int load_b_gmem_k = bk*BK+load_b_smem_k;
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[OFFSET(load_a_gmem_m,load_a_gmem_k,K)]);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[OFFSET(load_b_gmem_k,load_b_gmem_n,N)]);

        __syncthreads();

        for(int m = 0; m < TM;m++)
        {
            int comp_a_m = ty*TM + m;
            for(int n = 0; n < TN;n++)
            {
                float psum = 0.0;
                int comp_b_n = tx*TN + n;
                for(int k = 0; k < BK;k++)
                {
                     r_c[m][n]+=s_a[comp_a_m][k]*s_b[k][comp_b_n];
                }
            }
        }
        __syncthreads();
    }

    for(int m = 0; m < TM; m++)
    {
        for(int n = 0; n < TN;n++)
        {
            int store_c_gmem_m = by*BM + ty*TM +m;
            int store_c_gmem_n = bx*BN + tx*TN +n;
            c[OFFSET(store_c_gmem_m,store_c_gmem_n,N)] = r_c[m][n];
        }
    }

}

int main(int argc,char* argv[])
{
    std::cout << "main" << std::endl;

    const int M = 256;
    const int N = 256;
    const int K = 32;

    float a[M*K] = {0};
    float b[N*K] = {0};
    float c[M*N] = {0};
    float *a_d = nullptr;
    float *b_d = nullptr;
    float *c_d = nullptr;

    cudaMalloc((void**)&a_d,sizeof(float)*M*K);
    cudaMalloc((void**)&b_d,sizeof(float)*N*K);
    cudaMalloc((void**)&c_d,sizeof(float)*M*N);


    for(int i = 0; i < M*K; i++)
    {
        a[i] = 1;
    }
    for(int i = 0; i < N*K; i++)
    {
        b[i] = 1;
    }
    {
        // auto c11_start = std::chrono::steady_clock::now();
        // cpu_sgemm(a,b,c,M,N,K);
        // auto c11_end = std::chrono::steady_clock::now();
        // auto c11_duration = std::chrono::duration_cast<std::chrono::microseconds>(c11_end-c11_start);
        // std::cout<<"duration = "<<static_cast<float>(c11_duration.count())<<std::endl;
    }
    {
        // const int BM = 32;
        // const int BN = 32;
        // dim3 gridDim((N+BN-1)/BN,(M+BM-1)/BM);
        // dim3 blockDim(BN,BM);

        // cudaMemcpy(a_d,a,sizeof(float)*M*K,cudaMemcpyHostToDevice);
        // cudaMemcpy(b_d,b,sizeof(float)*N*K,cudaMemcpyHostToDevice);
    

        // auto c11_start = std::chrono::steady_clock::now();
        //             cudaEvent_t start, end;
        // cudaEventCreate(&start);
        // cudaEventCreate(&end);
        // cudaEventRecord(start);


        // cuda_sgemm<<<gridDim,blockDim,0>>>(a_d,b_d,c_d,M,N,K);
        // cudaEventRecord(end);
        // cudaEventSynchronize(end);

        // auto c11_end = std::chrono::steady_clock::now();
        // auto c11_duration = std::chrono::duration_cast<std::chrono::microseconds>(c11_end-c11_start);


        // float this_msec, this_sec;
        // cudaEventElapsedTime(&this_msec, start, end);
        //  std::cout<<"cuda duration = "<<this_msec<<std::endl;
        // std::cout<<"duration = "<<static_cast<float>(c11_duration.count())<<std::endl;
        // cudaMemcpy(c,c_d,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    }
    {
        const int BM = 128;
        const int BN = 128;
        const int TM = 8;
        const int TN = 8;
        dim3 gridDim((N+BN-1)/BN,(M+BM-1)/BM);
        dim3 blockDim(BN/TN,BM/TN);

        cudaMemcpy(a_d,a,sizeof(float)*M*K,cudaMemcpyHostToDevice);
        cudaMemcpy(b_d,b,sizeof(float)*N*K,cudaMemcpyHostToDevice);
    
        auto c11_start = std::chrono::steady_clock::now();
        
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);

        cuda_sgemmv2<<<gridDim,blockDim,0>>>(a_d,b_d,c_d,M,N,K);
        
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        auto c11_end = std::chrono::steady_clock::now();
        auto c11_duration = std::chrono::duration_cast<std::chrono::microseconds>(c11_end-c11_start);


        float this_msec, this_sec;
        cudaEventElapsedTime(&this_msec, start, end);
        std::cout<<"cuda duration = "<<this_msec<<std::endl;
        std::cout<<"duration = "<<static_cast<float>(c11_duration.count())<<std::endl;
        cudaMemcpy(c,c_d,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    }
    for(int m = 0; m < M; m++)
    {
        std::cout << std::endl;
        for(int n = 0; n < N;n++)
        {
            std::cout << c[OFFSET(m,n,N)];
        }
    }
    
    return 0 ;
}