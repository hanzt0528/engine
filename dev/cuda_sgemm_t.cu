#include <iostream>
#include <cuda_runtime.h>

#define OFFSET(row,col,ld) ((ld)*(row)+(col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>((&pointer))[0])

__global__ void native_sgemm(float *a,float *b,float*c,int M,int N,int K)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if(m< M && n < N)
    {
        float psum = 0.0;
        #pragma unroll
        for(int k = 0;k < K;k++)
        {
            psum += a[OFFSET(m,k,K)]*b[OFFSET(k,n,N)];
        }
        c[OFFSET(m,n,N)] = psum;
    }
}


__global__ void native_sgemm_v2(float *a,float *b,float*c,int M,int N,int K)
{
 
    const int BM=128;
    const int BN=128;
    const int BK=8;
    const int TM=8;
    const int TN=8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty*blockDim.x+tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid>>1;
    int load_a_smem_k = (tid&1)<<2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid&31)<<2;

    int load_a_gmem_m=by*BM+load_a_smem_m;
    int load_b_gmem_n=bx*BN+load_b_smem_n;

    for(int bk=0; bk<(K+BK-1)/BK;bk++)
    {
        int load_a_gmem_k = bk*BK+load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m,load_a_gmem_k,K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        int load_b_gmem_k = bk*BK +load_b_smem_k;
        int load_b_gmem_addr=OFFSET(load_b_gmem_k,load_b_gmem_n,N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n])=FLOAT4(b[load_b_gmem_addr]);
        __syncthreads();
        #pragma unroll
        for(int k = 0; k < BK; k++)
        {
            #pragma unroll
            for(int m = 0; m < TM; m++)
            {
                #pragma unroll
                for(int n = 0; n < TN; n++)
                {
                    int comp_a_smem_m = ty*TM+m;
                    int comp_b_smem_n = tx*TN+n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }
    #pragma unroll
    for(int i = 0; i < TM; i++)
    {
        int store_c_gmem_m = by*BM + ty*TM +i;
        #pragma unroll
        for(int j = 0; j < TN; j+=4)
        {
            int store_c_gmem_n = bx*BN + tx*TN +j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m,store_c_gmem_n,N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}


int main(int argc,char * argv[])
{
    std::cout<<"gemm_t"<<std::endl;

    int BM =128;
    int BN =128;

    int TN=8;
    int TM=8;
    int BK=8;

    int M=512;
    int N=512;
    int K=512;

    float a[M*K] = {0};
    float b[K*N] = {0};
    float c[M*N] = {0};

    float *a_d = nullptr;
    float *b_d = nullptr;
    float *c_d = nullptr;


    for(int i = 0; i < M*K; i++)
    {
        a[i] = 1;
    }
    
    for(int i = 0; i < N*K; i++)
    {
        b[i] = 1;
    }

    cudaMalloc((void**)&a_d,sizeof(float)*M*K);
    cudaMalloc((void**)&b_d,sizeof(float)*K*N);
    cudaMalloc((void**)&c_d,sizeof(float)*M*N);
    
    cudaMemcpy(a_d,a,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,sizeof(float)*N*K,cudaMemcpyHostToDevice);

    
    dim3 gridDim((N+BN-1)/BN,(M+BM-1)/BM);
    dim3 blockDim(BN/TN,BM/TM);
    cudaEvent_t start,end;
    float this_msec = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    //native_sgemm<<<gridDim,blockDim,0>>>(a_d,b_d,c_d,M,N,K);
    native_sgemm_v2<<<gridDim,blockDim,0>>>(a_d,b_d,c_d,M,N,K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&this_msec,start,end);

    std::cout << "ElapsedTime = "<<this_msec<<std::endl;

    cudaMemcpy(c,c_d,sizeof(float)*M*N,cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++)
    {
        std::cout << c[i]<<std::endl;
    }

    return 0;
}