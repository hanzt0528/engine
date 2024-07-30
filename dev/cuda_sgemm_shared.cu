#include <iostream>
#include <cuda_runtime.h>

#define OFFSET(row,col,ld) ((row)*(ld)+(col))

#define FLOAT4(pointer)(reinterpret_cast<float4*>(&(pointer))[0])
//#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

//#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


__global__ void cuda_sgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];//[128,8]，申请共享内存，同一个block内的所有线程都可以访问。s_a有128行，有8列。
    __shared__ float s_b[BK][BN];//[8,128]，同上，s_b有8行，128列

    //一个线程处理的C中的结果数据块r_c[8,8]，这个数据块是block中的一个子块，所有数据子块完成，整个block完成。
    float r_c[TM][TN] = {0.0};

    //一个block块16x16个线程，为了将数据从全局内存加载到s_a中，每个线程加载4（(128x8)/(16x16)）个float数据
    int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
    int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

    int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

    //遍历A和B矩阵的K维度，步长为BK.循环16次，完成一个C中大小为[BM，BN]块的计算。
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {

        //读取全局内存A,B数据到共享内存中
        int load_a_gmem_k = bk * BK + load_a_smem_k;   // global col of a
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;   // global row of b
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);
        __syncthreads();//同步一个线程块内的所有线程（比如16x16=256个线程）,256个线程读取256个线程需要的所有数据，共读取A中128*8个数据，每个线程读取A中的4个float，读取B同理。

        //每个线程计算自己对应的r_c数据
        #pragma unroll
        for (int k = 0; k < BK; k++) 
        {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        //同步一个线程块内的所有线程（比如16x16=256个线程）
        //每个线程计算完自己的r_c后，同步进行下一个bk块的计算
        __syncthreads();
    }

    //一个线程处理的数据块TMxTN写入结果矩阵C全局内存中。
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}



int main(int argc,char* argv[])
{
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 256;

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

    int BM = 128;
    int BN = 128;
    int TM = 8;
    int TN = 8;
    int TK = 8;

    dim3 gridDim((N+BN-1)/BN,(M+BM-1)/BM);
    dim3 blockDim(BN/TN,BM/TM);

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
