#include <iostream>
#include <float.h>
#include <cuda_runtime.h>
#include <thread>
//compiler : /usr/local/cuda/bin/nvcc cuda_sgemm2.cu
//run: /opt/nvidia/nsight-systems/2022.4.1/bin/nsys profile --stats=true ./a.out 

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void naiveSgemm_v2(
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
    const int tid = ty * blockDim.x + tx;//block内部的线程索引id

    __shared__ float s_a[BM][BK];//[128,8]，申请共享内存，同一个block内的所有线程都可以访问。
    __shared__ float s_b[BK][BN];//[8,128]，同上

    //一个线程处理的C中的结果数据块r_c[8,8]，这个数据块是block中的一个子块，所有数据子块完成，整个block完成。
    float r_c[TM][TN] = {0.0};

    //一个block块16x16个线程，为了将数据从全局内存加载到s_a中，每个线程加载4（(128x8)/(16x16)）个float数据
    int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
    int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2;  // (tid & 32) * 4, col of s_b

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
        __syncthreads();//同步一个线程块内的所有线程（比如16x16=256个线程）,256个线程共读取A中128*8个数据，每个线程读取A中的4个float，读取B同理。

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

int main()
{
    std::cout << "cuda_sgemm2.cu"<<std::endl;

    
    const int BM = 128, BN = 128, TM = 8, TN = 8;

    const int M=5120,N=5120,K=5120;
    dim3 blockDim(BN / TN, BM / TM);//(16,16)
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);//(grad.x,grad.y)



    float *a_h = (float*)malloc(M*K*sizeof(float));
    float *a_d = nullptr;


    float *b_h = (float*)malloc(N*K*sizeof(float));
    float *b_d = nullptr;

    float *c_h = (float*)malloc(N*M*sizeof(float));
    float *c_d = nullptr;

    // memset(a_h,1.0,sizeof(float)*M*K);
    // memset(b_h,1.0,sizeof(float)*N*K);
    for(int i = 0; i < M*K; i++)
    {
        a_h[i] = 1;
    }
    
    for(int i = 0; i < N*K; i++)
    {
        b_h[i] = 1;
    }

    cudaMalloc((void**)&a_d,M*K*sizeof(float));
    cudaMalloc((void**)&b_d,N*K*sizeof(float));
    cudaMalloc((void**)&c_d,N*M*sizeof(float));
    
    cudaMemcpy(a_d,a_h,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b_h,N*K*sizeof(float),cudaMemcpyHostToDevice);

    
    int repeat_count = 10;

    double max_sec = 0.0;
    double min_sec = DBL_MAX;
    double total_sec = 0.0;

    for(int i = 0; i < repeat_count; i++)
    {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);

        naiveSgemm_v2<<<gridDim,blockDim,0>>>(a_d,b_d,c_d,M,N,K);

        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float this_msec, this_sec;
        cudaEventElapsedTime(&this_msec, start, end);
        this_sec = this_msec / 1000.0;

        max_sec = max(max_sec, this_sec);
        min_sec = min(min_sec, this_sec);
        total_sec += this_sec;
    }

    double avg_sec = total_sec / repeat_count;
    double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
    
    printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);

    //std::this_thread::sleep_for(std::chrono::microseconds(1000));
    //std::this_thread::sleep_for(std::chrono::microseconds(1000));
    cudaMemcpy(c_h,c_d,sizeof(float)*M*N,cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++)
    {
        std::cout << c_h[i]<<std::endl;
    }

    return 0;
}