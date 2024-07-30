#include <iostream>
#include <cuda_runtime.h>
#include <thread>
#include <float.h>
//compiler : /usr/local/cuda/bin/nvcc cuda_sgemm.cu
//run: /opt/nvidia/nsight-systems/2022.4.1/bin/nsys profile --stats=true ./a.out 
#define OFFSET(row, col, ld) ((row) * (ld) + (col))


__global__ void naiveSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
    //printf("threadIdx.x = %d\n",threadIdx.x);
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}


int main()
{
    // float a[] = {1,1,1,1};
    // float b[] = {2,1,1,1};
    // float c[] = {1,1,1,1};

    const int BM = 32, BN = 32;
    const int M = 5120, N = 5120, K = 5120;
    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    int grad_x = (N + BN - 1) / BN;
    int grad_y = (M + BM - 1) / BM;

    std::cout << "grad_x = "<<grad_x<<std::endl;
    std::cout << "grad_y = "<<grad_y<<std::endl;
    


    float *a_h = (float *)malloc(M*K*sizeof(float));
    float *a_d = nullptr;
    float *b_h = (float *)malloc(N*K*sizeof(float));
    float *b_d = nullptr;
    float *c_d = nullptr;
    float *c_h = (float *)malloc(N*M*sizeof(float));

    for(int i = 0; i < M*K; i++)
    {
        a_h[i] = 1;
    }
    
    for(int i = 0; i < N*K; i++)
    {
        b_h[i] = 1;
    }

    cudaMalloc((void**)&a_d,sizeof(float)*M*K);
    cudaMalloc((void**)&b_d,sizeof(float)*N*K);
    cudaMalloc((void**)&c_d,sizeof(float)*N*M);

    cudaMemcpy(a_d,a_h,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b_h,sizeof(float)*N*K,cudaMemcpyHostToDevice);


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

        naiveSgemm<<<gridDim,blockDim,0>>>(a_d,b_d,c_d,M,N,K);

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
    cudaMemcpy(c_h,c_d,sizeof(float)*N*M,cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++)
    {
        std::cout<<c_h[i]<<std::endl;
    }

    std::cout << "cuda main"<<std::endl;
    return 0;
}