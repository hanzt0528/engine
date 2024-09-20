#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#define MEMCOPY_ITERATIONS 100
int main() {
    int *d_A = nullptr, *d_B = nullptr;
    const size_t memSize = 64*1024 * 1024 * sizeof(int); // 假设拷贝1MB数据

    // 初始化两个GPU设备
    int gpuCount;
    cudaGetDeviceCount(&gpuCount);
    if (gpuCount < 2) {
        std::cerr << "需要至少两个GPU来进行测试。" << std::endl;
        return -1;
    }

    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    cudaEvent_t start, stop;


    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // 在两个GPU上分配内存
    cudaSetDevice(0);
    cudaMalloc(&d_A, memSize);
    cudaSetDevice(1);
    cudaMalloc(&d_B, memSize);

    // 准备要拷贝的数据
    int *h_A = new int[memSize/ sizeof(int)];
    for (int i = 0; i < memSize/ sizeof(int); ++i) {
        h_A[i] = i;
    }
    cudaMemcpy(d_A, h_A, memSize, cudaMemcpyHostToDevice);

    // 检查是否可以在两个设备之间进行内存拷贝
    int canAccessPeer = 0;
    cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);

      // 记录开始时间
    auto start1 = std::chrono::high_resolution_clock::now();

    if (canAccessPeer) {
        // 执行内存拷贝
        cudaEventRecord(start, 0);

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
        cudaMemcpyPeer(d_B, 1, d_A, 0, memSize);
    }

  // 记录结束时间
    auto stop1 = std::chrono::high_resolution_clock::now();

    // 计算时间差
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);
    std::cout << "Time taken for cudaMemcpyPeer: " << duration.count() << " milliseconds." << std::endl;


    cudaEventRecord(stop, 0);
    cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
    printf("elapsedTimeInMs = %f\n",elapsedTimeInMs);
    // calculate bandwidth in GB/s
    double time_s = duration.count()*1.0 / 1e6;
    printf("time_s = %f\n",time_s);

    bandwidthInGBs = (2.0f * memSize * (float)MEMCOPY_ITERATIONS) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;
        
    } else {
        std::cerr << "GPU 0 和 GPU 1 之间无法进行内存拷贝。" << std::endl;
    }

     printf("   Transfer Size (Bytes)\tBandwidth(GB/s)\n");
         printf("   %u\t\t\t%s%.1f\n", memSize,
           (memSize < 10000) ? "\t" : "", bandwidthInGBs);
    // 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    delete[] h_A;

    return 0;
}