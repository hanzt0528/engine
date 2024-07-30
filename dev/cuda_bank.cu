#include <iostream>
#include <cuda_runtime.h>

__global__ void MyKernel(unsigned long long *time)
{
    __shared__ float shared[1024];
    unsigned long long startTime = clock();

    shared[threadIdx.x]++;
    unsigned long long finishTime = clock();
    //printf("block id = %d,Thread id = %d\n",blockIdx.x,threadIdx.x);

    *time=(finishTime - startTime);
}
// /usr/local/cuda/bin/nvcc cuda_bank.cu
int main(int argc,char *argv[])
{
    unsigned long long time;
    unsigned long long *d_time;
    cudaMalloc(&d_time,sizeof(unsigned long long));

    //for(int i=0; i < 10; i++)
    {
        MyKernel<<<1,32>>>(d_time);
        cudaMemcpy(&time,d_time,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
        //std::cout <<"Time: "<<(time-14)/32<<std::endl;
        std::cout <<"Time: "<<(time)<<std::endl;
    }

    
    std::cout << "main"<<std::endl;

    cudaFree(d_time);
   // _getch();
    cudaDeviceReset();
    return 0;
}