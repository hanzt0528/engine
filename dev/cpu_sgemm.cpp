#include <iostream>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

int main()
{
    float a[] = {1,1,1,1};
    float b[] = {2,1,1,1};
    float c[] = {1,1,1,1};

    cpuSgemm(a,b,c,2,2,2);

    int length = sizeof(c)/sizeof(float);
    for(int i = 0; i < length; i++)
    {
        std::cout << c[i]<<std::endl;
    }

    return 0;
}