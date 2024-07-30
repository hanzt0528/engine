#include <iostream>

#define OFFSET(row,col,ld) ((row)*(ld)+(col))

void cpu_sgemm(float *a,float *b,float *c,const int M,const int N,const int K)
{
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < N; n++)
        {
            float sum = 0.0;
            for(int k = 0; k < K; k++)
            {
                sum += a[OFFSET(m,k,K)]*b[OFFSET(k,m,M)];
            }

            c[OFFSET(m,n,N)] = sum;
        }
    }
}

int main()
{
    int S = 5120;
    int M = S;
    int N = S;
    int K = S;


    float *a = (float *)malloc(M*K*sizeof(float));
    float *b = (float *)malloc(N*K*sizeof(float));
    float *c = (float *)malloc(M*N*sizeof(float));

    for(int i = 0; i < M*K; i++)
    {
        a[i] = 1;

    }
    
    for(int i = 0; i < N*K; i++)
    {
        b[i] = 1;
    }

    cpu_sgemm(a,b,c,M,N,K);

    int count = sizeof(c)/sizeof(float);

    for(int i = 0; i < 10; i++)
    {
        std::cout << "c["<<i<<"]="<<c[i]<<std::endl;
    }
    
    std::cout << " cpu_sgemm2 main "<<std::endl;
    return 0;
}