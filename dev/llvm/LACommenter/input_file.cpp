extern void foo(int some_arg);

void bar() {
  foo(123);
}

void func(float *a,float *b,float *c,const int M,const int N,const int K)
{
    return;
}
void bee(float b) {
  
}


int main(int argc,char * argv[])
{
    bar();
    const int M = 10;
    const int N = 10;
    const int K = 10;
    float a[M*K] ={0};
    float b[N*K] = {0};
    float c[M*N] = {0};
    func(a,b,c,M,N,K);

    bee(10.1);
    return 0;
}