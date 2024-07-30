//#include <stdio.h>
//clang -fmodules -fsyntax-only -Xclang -dump-tokens code.c 

long f(long a,long b)
{
    long x = a;
    if(a>b)
    {
        x+=20;
    }
    else
    {
        x+=b;
    }

    return x;
}

int main(int argc,char*argv[])
{
    int k = f(12,22);
    //printf("hh k = %d",k);
    return 0;
}