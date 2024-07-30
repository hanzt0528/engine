#include <iostream>

int main(int agrc,char* argv[])
{
    for(int i = 0; i < 16*16; i++)
    {
       // std::cout <<"thread "<<i<<" m,k = "<<(i>>1)<<","<<((i&1)<<2)<<std::endl;
        std::cout <<"thread "<<i<<" i&31 = "<<(i&31)<<std::endl;
        std::cout <<"thread "<<i<<" i%31 = "<<(i%31)<<std::endl;
    }
}