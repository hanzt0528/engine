#include "threadpool.h"


int func(int k)
{
    return k+10;
}
int main(int argc,char* argv[])
{
    std::cout << "main:"<<std::endl;

    ThreadPool pool(5);

    auto ret = pool.enqueue([](int answer) ->float { return answer; }, 42);

    std::cout <<"ret = "<<typeid(ret.get()).name()<<std::endl;

    return 0;
}
