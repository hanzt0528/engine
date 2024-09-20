
#include <iostream>

#include <thread>
#include <queue>
#include <mutex>
#include <vector>

#include <condition_variable>
#include <future>
#include <functional>

class ThreadPool
{
    private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::condition_variable condition;
    std::mutex queue_mtx;
    bool stop;

    public:
    ThreadPool(size_t size)
    :stop(false)
    {
        for(size_t i = 0; i < size; i++)
        {
            workers.emplace_back(std::thread([this]{

                for(;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mtx);
                        condition.wait(lock,[this] {return this->stop || !this->tasks.empty(); });

                        if(this->stop && this->tasks.empty())
                            break;

                        task = std::move(this->tasks.front());
                        this->tasks.pop();

                    }   
                    task();
                }
            }));
        }

    }

    template<class F,class... Args>
    auto enqueue(F &&f,Args&&... args)-> std::future< typename std::result_of<F(Args...)>::type >
    {
        using return_type = typename std::result_of<F(Args...)>::type;


        auto task = std::make_shared<std::packaged_task<return_type>>
        std::future<return_type> res;

        return res;

    }
};
int main(int argc,char* argv[])
{
    std::cout << "main:"<<std::endl;
    return 0;
}