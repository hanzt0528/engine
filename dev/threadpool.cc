#include <iostream>
#include <thread>
#include <vector>
#include <queue>

#include <functional>
#include <mutex>
#include <future>
#include <condition_variable>
#include <stdexcept>

class ThreadPool
{
    private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mtx;
    std::condition_variable condition;
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
                        condition.wait(lock,[this]{return !this->tasks.empty() || this->stop;});
                        if(this->tasks.empty() && this->stop)
                        {
                            break;
                        }
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            })
            );
        }
    }

    template<class F,class... Args>
    auto enqueue(F &&f,Args&&... args)-> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f),std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mtx);

            if(this->stop) throw std::runtime_error("enqueue on stoped pool!");

            this->tasks.emplace([task]{(*task)();});
        }

        condition.notify_one();


        return res;
    }
    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mtx);
            stop = true;
        }
        condition.notify_all();

        for(std::thread &worker:workers)
        {
            worker.join();
        }
    }
};

int func(int k)
{
    return k+10;
}

int main(int argc,char* argv[])
{
    std::cout << "main:"<<std::endl;

    ThreadPool pool(5);
    auto ret = pool.enqueue([](int answer){return answer;},100);

    std::cout << "ret = "<< ret.get()<<std::endl;
    return 1;
}