#include<iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>

#include<future>
#include <functional>
#include <condition_variable>
#include <stdexcept>


class ThreadPool
{
    private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::condition_variable condition;
    std::mutex queue_mtx;
    bool stop;

    public:
    ThreadPool(size_t threads)
    :stop(false)
    {
        for(size_t i = 0; i < threads; i++)
        {
            workers.push_back(std::thread([this]{
                for(;;)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mtx);
                        condition.wait(lock,[this]{ return this->stop || !this->tasks.empty();});

                        if(this->stop && this->tasks.empty())
                        {
                            break;
                        }

                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            }));
        }

    }

    template<class F,class... Args>
    auto enqueue(F &&f,Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f),std::forward<Args>(args)...)
        );

        std::future<return_type> res;

        res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mtx);

            if(this->stop) throw std::runtime_error("on the stopped pool!");

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