#ifndef _ATOMIC_QUEUE
#define _ATOMIC_QUEUE

#include <condition_variable>
#include <mutex>
#include <queue>

template<typename Data>
class AtomicQueue
{
private:
	std::queue<Data> dataQueue;
	mutable std::mutex conditionMutex;
	std::condition_variable conditionVar;
public:
	void push(Data const& data)
	{
		{
			std::lock_guard<std::mutex> lock(conditionMutex);
			dataQueue.push(data);
		}
		conditionVar.notify_one();
	}

	bool empty() const
	{
		std::lock_guard<std::mutex> lock(conditionMutex);
		return dataQueue.empty();
	}

	bool try_pop(Data& popped_value)
	{
		std::lock_guard<std::mutex> lock(conditionMutex);
		if(dataQueue.empty())
			return false;		

		popped_value=dataQueue.front();
		dataQueue.pop();
		return true;
	}

	void wait_and_pop(Data& popped_value)
	{
		std::unique_lock<std::mutex> lock(conditionMutex);
		while(dataQueue.empty())		
			conditionVar.wait(lock);		

		popped_value=dataQueue.front();
		dataQueue.pop();
	}

	size_t size()
	{
		std::lock_guard<std::mutex> lock(conditionMutex);
		return dataQueue.size();
	}
};

#endif
