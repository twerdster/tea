#ifndef _ATOMIC_QUEUE
#define _ATOMIC_QUEUE

#include <boost/thread.hpp>
#include <queue>

template<typename Data>
class AtomicQueue
{
private:
	std::queue<Data> dataQueue;
	mutable boost::mutex conditionMutex;
	boost::condition_variable conditionVar;
public:
	void push(Data const& data)
	{
		boost::mutex::scoped_lock lock(conditionMutex);
		dataQueue.push(data);
		lock.unlock();
		conditionVar.notify_one();
	}

	bool empty() const
	{
		boost::mutex::scoped_lock lock(conditionMutex);
		return dataQueue.empty();
	}

	bool try_pop(Data& popped_value)
	{
		boost::mutex::scoped_lock lock(conditionMutex);
		if(dataQueue.empty())
			return false;		

		popped_value=dataQueue.front();
		dataQueue.pop();
		return true;
	}

	void wait_and_pop(Data& popped_value)
	{
		boost::mutex::scoped_lock lock(conditionMutex);
		while(dataQueue.empty())		
			conditionVar.wait(lock);		

		popped_value=dataQueue.front();
		dataQueue.pop();
	}

	size_t size()
	{
		boost::mutex::scoped_lock lock(conditionMutex);
		return dataQueue.size();
	}
};

#endif