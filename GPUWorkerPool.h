#ifndef _GPUWORKER_POOL
#define _GPUWORKER_POOL

#include <assert.h>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>
#include "AtomicQueue.h"
#include "GPUWorker.h"
#include "GPUDevice.h"
#include "defines.h"


//class GPUWorkerPoolBase {};

template <class Ftype>
class GPUWorkerPool //: public GPUWorkerPoolBase
{
private:
	int _numWorkers, _numDevices, _numStreams;

	//The _availableQueue holds the tokens of the usable worker spots
	AtomicQueue<int> _availableQueue;
	std::vector<GPUWorker<Ftype> > _pool;
	std::vector<GPUDevice> *_gpuDevice;

	mutable std::mutex _listMutex;

	// These are used for load thread dispatches
	mutable std::mutex _aquireMutex;
	std::condition_variable _acquireConditionVar;
	int _acquireCount;

public:
	// A pool is defined by the number of devices and the number of streams for each worker
	GPUWorkerPool(std::vector<GPUDevice> *gpuDevice, int numStreams): _gpuDevice(gpuDevice), _numStreams(numStreams)
	{		
		_numDevices = _gpuDevice->size();
		_numWorkers = _numDevices*_numStreams;
		_acquireCount = 0;

		// NB
		// The order of the loops can potentially have a dramatic impact on the runtime
		// It is important that workers be called from different devices as much as possible.		
		_pool.resize(_numWorkers);	
		for (int stream = 0; stream < _numStreams; stream++)
		{
			for (int deviceId = 0; deviceId < _numDevices; deviceId++)
			{
				if (!(*_gpuDevice)[deviceId].isValid()) 
					continue;
				int slot = stream*_numDevices + deviceId;
				_pool[slot].init(&((*_gpuDevice)[deviceId]));
				_availableQueue.push(slot);
			}
		}		
	}

	// Waits until a worker is available.
	// This is only ever called from one main thread and not from separate threads.
	// Also, it will only ever be called before a corresponding release and is gauranteed to be called
	// a fixed number of times before waitForGPUWorkers is called. This means that as long as a worker is still in flight
	// _aquireCount will be nonzero. 
	int acquireGPUWorker()
	{
		{
			//We need to protect against the possibility that a release is done as we increase the acquirecount
			std::lock_guard<std::mutex> lock(_aquireMutex);
			_acquireCount++;
		}

		int workerId;
		_availableQueue.wait_and_pop(workerId);

		return workerId;
	}

	//Defines a worker as processed and releases the memory position of the worker.
	void releaseGPUWorker(int workerId)
	{
		{
			std::lock_guard<std::mutex> lock(_aquireMutex);
			_acquireCount--;
			assert(_acquireCount>=0);		

			if (!_acquireCount) // Then all the threads have finished
				_acquireConditionVar.notify_one(); 
		}
		_availableQueue.push(workerId);
	}

	void waitForGPUWorkers()
	{
		std::unique_lock<std::mutex> lock(_aquireMutex);

		while (_acquireCount) 
			_acquireConditionVar.wait(lock);
	}

	GPUWorker<Ftype>* getWorker(int workerId)
	{
		return &_pool[workerId];
	}

	//Shuts the workerLoader thread down and releases all resources.
	~GPUWorkerPool()
	{
		// Needs proper destruction
	}	
};

#endif
