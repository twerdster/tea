#ifndef _FEATURE_POOL
#define _FEATURE_POOL

#include <assert.h>
#include <boost/thread.hpp>
#include <queue>
#include <vector>
#include <iostream>
#include <string>
#include "log.h"
#include "AtomicQueue.h"
#include "Feature.h"
#include "defines.h"

//class FeaturePoolBase {};

template <class Ftype>
class FeaturePool //: public FeaturePoolBase
{
private:
	int _numFeatures;
	int _poolSize;

	//The free list holds the tokens of the usable feature spots
	AtomicQueue<int> _availableQueue;

	//The available list holds features that need to be processed this round and the waiting list features that need to be processed next round
	AtomicQueue<int> *_processingQueue, *_waitingQueue, *_tmp;
	std::vector<Feature<Ftype> > _pool;

	bool _needsReset;
	bool _loaderEnabled;

	mutable boost::mutex _queueMutex;

	// These are used for load thread dispatches
	mutable boost::mutex _loadMutex;
	boost::condition_variable _loadConditionVar;
	int _loadCount, _totalLoaded;


public:
	// A pool is defined by the number of features where each feature will be indexed
	// and the poolSize determines how many features can simultaneously be available.
	FeaturePool(int numFeatures, int poolSize, std::string strFeatureBaseName, int numSamples)://, const uint *index):
	  _numFeatures(numFeatures), _poolSize(poolSize)
	  {
		  assert(numFeatures >= poolSize && numFeatures && poolSize);
		  _loaderEnabled = (numFeatures != poolSize); 
		  _needsReset = false;
		  _loadCount = 0;
		  _totalLoaded = 0;

		  // Here we reserve poolSize features and then init each feature. The term slot here is used to represent
		  // the fact that we are putting data in a slot. Then later each slot is represented by a token which is passed around.
		  _pool.resize(poolSize);
		  for (int slot = 0; slot < poolSize; slot++)
			  _pool[slot].init(strFeatureBaseName,numSamples);//,index); //To generalize this class you could make the input to this class as having variable argument list

		  _processingQueue = new AtomicQueue<int>;
		  _waitingQueue = new AtomicQueue<int>;

		  if (_loaderEnabled)
			  for (int token = 0; token < _poolSize; token++) 			 
				  // Intialize the swap tokens	
				  _availableQueue.push(token);		

		  else {
			  Clock clk;
			  clk.tic();
			  for (int token = 0; token < _poolSize; token++) 	
			  {
				  // This loads all data and adds all features to available list. 
				  _loadAcquire();
				 // boost::thread loaderThread(&FeaturePool::_load,this,token,token);
				  _load(token,token);
			  }
			  FILE_LOG(LOG2) << "Total load time " << clk.toc() << "s";
		  }

		  //This wait is necessary to gaurantee correct operation of the first 
		  //run because it would be possible for an acquireFeature call to see an empty availabeList and then reset itself
		  //incorrectly
		  _waitForLoads();

		  // Launch the loader thread. This will exit immediately if the loader is disabled.
		  //We will need some way to ensure that all threads have stopped before using a destructor
		  boost::thread loaderThread(&FeaturePool<Ftype>::_featureLoader,this);
	  }

	  Feature<Ftype>* getFeature(int token)  {	return &_pool[token];  }

	  // Waits until a feature is available and then locks the memory position
	  // Returns NULL if there are no more features to be acquireed. Otherwise returns the loaded feature.
	  int acquireFeatureToken()
	  {
		  // If the loader has finished a full set then no more features can be made available
		  // until the loader has been reset. The main thread will be made aware of this by this 
		  // function returning a NULL value
		  {
			  //We lock this so that access to the list and _totalLoaded is atomic
			  boost::mutex::scoped_lock lock(_queueMutex);
			  if ((_totalLoaded == _numFeatures || !_loaderEnabled) && _processingQueue->empty()  ) 
			  {
				  _reset();
				  return -1;
			  }
		  }

		  // Now, wait until there is a token available in the available list which will be added by the 
		  // classes main thread loop when the load is finished
		  int token;
		  _processingQueue->wait_and_pop(token);

		  return token;
	  }

	  //Defines a feature as processed and releases the memory position of the feature.
	  void releaseFeatureToken(int token)
	  {
		  // Make the token used by a particular feature available for loading into
		  if (_loaderEnabled)
			  _availableQueue.push(token);
	  }	  

	  //Shuts the featureLoader thread down and releases all resources.
	  ~FeaturePool()
	  {
		  _waitForLoads();
		  // These all need to be propperly destructed. Its not critical atm because FeaturePool should only
		  //be released at the very end of the program
		  // _processingQueue and _waitingQueue might need to check that they are not being used by any threads	
		  //_pool is a vector and a destructor will be automatically called for each element
		  delete _processingQueue;
		  delete _waitingQueue;
	  }

private:
	//Resets the feature pool so that the features that have been loaded for the next round
	  //are subsequently made available to the acquireFeature function. If reset is called before
	  //all threads have finished the behaviour will be undefined
	  void _reset()
	  {
		  if (_loaderEnabled) 
		  {			
			  // Swap the waiting list and available list
			  _tmp = _waitingQueue;
			  _waitingQueue = _processingQueue;
			  _processingQueue = _tmp;

			  // Tell the feature loader thread it can add new features into the available list
			  _totalLoaded = _processingQueue->size();
		  } 
		  else
		  {
			  for (int token = 0; token < _numFeatures; token++)
				  // Used ONLY when the poolsize equals the number of features. The features will be loaded only once in this case.
				  // Also note they are being loaded according to their positions
				  _processingQueue->push(token);
		  }
	  }

	void _loadAcquire()
	{
		boost::mutex::scoped_lock lock(_loadMutex);
		_loadCount++;
	}

	void _loadRelease()
	{
		boost::mutex::scoped_lock lock(_loadMutex);
		_loadCount--;
		assert(_loadCount>=0);

		if (!_loadCount) // Then all the load threads have finished
			_loadConditionVar.notify_one(); 
	}

	void _waitForLoads()
	{
		boost::mutex::scoped_lock lock(_loadMutex);

		while (_loadCount) 
			_loadConditionVar.wait(lock);
	}

	void _load(int token, int fId)
	{
		//_loadAquire is called BEFORE the thread is launched and then _loadRelease is called at the end of this method

		// Request the Feature class to do the load
		_pool[token].load(fId);			

		// Add the newly loaded feature as being available or put it on the waiting list
		// The lock is to ensure that the correct list will be updated without corruption if reset is called
		boost::mutex::scoped_lock lock(_queueMutex); 

		if (_totalLoaded < _numFeatures) 
		{			
			_processingQueue->push(token);
			_totalLoaded++;
		}
		else 
			_waitingQueue->push(token);

		// Release the acquire count for the load threads
		_loadRelease();				
	}

	void loadThread(int token, int fId)
	{
		// Here we acquire a load and then load a particular feature into the given slot identified by the token					
		_loadAcquire();
		_load(token,fId);
		//boost::thread loadThread(&FeaturePool::_load,this, token, fId); // The load thread releases the acquire at the end
	}

	void _featureLoader()
	{
		Clock clk;
		while (_loaderEnabled) 
		{
			clk.tic();
			// Run through all the features 
			for (int fId = 0; fId < _numFeatures; fId++)
			{				
				// Get the token representing the currently available spot for loading data into
				int token;
				_availableQueue.wait_and_pop(token);

				loadThread(token,fId);				
			}

			//We need to wait for all the threaded loads to return i.e. execute _loadRelease
			_waitForLoads();		
			//printf("Total load time for all features:%f\n",clk.toc());
		}
	}
};



#endif