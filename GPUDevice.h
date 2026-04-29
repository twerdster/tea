#ifndef _GPU_DEVICE
#define _GPU_DEVICE

#include <fstream>
#include <mutex>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "log.h"
#include "defines.h"
#include "kernels.h"
#include "system_utils.h"


class GPUDevice
{
private:
	int _deviceId;
	int _maxDepth;
	int _maxNodes;
	int _numClasses;
	int _numSamples;
	int _foldingDepth;
	int _histBytes;
	mutable std::mutex *_deviceMutex;
	bool _aNeedsCalculating;
	bool _isValid;
	WeightType _weightType;

	uint* _histograms;
	uint* _partialHistograms;
	uint* _apriori;
	RFTrainNode* _nodes;
	Sample* _samples;
	

public:
	GPUDevice():_deviceMutex(0),_deviceId(-1), _histograms(0),_histBytes(0), 
		_nodes(0), _samples(0), _maxNodes(0), _maxDepth(0), 
		_numClasses(0), _numSamples(0), _partialHistograms(0) {}

	void init(int deviceId, int numSamples, int maxNodes, int maxDepth, int numClasses, int foldingDepth)
	{
		_deviceId =  deviceId;
		_maxNodes = maxNodes;
		_maxDepth = maxDepth;
		_numClasses = numClasses;
		_numSamples = numSamples;
		_foldingDepth = foldingDepth;
		_deviceMutex = new std::mutex();
		_histBytes = 2*numClasses*pow(2.0,foldingDepth)*sizeof(uint);
		_aNeedsCalculating = true;
		_isValid = true;
		_weightType = W_ONES;

		setCurrent();
		size_t free,total;
		cudaMemGetInfo(&free,&total);
		size_t partialMem = MAX_PARTIAL_HISTOGRAM64_COUNT * HISTOGRAM64_BIN_COUNT * sizeof(uint);
		size_t histMem = _histBytes;
		size_t nodeMem = 2*maxNodes*sizeof(RFTrainNode);
		size_t sampleMem = numSamples*sizeof(Sample);
		size_t classesMem = numClasses*sizeof(uint);
		
		FILE_LOG(LOG1) << "Allocating for GPUDevice("<< _deviceId << "):";
		FILE_LOG(LOG1) << "     partialMem: " << partialMem/1024 << "kb";
		FILE_LOG(LOG1) << "     histMem: " << histMem/1024 << "kb";
		FILE_LOG(LOG1) << "     nodeMem: " << nodeMem/1024 << "kb";
		FILE_LOG(LOG1) << "     sampleMem: " << sampleMem/1024 << "kb";
		FILE_LOG(LOG1) << "     classesMem: " << classesMem/1024 << "kb";
		
		cudaMalloc( (void **)&_partialHistograms, partialMem) ;
		cudaMalloc( &_histograms, histMem); // foldingDepth defines how many histograms we create.  
		cudaMalloc( &_nodes, nodeMem); //In this case we take the total number of nodes that will be calculated for this whole tree which is maxNodes + 0.5*maxNodes + 0.25* ... = 2*maxNodes
		cudaMalloc( &_samples, sampleMem);
		cudaMalloc( &_apriori, classesMem);

		cudaError ce = cudaGetLastError();
		cudaDeviceProp deviceProps;
		cudaGetDeviceProperties(&deviceProps, _deviceId);
		if (ce == cudaSuccess)
			FILE_LOG(LOG0) << "Succesfully added CUDA device [" << deviceProps.name << "]";
		else
		{
			FILE_LOG(logERROR) << "Found CUDA device [" << deviceProps.name << "] but could not add it";
			FILE_LOG(logERROR) << "Error: " << errCESTRing(ce);
			discardSystemResult("pause");
		}
	}


	void pushSamples(const Sample* h_src)
	{
		setCurrent();
		cudaMemcpy( _samples, h_src, _numSamples*sizeof(Sample), cudaMemcpyHostToDevice); 	

	}
	
	void pullSamples(Sample* h_dst, const int idxStart, const int idxEnd)
	{
		setCurrent();
		int numSubSamples = idxEnd - idxStart + 1;
		
		if (numSubSamples > 0)
			cudaMemcpy( h_dst + idxStart, _samples + idxStart, numSubSamples*sizeof(Sample), cudaMemcpyDeviceToHost); 
	}

	// Push whatever is in nodes into _nodes from _nodes[startIdx] to _nodes[startIdx+numElements-1]
	void pushNodes(const RFTrainNode* nodes, const int startIdx, const int numElements)
	{
		setCurrent();
		cudaMemcpy( _nodes + startIdx, nodes, numElements * sizeof(RFTrainNode), cudaMemcpyHostToDevice); 
	}

	// Pull whatever is in _nodes from _nodes[startIdx] to _nodes[startIdx+numElements-1] into nodes
	void pullNodes(RFTrainNode* nodes, const int startIdx, const int numElements)
	{
		setCurrent();
		cudaMemcpy( nodes, _nodes + startIdx, numElements * sizeof(RFTrainNode), cudaMemcpyDeviceToHost); 
	}

	void setWeights(WeightType weightType, std::string fileName)
	{
		if (!_aNeedsCalculating) return;
		_aNeedsCalculating=false;

		setCurrent();

		_weightType = weightType;

		uint *tmp = new uint[_numClasses];
		for (int i = 0 ; i < _numClasses; i++)
				tmp[i]=1;

		switch (_weightType)
		{
		case W_ONES: // No weighting is done when calculating the probabilities for entropy/impurity 						
			break;

		case W_APRIORI: // Weighting is used for calculating probabilities according the the inverse of the apriori disribution of samples
			RFTrainNode node;
			node.idxBegin = 0;
			node.idxEnd = _numSamples-1;
			buildHistogram(PARENT_NODE, HIST_REGULAR, 0, (char*)0, this, &node, 0, 0, 0);
			cudaMemcpy(tmp, _histograms, _numClasses*sizeof(uint),cudaMemcpyDeviceToHost);				
			break;

		case W_FILE: //Unsigned integer weights are read from a file for calculting the probablities for entropy/impurity
			std::ifstream ifs(fileName.c_str());
			while (!ifs.eof())
			{
				int i,weight;
				ifs >> i;				
				ifs >> weight;
				if (i >= 0 && i < _numClasses)
					tmp[i] = weight;
			}
			ifs.close();
			break;
		}
		
		cudaMemcpy(_apriori, tmp, _numClasses*sizeof(uint),cudaMemcpyHostToDevice);		

		std::stringstream ss;
		for (int i = 0; i < _numClasses; i++)
			ss << tmp[i] << " ";

		FILE_LOG(logINFO) << "Weights: " << ss.str();

		delete []tmp;
		return;


		
	}

	void setCurrent()			
	{ 		
		cudaSetDevice(_deviceId); 
	}

	bool isValid()				{ return _isValid; }
	uint* histograms()			{ return _histograms; }
	uint* partialHistograms()	{ return _partialHistograms; }
	RFTrainNode* nodes()  		{ return _nodes; }			
	Sample* samples()			{ return _samples; }	
	uint* W()					{ return _apriori; } // NOTE: this should be made into a float array

	uint histBytes()			{ return _histBytes; }
	uint histWords()			{ return _histBytes/sizeof(uint); }
	uint maxNodes()				{ return _maxNodes; }
	uint maxDepth()				{ return _maxDepth; }
	uint foldingDepth()			{ return _foldingDepth; }
	uint numClasses()			{ return _numClasses; }
	uint numSamples()			{ return _numSamples; }	
	uint deviceId()				{ return _deviceId; }	
	std::mutex* getMutex()		{ return _deviceMutex; }

	~GPUDevice()
	{
		if (_deviceId < 0 ) return;

		setCurrent();
		if (_partialHistograms)
			cudaFree(_partialHistograms);
		if (_deviceMutex)
			delete _deviceMutex;
		if (_histograms)
			cudaFree(_histograms);
		if (_nodes)
			cudaFree(_nodes);
		if (_samples)
			cudaFree(_samples);
		if (_apriori)
			cudaFree(_apriori);
		cudaDeviceReset();
	}
};

#endif //_GPU_DEVICE
