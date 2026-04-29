#ifndef _GPU_WORKER
#define _GPU_WORKER

#include "GPUDevice.h"
#include "kernels.h"
#include "Clock.h"
#include "log.h"
#include "system_utils.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>


template <class Ftype>
class GPUWorker
{
private:
	
	GPUDevice *_gpuDevice;
	cudaStream_t _stream;
	Ftype* _d_data;
	long _numSamples;
	int _depthCurrent;
	bool _noProbe;
	HistType _htype;
	uint* _indexBuffer;
	Ftype* _dataBuffer;
	std::string _logHdr;

public:
	//std::stringstream outStr;

	GPUWorker():_gpuDevice(0), _stream(0), _d_data(0), _numSamples(-1),_noProbe(false), 
		_indexBuffer(0), _dataBuffer(0), _depthCurrent(-1), _htype(HIST_64){}

	void init(GPUDevice* gpuDevice)
	{
		_gpuDevice = gpuDevice;
		_numSamples = gpuDevice->numSamples();
		_gpuDevice->setCurrent();
		//outStr.str();

		size_t featureMem = _numSamples*sizeof(Ftype);
		size_t indexBufMem = BLOCKSIZE_1 * sizeof(uint);
		size_t dataBufMem = BLOCKSIZE_1 * sizeof(Ftype);

		cudaStreamCreate( &_stream);		
		FILE_LOG(LOG1) << "Allocating for GPUWorker ("<< gpuDevice->deviceId() << "):";
		FILE_LOG(LOG1) << "     featureMem:  " << featureMem/1024 << "kb";
		FILE_LOG(LOG1) << "     indexBufMem: " << indexBufMem/1024 << "kb";
		FILE_LOG(LOG1) << "     dataBufMem:  " << dataBufMem/1024 << "kb";

		cudaError_t err = cudaMalloc( &_d_data, featureMem);
		if (err != cudaSuccess)
		{
			FILE_LOG(logERROR) << "Could not allocate worker memory on gpuDevice: "<< gpuDevice->deviceId();
			FILE_LOG(logERROR) << "Error: " << errCESTRing(cudaGetLastError());
			discardSystemResult("pause");
		}
		cudaMalloc( &_indexBuffer, indexBufMem);
		cudaMalloc( &_dataBuffer, dataBufMem);
	}

	inline void setWorkerHeading(int workerId, int fId)
	{
		std::ostringstream os;
		os << "wId: " << workerId << ", fId: " << fId << ", ";
		_logHdr = os.str();
	}

	void sendData(const Ftype* h_data, const uint* index)
	{
		Clock clk;
		clk.tic();
		_gpuDevice->setCurrent();
		
		for (long i = 0; i < _numSamples; i += BLOCKSIZE_1 )
		{									
			long dataCount = std::min((long int)BLOCKSIZE_1, _numSamples - i); 
			cudaMemcpyAsync(_dataBuffer, h_data + i, dataCount*sizeof(Ftype), cudaMemcpyHostToDevice, _stream); 
			cudaMemcpyAsync(_indexBuffer, index + i, dataCount*sizeof(uint), cudaMemcpyHostToDevice, _stream); 
			gpuScatteredWrite(_d_data, _dataBuffer, _indexBuffer, dataCount, _stream); //_d_data[_indexBuffe[j]] = _dataBuffer[j];
		}
		cudaStreamSynchronize(_stream);
		
		cudaError_t ce = cudaGetLastError();
		if (ce!=cudaSuccess)
		{ 
			FILE_LOG(logERROR) << _logHdr << "Error in sending data to gpuDevice: " <<_gpuDevice->deviceId();
			FILE_LOG(logERROR) << "Error: " << errCESTRing(cudaGetLastError());
			discardSystemResult("pause");	
		}
		FILE_LOG(LOG2) << _logHdr << " Sending data: " << clk.toc() << "s"; 
	}
	
	float calculateThreshold(int tId, int fId, std::vector<float> & thresh, int numThresh)
	{
		float eps = 1e-5f;
		float maxVal = floor(thresh[fId*3 + 0]/2.0f - 1); // maxVal comes from the maximum positive value a pixel can have. See generatedata and getfeatures for more info
		float featureMin = thresh[fId*3 + 1];
		float featureMax = thresh[fId*3 + 2];

		if (numThresh == 1)
		{
			return maxVal - eps; // Not necessarily the best choice
		}

		if (numThresh == 2)
		{
			if (tId==0) return maxVal - eps;
			if (tId==1) return 0 - eps;
		}		

		if (numThresh >= 3)
		{
			if (tId==0) return maxVal - eps;
			if (tId==1) return 0 - eps;
			if (tId==2) return featureMin - eps;

			float tDelta = (featureMax - featureMin)/(numThresh - 3);

			return (featureMin - eps) + (tId-2)*tDelta;	
			//return (tId < numThresh -1)? (featureMin - eps) + rand()/(float)RAND_MAX*(featureMax - featureMin) : maxVal - eps;		
		}
		return  3.4e38f;
	}


	//This probe determines which histogram method is appropriate according to runtime
	void runProbe(int depth, int fId, RFTrainNode * nodes)
	{		
		if ( (_noProbe) || (depth == _depthCurrent) || (_htype == HIST_REGULAR) ) //then no need to perform the probe
			return;
		_depthCurrent = depth;

		Clock clk;
		float t1 = 0, t2=0;

		int numNodes = pow(2.0f, depth); // depth starts from 0 
		int idxDepth = numNodes - 1; 
		int foldingDepth = _gpuDevice->foldingDepth(); // This is the depth where the passes over the data start getting split according to node
		int delta = numNodes / ceil(pow(2.0, depth - foldingDepth));		
		int numClasses = _gpuDevice->numClasses();
		
		std::ostringstream log;
		log << _logHdr << "Speed probe ratio:";		

		if (numClasses > 256) {
			_htype = HIST_REGULAR;
			_noProbe = true;
			log << " numClasses>256. Using HIST_REGULAR";
			FILE_LOG(LOG2) << log.str();
			return;
		}
		
		for (int j = 0; j < numNodes; j += delta)
		{
			int nodeBegin = j + idxDepth; 
			int nodeEnd = j + delta - 1 + idxDepth;				
			
			if (numClasses <=64)	t1 += buildHistogram(PARENT_NODE, HIST_64, 0, _d_data, _gpuDevice, nodes, nodeBegin, nodeEnd,_stream);
			else					t1 += buildHistogram(PARENT_NODE, HIST_256, 0, _d_data, _gpuDevice, nodes, nodeBegin, nodeEnd,_stream);

			t2 += buildHistogram(PARENT_NODE, HIST_REGULAR, 0, _d_data, _gpuDevice, nodes, nodeBegin, nodeEnd,_stream);
		}

		if (t2/t1 > 1.1)
		{
			if (numClasses <=64)	{ _htype = HIST_64;  log << " " << t2/t1 << " . Using HIST_64";  }
			else					{ _htype = HIST_256; log << " " << t2/t1 << " . Using HIST_256"; }
		}
		else
		{
			_htype = HIST_REGULAR;	log << " " << t2/t1 << " . Using HIST_REGULAR";
			_noProbe = true;
		}
		FILE_LOG(LOG2) << log.str();
	}


	void map(int depth, int fId, int numThresh, std::vector<float> & thresh, RFTrainNode * nodes)
	{		
		//We need to lock the cpu thread so that no more requests are added to any of the streams 
		std::lock_guard<std::mutex> lock(*_gpuDevice->getMutex());
		_gpuDevice->setCurrent();
		Clock clk;		
		float t1,t2;

		int numNodes = pow(2.0f, depth); // depth starts from 0 
		int idxDepth = numNodes - 1; 
		int foldingDepth = _gpuDevice->foldingDepth(); // This is the depth where the passes over the data start getting split according to node
		long delta = numNodes / ceil(pow(2.0, depth - foldingDepth));		
		int numClasses = _gpuDevice->numClasses();
		
		runProbe(depth, fId, nodes);	
		//_htype = HIST_REGULAR;
		//_htype = HIST_64;
		//_htype = HIST_256;
		//To mitigate the problem of too much memory being allocated to histograms we partition
		// their calculation into parts. This is made possible by the fact that samples are sorted according to node
		// and the nodes contain the start and stop positions of the contiguous sample subsets 
		for (int j = 0; j < numNodes; j += delta)
		{			
			int nodeBegin = j + idxDepth; 
			int nodeEnd = j + delta - 1 + idxDepth;		

			//Now, here we calculate a histogram for the parents of the current level to be calculated
			//this is stored in the first half of available histogram memory			
			
			buildHistogram(PARENT_NODE, _htype, 0, _d_data, _gpuDevice, nodes, nodeBegin, nodeEnd,_stream);
			
			//now for each subset of nodes we run threshold tests
			t1=0; t2=0;
			//int cnt=0;			
			for (int i = 0; i < numThresh; i++)
			{			
				clk.tic();
				float threshold = calculateThreshold(i, fId, thresh, numThresh); 
				buildHistogram(CHILD_NODE, _htype, threshold, _d_data, _gpuDevice, nodes, nodeBegin, nodeEnd, _stream);				
				t1+=clk.toc();

				clk.tic();
				// We only need to update next node level entropies if we are before the last level					
				computeEntropies(depth < _gpuDevice->maxDepth(), fId, threshold, _gpuDevice, nodeBegin, nodeEnd, _stream );			
				t2+=clk.toc();
			}			
			
			std::ostringstream log;
			log << _logHdr << "Nodes:(" << nodeBegin << " -> " << nodeEnd << "), Hist(s):"
				<< t1/numThresh << ", Hist/Ent:" << t1/t2;
			FILE_LOG(LOG2) << log.str();
		}

		// Here we want to go through all the samples and for the ones whos node has been updated we want to update its bestTraj 
		// NOT its nodetraj. The nodetraj gets updated after sorting happens.
		updateSamples(_d_data, _gpuDevice, idxDepth, idxDepth + numNodes - 1, _stream); 

		// This IS with the idxDepth offset because we only need to reset the current level
		resetUpdateFlag(_gpuDevice->nodes() + idxDepth, numNodes, _stream);				
	}


	~GPUWorker()
	{
		if (!_gpuDevice)
			return;

		_gpuDevice->setCurrent();
		if (_stream)
			cudaStreamDestroy(_stream);
		if (_d_data)
			cudaFree(_d_data);
		if (_indexBuffer)
			cudaFree(_indexBuffer);
		if (_dataBuffer)
			cudaFree(_dataBuffer);
	}

};

#endif

