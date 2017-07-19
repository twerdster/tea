#include "kernels.h"
#include "bsearch.cuh"
#include "GPUDevice.h"
#include <assert.h>
#include <cuda_runtime.h>

template <class Ftype, class Ftype4>
__global__ 
	void _gpuScatteredWrite(Ftype* dst, const Ftype4 *src, const uint4 *index, const uint dataCount)
{
	const int idx = ( blockIdx.x*gridDim.y + blockIdx.y ) * blockDim.x + threadIdx.x;

	if (idx < dataCount)   
	{
		uint4 _index = index[idx];
		Ftype4 data = src[idx];

		dst[ _index.x ] = data.x;
		dst[ _index.y ] = data.y;
		dst[ _index.z ] = data.z;
		dst[ _index.w ] = data.w;    		
	}
}

template <class Ftype, class Ftype4>
void gpuScatteredWrite(Ftype* dst, const Ftype *src, const uint *index, const uint dataCount, cudaStream_t stream)
{
	assert( dataCount % (4) == 0 );

	const float THREADS = 512;
	const float BLOCKS = ceil(dataCount/THREADS/4.0f);
	uint sqrtVal = ceil(sqrt(BLOCKS));        
	dim3 numBlocks(sqrtVal,sqrtVal);
	dim3 numThreads(THREADS);

	_gpuScatteredWrite<<< numBlocks, numThreads , 0, stream >>>(dst, (const Ftype4*)src, (const uint4*)index, dataCount/4);
	if (cudaGetLastError()!=cudaSuccess){ 
		FILE_LOG(logERROR) << "Error occurred in gpuScatteredWrite";
		FILE_LOG(logERROR) << "Error: " << errCESTRing(cudaGetLastError());
		system("pause");		
	}
}

template <class Ftype, class Ftype4>
__global__ 
	void _gpuScatteredRead(Ftype4* dst, const Ftype *src, const uint4 *index, const uint dataCount)
{
	const int idx = ( blockIdx.x*gridDim.y + blockIdx.y ) * blockDim.x + threadIdx.x;

	if (idx < dataCount)   
	{
		uint4 _index = index[idx];
		Ftype4 *data = &dst[idx];

		data->x = src[ _index.x ] ;
		data->y = src[ _index.y ] ;
		data->z = src[ _index.z ] ;
		data->w = src[ _index.w ] ;    		
	}
}

template <class Ftype, class Ftype4>
void gpuScatteredRead(Ftype* dst, const Ftype *src, const uint *index, const uint dataCount, cudaStream_t stream)
{
	assert( dataCount % (4) == 0 );

	const float THREADS = 512;
	const float BLOCKS = ceil(dataCount/THREADS/4.0f);
	uint sqrtVal = ceil(sqrt(BLOCKS));        
	dim3 numBlocks(sqrtVal,sqrtVal);
	dim3 numThreads(THREADS);

	_gpuScatteredRead<<< numBlocks, numThreads , 0, stream >>>(	(Ftype4*)dst, src, (const uint4*)index, dataCount/4);	
	if (cudaGetLastError()!=cudaSuccess){ 
		FILE_LOG(logERROR) << "Error occurred in gpuScatteredRead";
		FILE_LOG(logERROR) << "Error: " << errCESTRing(cudaGetLastError());
		system("pause");		
	}
}

template <typename Ftype>
__global__
	void _updateSamples(const Ftype* d_feature, Sample *d_sample, RFTrainNode *d_nodes, const uint dataCount,const uint nodeBegin, const uint nodeEnd)
{
	const int idx = ( blockIdx.x*gridDim.y + blockIdx.y ) * blockDim.x + threadIdx.x;

	if (idx < dataCount )
	{
		int node = binarySearch(idx, d_nodes, nodeBegin, nodeEnd, 0);
		if (d_nodes[ node ].didUpdate)
			d_sample[idx].bestTraj = CHILD( d_feature[idx], d_nodes[ node ].bestThresh);
	}
}

template <typename Ftype>
void updateSamples(const Ftype* d_feature, GPUDevice *gpuDevice, const uint nodeBegin, const uint nodeEnd, cudaStream_t stream)
{
	const float THREADS = 512; 
	const float BLOCKS = ceil(gpuDevice->numSamples()/THREADS);
	uint sqrtVal = ceil(sqrt(BLOCKS));        
	dim3 numBlocks(sqrtVal,sqrtVal);
	dim3 numThreads(THREADS);

	_updateSamples<<< numBlocks, numThreads , 0, stream >>>(
		d_feature, gpuDevice->samples(), gpuDevice->nodes(),	gpuDevice->numSamples(), nodeBegin, nodeEnd);

	cudaStreamSynchronize(stream);
	if (cudaGetLastError()!=cudaSuccess){ 
		FILE_LOG(logERROR) << "Error occurred in updateSamples";
		FILE_LOG(logERROR) << "Error: " << errCESTRing(cudaGetLastError());
		system("pause");		
	}
}

__global__
	void _resetUpdateFlag(RFTrainNode *nodes, const uint numNodes)
{
	const int idx = ( blockIdx.x*gridDim.y + blockIdx.y ) * blockDim.x + threadIdx.x;

	if (idx < numNodes)
		nodes[idx].didUpdate = 0;
}

void resetUpdateFlag(RFTrainNode *nodes, const uint numNodes, cudaStream_t stream)
{
	const float THREADS = 32; 
	const float BLOCKS = ceil(numNodes/THREADS);
	uint sqrtVal = ceil(sqrt(BLOCKS));        
	dim3 numBlocks(sqrtVal,sqrtVal);
	dim3 numThreads(THREADS);

	_resetUpdateFlag<<< numBlocks, numThreads , 0, stream >>>
		(nodes,	numNodes);

	cudaStreamSynchronize(stream);
	if (cudaGetLastError()!=cudaSuccess){ 
		FILE_LOG(logERROR) << "Error occurred in resetUpdateFlag";
		FILE_LOG(logERROR) << "Error: " << errCESTRing(cudaGetLastError());
		system("pause");		
	}
}

//--- We use macro expansions of templated function stubs to force the nvcc compiler to generate object code
// for all the relevant functions because otherwise they wont be generated at all

#define MAKE_SCATTER(Ftype) \
void gpuScatteredWrite(Ftype* dst, const Ftype *src, const uint *index, const uint dataCount, cudaStream_t stream) \
{gpuScatteredWrite<Ftype,Ftype##4>(dst, src, index, dataCount, stream);} \
void gpuScatteredRead(Ftype* dst, const Ftype *src, const uint *index, const uint dataCount, cudaStream_t stream) \
{gpuScatteredRead<Ftype,Ftype##4>(dst, src, index, dataCount, stream);} \

MAKE_SCATTER(char);
MAKE_SCATTER(short);
MAKE_SCATTER(int);
MAKE_SCATTER(float);

#define MAKE_UPDATE(Ftype) template void updateSamples<Ftype> (const Ftype* d_feature, GPUDevice *gpuDevice, const uint nodeBegin, const uint nodeEnd, cudaStream_t stream)

MAKE_UPDATE(char);
MAKE_UPDATE(short);
MAKE_UPDATE(int);
MAKE_UPDATE(float);
