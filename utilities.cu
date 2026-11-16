#include "kernels.h"
#include "bsearch.cuh"
#include "GPUDevice.h"
#include "system_utils.h"
#include <assert.h>
#include <cuda_runtime.h>

// Scalar scatter/gather supports final chunks of any size.
template <class Ftype>
__global__ void scatterKernel(Ftype* dst, const Ftype* src, const uint* index,
                              uint count)
{
    uint i = (blockIdx.x*gridDim.y + blockIdx.y)*blockDim.x + threadIdx.x;
    if (i < count) dst[index[i]] = src[i];
}
template <class Ftype, class Ftype4>
void gpuScatteredWrite(Ftype* dst, const Ftype* src, const uint* index,
                       uint count, cudaStream_t stream)
{
    if (!count) return;
    uint side = static_cast<uint>(ceil(sqrt(ceil(count/256.0))));
    scatterKernel<<<dim3(side, side), 256, 0, stream>>>(dst, src, index, count);
    teaCudaCheck(cudaGetLastError(), "scatterKernel");
}
template <class Ftype>
__global__ void gatherKernel(Ftype* dst, const Ftype* src, const uint* index,
                             uint count)
{
    uint i = (blockIdx.x*gridDim.y + blockIdx.y)*blockDim.x + threadIdx.x;
    if (i < count) dst[i] = src[index[i]];
}
template <class Ftype, class Ftype4>
void gpuScatteredRead(Ftype* dst, const Ftype* src, const uint* index,
                      uint count, cudaStream_t stream)
{
    if (!count) return;
    uint side = static_cast<uint>(ceil(sqrt(ceil(count/256.0))));
    gatherKernel<<<dim3(side, side), 256, 0, stream>>>(dst, src, index, count);
    teaCudaCheck(cudaGetLastError(), "gatherKernel");
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
	const float THREADS = 256;
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
		discardSystemResult("pause");
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
		discardSystemResult("pause");
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
