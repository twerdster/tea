#include "kernels.h"
#include "Clock.h"
#include "GPUDevice.h"
#include "bsearch.cuh"
#include "system_utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Histograms are generated per depth level
template <NodeType ntype, class Ftype> 
__global__ 
	void _buildHistogram(float thresh, const Ftype *d_feature, uint *d_histogram, const RFTrainNode *d_nodes, const Sample *d_sample, const uint numClasses, const uint dataCount, const uint nodeBegin,const uint nodeEnd, const int idxBegin)
{   
	int idx = ( blockIdx.x*gridDim.y + blockIdx.y ) * blockDim.x + threadIdx.x;
	
	if (idx < dataCount)   
	{
		if ( ntype == PARENT_NODE || CHILD(d_feature[idx], thresh) ) 
		{
			int node = binarySearch(idx, d_nodes, nodeBegin, nodeEnd, idxBegin);			
			atomicAdd( &d_histogram[ (node - nodeBegin) * numClasses + LABEL(d_sample[idx])  ]  , 1 ); // NOTE: will overflow for more than 4bn samples.
		}
	}
}


template <NodeType ntype, class Ftype>
double buildHistogram(float thresh, const Ftype *d_feature, GPUDevice *gpuDevice, const RFTrainNode *nodes, const uint nodeBegin, const uint nodeEnd, cudaStream_t stream)
{   
	int idxBegin = nodes[nodeBegin].idxBegin; // Samples begin index
	int idxEnd = nodes[nodeEnd].idxEnd; //Samples end index
	int dataCount = idxEnd - idxBegin + 1;
	cudaError ce = cudaGetLastError();

	if (dataCount<=0) 
		return 0;

	Clock clk;

	uint *d_histogram;
	const Sample *d_sample = gpuDevice->samples();
	const RFTrainNode *d_nodes = gpuDevice->nodes();
	const uint numClasses = gpuDevice->numClasses();

	const float THREADS = 256;
	const float BLOCKS = ceil((dataCount/1.0f)/THREADS);
	uint sqrtVal = ceil(sqrt(BLOCKS));        
	dim3 numBlocks(sqrtVal,sqrtVal);
	dim3 numThreads(THREADS);

	clk.tic();
	cudaStreamSynchronize(stream);

	d_histogram  =  gpuDevice->histograms()  + ((ntype == CHILD_NODE)? gpuDevice->histWords()/2 : 0);

	cudaMemsetAsync(d_histogram, 0, gpuDevice->histBytes()/2, stream); 		

	_buildHistogram<ntype, Ftype> <<< numBlocks, numThreads , 0, stream >>> 
		(thresh, d_feature + idxBegin, d_histogram, d_nodes, d_sample  + idxBegin, numClasses, dataCount, nodeBegin,nodeEnd,idxBegin); 

	cudaStreamSynchronize(stream);
	if (cudaGetLastError()!=cudaSuccess){ 
		FILE_LOG(logERROR) << "Error occurred in buildHistogram";
		FILE_LOG(logERROR) << "Error: " << errCESTRing(cudaGetLastError());
		discardSystemResult("pause");		
	}
	return clk.toc();
}


//Count a byte into shared-memory storage
inline __device__ void addByte(uchar *s_ThreadBase, uint data){
	s_ThreadBase[UMUL(data & 0x3FU, HISTOGRAM64_THREADBLOCK_SIZE)]++;
}

template <NodeType ntype, class Ftype, class Ftype4>
__global__ 
	void _buildHistogram64(const float thresh, const Ftype *d_feature, uint *d_PartialHistograms, const Sample *d_sample, const uint numClasses, const uint dataCount)
{
		//Encode thread index in order to avoid bank conflicts in s_Hist[] access:
		//each group of SHARED_MEMORY_BANKS threads accesses consecutive shared memory banks
		//and the same bytes [0..3] within the banks
		//Because of this permutation block size should be a multiple of 4 * SHARED_MEMORY_BANKS
		const uint threadPos = 
			( (threadIdx.x & ~(SHARED_MEMORY_BANKS * 4 - 1)) << 0 ) | // leave bits 7,8 in 7,8
			( (threadIdx.x &  (SHARED_MEMORY_BANKS     - 1)) << 2 ) | // move 1,2,3,4 to 3,4,5,6
			( (threadIdx.x &  (SHARED_MEMORY_BANKS * 3    )) >> 5 ); // move bits 5,6 to positions 1,2

		//Per-thread histogram storage
		__shared__ uchar s_Hist[HISTOGRAM64_THREADBLOCK_SIZE * HISTOGRAM64_BIN_COUNT];
		uchar *s_ThreadBase = s_Hist + threadPos;

		//Initialize shared memory (writing 32-bit words)
#pragma unroll
		for(uint i = 0; i < (HISTOGRAM64_BIN_COUNT / 4); i++)
			((uint *)s_Hist)[threadIdx.x + i * HISTOGRAM64_THREADBLOCK_SIZE] = 0;

		//Read data from global memory and submit to the shared-memory histogram
		//Since histogram counters are byte-sized, every single thread can't do more than 255 submission
		__syncthreads();
		for(uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount/sizeof(SampleType4); pos += UMUL(blockDim.x, gridDim.x))
		{		
			/*			
			for (uint szPos = 0; szPos < sizeof(SampleType4) && pos*sizeof(SampleType4)+szPos < dataCount; szPos++)
			{  
				if (ntype == PARENT_NODE || CHILD(d_feature[pos*sizeof(SampleType4)+szPos], thresh) )			
					addByte(s_ThreadBase, LABEL(d_sample[pos*sizeof(SampleType4)+szPos]) ); 
			}*/
			
			const SampleType4 sample = ((SampleType4*)d_sample)[pos];			
			const Ftype4 feature = ((Ftype4*)d_feature)[pos];
						
			if ( (ntype == PARENT_NODE || CHILD(feature.x, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.x) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.y, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.y) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.z, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.z) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.w, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.w) ); 			
		}

		//Accumulate per-thread histograms into per-block and write to global memory
		__syncthreads();
		if(threadIdx.x < numClasses)
		{
			uchar *s_HistBase = s_Hist + UMUL(threadIdx.x, HISTOGRAM64_THREADBLOCK_SIZE);

			uint sum = 0;
			uint pos = 4 * (threadIdx.x & (SHARED_MEMORY_BANKS - 1));

#pragma unroll
			for(uint i = 0; i < (HISTOGRAM64_THREADBLOCK_SIZE / 4); i++){
				sum += 
					s_HistBase[pos + 0] +
					s_HistBase[pos + 1] +
					s_HistBase[pos + 2] +
					s_HistBase[pos + 3];
				pos = (pos + 4) & (HISTOGRAM64_THREADBLOCK_SIZE - 1);
			}

			d_PartialHistograms[blockIdx.x * numClasses + threadIdx.x] = sum;
		}
}

template <NodeType ntype, class Ftype, class Ftype4>
__global__ 
	void _buildHistogram256(const float thresh, const Ftype *d_feature, uint *d_PartialHistograms, const Sample *d_sample, const uint numClasses, const uint dataCount)
{
	//Per-warp subhistogram storage
	__shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
	uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

	//Clear shared memory storage for current threadblock before processing
#pragma unroll
	for(uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
		s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;

	__syncthreads();
	for(uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount/sizeof(SampleType4); pos += UMUL(blockDim.x, gridDim.x))
	{		
		/*
		for (uint szPos = 0; szPos < sizeof(SampleType4) && pos*sizeof(SampleType4)+szPos < dataCount; szPos++)
		{        
			if (ntype == PARENT_NODE || CHILD(d_feature[pos*sizeof(SampleType4)+szPos], thresh) )							
			atomicAdd(s_WarpHist + (LABEL(d_sample[pos*sizeof(SampleType4)+szPos]) & 0xFFU), 1); 	
		}	*/
			const SampleType4 sample = ((SampleType4*)d_sample)[pos];			
			const Ftype4 feature = ((Ftype4*)d_feature)[pos];
						
			if ( (ntype == PARENT_NODE || CHILD(feature.x, thresh) ) )	atomicAdd(s_WarpHist + (LABEL(*(Sample*)&sample.x) & 0xFFU), 1); 
			if ( (ntype == PARENT_NODE || CHILD(feature.y, thresh) ) )	atomicAdd(s_WarpHist + (LABEL(*(Sample*)&sample.y) & 0xFFU), 1); 
			if ( (ntype == PARENT_NODE || CHILD(feature.z, thresh) ) )	atomicAdd(s_WarpHist + (LABEL(*(Sample*)&sample.z) & 0xFFU), 1); 
			if ( (ntype == PARENT_NODE || CHILD(feature.w, thresh) ) )	atomicAdd(s_WarpHist + (LABEL(*(Sample*)&sample.w) & 0xFFU), 1); 
	}

	//Merge per-warp histograms into per-block and write to global memory
	__syncthreads();
	for(uint bin = threadIdx.x; bin < numClasses; bin += HISTOGRAM256_THREADBLOCK_SIZE){
		uint sum = 0;

		for (uint i = 0; i < WARP_COUNT; i++)
			sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT];

		d_PartialHistograms[blockIdx.x * numClasses + bin] = sum;
	}
}

// Merge output
// Run one threadblock per bin; each threadbock adds up the same bin counter 
// from every partial histogram. Reads are uncoalesced, but mergeHistogram64
// takes only a fraction of total processing time
__global__ 
	void _mergeHistogram( uint *d_Histogram, uint *d_PartialHistograms, const uint histogramCount, const uint numClasses)
{
	__shared__ uint data[MERGE_THREADBLOCK_SIZE];

	uint sum = 0;
	for(uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)
		sum += d_PartialHistograms[blockIdx.x + i * numClasses];
	data[threadIdx.x] = sum;

	for(uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1){
		__syncthreads();
		if(threadIdx.x < stride)
			data[threadIdx.x] += data[threadIdx.x + stride];
	}

	if(threadIdx.x == 0)
		d_Histogram[blockIdx.x] = data[0];
}

template <NodeType ntype, HistType htype, class Ftype, class Ftype4>
double buildHistogramFast(float thresh, const Ftype *d_feature, GPUDevice *gpuDevice, const RFTrainNode *nodes, const uint nodeBegin, const uint nodeEnd, cudaStream_t stream)
{   	
	uint *d_histogram;
	volatile  uint histogramCount;
	const Sample *d_sample = gpuDevice->samples();
	const uint numClasses = gpuDevice->numClasses();
	const uint nodeCount = nodeEnd - nodeBegin + 1;
	uint *d_PartialHistograms = gpuDevice->partialHistograms();

	d_histogram = gpuDevice->histograms()  + ((ntype == CHILD_NODE)?gpuDevice->histWords()/2:0);
	cudaMemsetAsync(d_histogram, 0, gpuDevice->histBytes()/2, stream); 	

	Clock clk;
	clk.tic();
	cudaStreamSynchronize(stream);
	int cnt=0;
	
	for ( int nIdx = 0; nIdx < nodeCount; nIdx++)
	{ 
		int idxBegin = nodes[nIdx + nodeBegin].idxBegin;	// Samples begin index
		int idxEnd = nodes[nIdx + nodeBegin].idxEnd;		//Samples end index
		int dataCount = idxEnd - idxBegin + 1;
		d_histogram  =  gpuDevice->histograms()  + ((ntype == CHILD_NODE)?gpuDevice->histWords()/2:0) + nIdx*numClasses;

		if (dataCount<=0) continue;
		cnt++;
		switch (htype)
		{
		case HIST_64:		
			histogramCount = iDivUp(dataCount, HISTOGRAM64_THREADBLOCK_SIZE * iSnapDown(255, sizeof(SampleType4)));
			_buildHistogram64<ntype, Ftype, Ftype4> <<< histogramCount, HISTOGRAM64_THREADBLOCK_SIZE, 0, stream >>>  
				(thresh, d_feature + idxBegin, d_PartialHistograms, d_sample + idxBegin, numClasses, dataCount);	
			break;
		case HIST_256:
			histogramCount = PARTIAL_HISTOGRAM256_COUNT;
			_buildHistogram256<ntype, Ftype, Ftype4> <<< histogramCount, HISTOGRAM256_THREADBLOCK_SIZE, 0, stream>>>
				(thresh, d_feature + idxBegin, d_PartialHistograms, d_sample + idxBegin, numClasses, dataCount);
			break;				
		}

		_mergeHistogram<<< numClasses, MERGE_THREADBLOCK_SIZE, 0 , stream>>> 
			(d_histogram, d_PartialHistograms, histogramCount, numClasses);
		
	}

	cudaStreamSynchronize(stream);	
	if (cudaGetLastError()!=cudaSuccess){ 
		FILE_LOG(logERROR) << "Error occurred in buildHistogramFast";
		FILE_LOG(logERROR) << "Error: " << errCESTRing(cudaGetLastError());
		discardSystemResult("pause");		
	}
	return clk.toc();
}

template <class Ftype, class Ftype4 > 
double buildHistogram(NodeType ntype, HistType htype, float thresh, const Ftype *d_feature, GPUDevice *gpuDevice, const RFTrainNode *nodes, const uint nodeBegin, const uint nodeEnd, cudaStream_t stream) 
{ 
	if (ntype == CHILD_NODE) 
		switch (htype) 
		{ 
		case HIST_64:		return buildHistogramFast<CHILD_NODE, HIST_64, Ftype, Ftype4>	(thresh, d_feature, gpuDevice, nodes, nodeBegin, nodeEnd, stream);	 
		case HIST_256:		return buildHistogramFast<CHILD_NODE, HIST_256, Ftype, Ftype4>	(thresh, d_feature, gpuDevice, nodes, nodeBegin, nodeEnd, stream);	 
		case HIST_REGULAR:	return buildHistogram<CHILD_NODE, Ftype>						(thresh, d_feature, gpuDevice, nodes, nodeBegin, nodeEnd, stream);	 
		} 
		 
	if (ntype == PARENT_NODE) 
		switch (htype) 
		{ 
		case HIST_64:		return buildHistogramFast<PARENT_NODE, HIST_64, Ftype, Ftype4>  (thresh, d_feature, gpuDevice, nodes, nodeBegin, nodeEnd, stream); 
		case HIST_256:		return buildHistogramFast<PARENT_NODE, HIST_256, Ftype, Ftype4> (thresh, d_feature, gpuDevice, nodes, nodeBegin, nodeEnd, stream);	 
		case HIST_REGULAR:	return buildHistogram<PARENT_NODE, Ftype>						(thresh, d_feature, gpuDevice, nodes, nodeBegin, nodeEnd, stream);	 
		} 
		 
	return -1; 
} 

#define MAKE_BUILD_HISTOGRAM(ftype) \
	double buildHistogram(NodeType ntype, HistType htype, float thresh, const ftype *d_feature, GPUDevice *gpuDevice, \
	const RFTrainNode *nodes, const uint nodeBegin, const uint nodeEnd, cudaStream_t stream) { return buildHistogram<ftype, ftype ## 4> \
	(ntype, htype, thresh, d_feature, gpuDevice, nodes, nodeBegin, nodeEnd, stream); } \

MAKE_BUILD_HISTOGRAM(char);
MAKE_BUILD_HISTOGRAM(short);
MAKE_BUILD_HISTOGRAM(int);
MAKE_BUILD_HISTOGRAM(float);
/*
int main (int argc, char ** argv) { 
double a;
cudaStream_t stream;
RFTrainNode *nodes=0;
char* _d_data=0;
a=buildHistogram(PARENT_NODE, HIST_64, 0, _d_data, 0, nodes, 0, 0,stream);
	return a;
}
*/
/*
//MAKE SURE THIS WORKS THEN TRY UINT4
			const uint4 sample_ = ((uint4*)d_sample)[pos];
			uchar4 sample;
			const uint4 feature_ = ((uint4*)d_feature)[pos];
			char4 feature;
			
			sample = *(uchar4*)&(sample_.x);
			feature = *(char4*)&(feature_.x);
			if ( (ntype == PARENT_NODE || CHILD(feature.x, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.x) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.y, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.y) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.z, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.z) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.w, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.w) ); 	
sample = *(uchar4*)&(sample_.y);
			feature = *(char4*)&(feature_.y);
			if ( (ntype == PARENT_NODE || CHILD(feature.x, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.x) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.y, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.y) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.z, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.z) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.w, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.w) ); 
			sample = *(uchar4*)&(sample_.z);
			feature = *(char4*)&(feature_.z);
			if ( (ntype == PARENT_NODE || CHILD(feature.x, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.x) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.y, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.y) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.z, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.z) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.w, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.w) ); 
			sample = *(uchar4*)&(sample_.w);
			feature = *(char4*)&(feature_.w);
			if ( (ntype == PARENT_NODE || CHILD(feature.x, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.x) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.y, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.y) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.z, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.z) ); 
			if ( (ntype == PARENT_NODE || CHILD(feature.w, thresh) ) )	addByte(s_ThreadBase, LABEL(*(Sample*)&sample.w) ); 
			

*/
/*
#define ADD(kid,samp,n)\
		if ( ntype == PARENT_NODE || CHILD((kid), thresh) )  \
		{ \
			int node = binarySearch(idx*16 + n, d_nodes, nodeBegin, nodeEnd, idxBegin);			 \
			atomicAdd( &d_histogram[ (node - nodeBegin) * numClasses + LABEL((samp))  ]  , 1 ); \
		} \


// Histograms are generated per depth level
template <NodeType ntype> 
__global__ 
	void _buildHistogram(float thresh, const SAMPLE_TYPE *d_feature, uint *d_histogram, const RFTrainNode *d_nodes, const Sample *d_sample, const uint numClasses, const uint dataCount, const uint nodeBegin,const uint nodeEnd, const int idxBegin)
{   
	int idx = ( blockIdx.x*gridDim.y + blockIdx.y ) * blockDim.x + threadIdx.x;

	if (idx < dataCount)   
	{
			const uint4 sample_ = ((uint4*)d_sample)[idx];
			uchar4 sample;
			const uint4 feature_ = ((uint4*)d_feature)[idx];
			char4 feature;
			
			sample = *(uchar4*)&(sample_.x);
			feature = *(char4*)&(feature_.x);
			ADD(feature.x,*(Sample*)&sample.x, 0);
			ADD(feature.y,*(Sample*)&sample.y, 1);			
			ADD(feature.z,*(Sample*)&sample.z, 2);			
			ADD(feature.w,*(Sample*)&sample.w, 3);	

			sample = *(uchar4*)&(sample_.y);
			feature = *(char4*)&(feature_.y);
			ADD(feature.x,*(Sample*)&sample.x, 4);
			ADD(feature.y,*(Sample*)&sample.y, 5);			
			ADD(feature.z,*(Sample*)&sample.z, 6);			
			ADD(feature.w,*(Sample*)&sample.w, 7);	

			sample = *(uchar4*)&(sample_.z);
			feature = *(char4*)&(feature_.z);
			ADD(feature.x,*(Sample*)&sample.x, 8);
			ADD(feature.y,*(Sample*)&sample.y, 9);			
			ADD(feature.z,*(Sample*)&sample.z, 10);			
			ADD(feature.w,*(Sample*)&sample.w, 11);	

			sample = *(uchar4*)&(sample_.w);
			feature = *(char4*)&(feature_.w);
			ADD(feature.x,*(Sample*)&sample.x, 12);
			ADD(feature.y,*(Sample*)&sample.y, 13);			
			ADD(feature.z,*(Sample*)&sample.z, 14);			
			ADD(feature.w,*(Sample*)&sample.w, 15);	
	}
}
*/
