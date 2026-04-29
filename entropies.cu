#include "kernels.h"
#include "GPUDevice.h"
#include "system_utils.h"
#include <assert.h>
#include <cuda_runtime.h>


#define eps 1e-5

inline __device__ 
	float sumSqrInvW(uint nClasses, const uint *arr, uint* W)
{
	float tot=0;
	for (int i=0; i<nClasses; i++)
		if (W[i]>0) 
			tot+=((float)arr[i]*(float)arr[i])/((float)W[i]*(float)W[i]);
	return tot;
}

inline __device__ 
	float sumSqr(uint nClasses, const uint *arr)
{
	float tot=0;
	for (int i=0; i<nClasses; i++)
		tot+=((float)arr[i]*(float)arr[i]);
	return tot;
}


inline __device__ 
	float sumInvW(uint nClasses, const uint *arr, uint* W)
{
	float tot=0;
	for (int i=0; i<nClasses; i++)
		if (W[i]>0) 
			tot+=((float)arr[i]/((float)W[i])); 
	return tot;
}

inline __device__ 
	float sum(uint nClasses, const uint *arr)
{
	float tot=0;
	for (int i=0; i<nClasses; i++)
		tot+=((float)arr[i]);
	return tot;
}


//Returns squared norm with each elemented divided by a value
inline __device__
	void gini(uint nClasses, const uint *arr, float val, float &tot) 
{
	tot=0;
	if (val<=eps) {
		tot=1;
		return;
	}

	for (int i=0;i<nClasses;i++) 
		tot+=((float)arr[i]*(float)arr[i]);

	tot=1.0f-tot/(val*val);
}

inline __device__
	void gini(uint nClasses, const uint *arr, float &I, float& S, uint *W) 
{
	S=sumInvW(nClasses,arr,W);
	I=0;
	if (S<=0) 
		return; //When the gain is calculated the S*I will give 0.

	I=sumSqrInvW(nClasses,arr,W);
	I=1.0f-I/(S*S);
}

inline __device__
	void shanon(uint nClasses, const uint *arr, float &I, float& S, uint *W) 
{
	S=sumInvW(nClasses,arr,W);
	I=0;
	if (S<=0) 
		return; //When the gain is calculated the S*I will give 0.

	for (int i=0; i<nClasses; i++)
		if (W[i]>0) 
			I+=( ((float)arr[i]/(float)W[i]) * log2(eps + ((float)arr[i]/(float)W[i])) );
	
	I= log2(S) - (1.0f/S) * I;
}

//USES APRIORI W
__global__
	void _computeEntropies(const int fId, const float thresh, RFTrainNode *nodes, uint* histograms, uint* W, uint nS, const int numClasses, const int childOffset, const int nodeCount, const int nodeBegin, const bool updateNextLevel)
{
	const int idx = ( blockIdx.x*gridDim.y + blockIdx.y ) * blockDim.x + threadIdx.x;

	if (idx < nodeCount)
	{//printf("\nIN %i\n",idx);
		int depth = log2f(nodeCount);
		//This is the index into the nodes array
		const uint nodeIdx = idx + nodeBegin;

		//These are the indices into the histograms
		const uint idxParent = idx * numClasses;
		const uint idxRightChild  = idxParent + childOffset; // The right child is the one who tested positive in the threshold comparison
		const uint idxLeftChild  = idxRightChild; // The left child is calculated in place from the right child and parent into the right childs position

		// Calculate Gini indices for left and right
		float iL,iR,sR,sL;		

		//DEBUG-REMOVE THIS
		//for (int j=0; j<numClasses;j++)
		//{
		//	printf("%i: %i -> (%i, %i)\n",j, histograms[idxParent + j],histograms[idxParent + j] - histograms[idxRightChild + j],histograms[idxRightChild + j]);
		//}
		//-----


		// ------ RIGHT CHILD ------
		shanon(numClasses , histograms + idxRightChild, iR, sR, W);						
		// -------------------------

		//------- LEFT CHILD----------		
		for (int j = 0; j<numClasses; j++) 
			histograms[idxLeftChild + j] = histograms[idxParent + j] - histograms[idxRightChild + j];		// Parent is always larger than child if histograms are not modified
		shanon(numClasses , histograms + idxRightChild, iL, sL, W);								
		//-----------------------------

		

		const float I = nodes[nodeIdx].entropy; // For the very first node in the whole tree this initial value is irrelevant except that it be the same across all devices.
		const float deltaI = I - (sL*iL + sR*iR)/(sL+sR +eps);


		// Update current best parameters for the split
		if (deltaI > nodes[nodeIdx].bestDelta) // weird stuff happens if you make this >= instead of >  : the nodes dont sort correctly for folded depths...
		{
			nodes[nodeIdx].bestDelta = deltaI;
			nodes[nodeIdx].bestFeat = fId;
			nodes[nodeIdx].bestThresh = thresh;						
			nodes[nodeIdx].didUpdate = 1;

			if (updateNextLevel) 
			{
				nodes[nodeIdx*2 + 1].entropy = iL;
				nodes[nodeIdx*2 + 2].entropy = iR;
			}
		}
	}
}


//CURRENTLY NOT USED. 
// One node is calculated per thread
__global__
	void _computeEntropies_(const int fId, const float thresh, RFTrainNode *nodes, uint* histograms, uint* W, const int numClasses, const int childOffset, const int nodeCount, const int nodeBegin, const bool updateNextLevel)
{
	const int idx = ( blockIdx.x*gridDim.y + blockIdx.y ) * blockDim.x + threadIdx.x;

	if (idx < nodeCount)
	{
		int depth = log2f(nodeCount);
		//This is the index into the nodes array
		const uint nodeIdx = idx + nodeBegin;

		//These are the indices into the histograms
		const uint idxParent = idx * numClasses;
		const uint idxRightChild  = idxParent + childOffset; // The right child is the one who tested positive in the threshold comparison
		const uint idxLeftChild  = idxRightChild; // The left child is calculated in place from the right child and parent into the right childs position

		// Calculate Gini indices for left and right
		float tot;
		float szParent = nodes[nodeIdx].idxEnd - nodes[nodeIdx].idxBegin + 1;

		// ------ RIGHT CHILD ------
		float szRightChild = 0;
		for (int j = 0; j<numClasses; j++) 
			szRightChild += histograms[idxRightChild + j];

		//gini(numClasses , histograms + idxRightChild, tot0, tot, W);
		gini(numClasses , histograms + idxRightChild, szRightChild + eps, tot);
		//shannon(numClasses , histograms + idxRightChild, szRightChild + eps, tot);		
		const float Ir =  tot;
		// -------------------------

		// ------- Now calculate the left child values
		for (int j = 0; j<numClasses; j++) 
			histograms[idxLeftChild + j] = histograms[idxParent + j] - histograms[idxRightChild + j];

		//------- LEFT CHILD----------
		float szLeftChild = szParent - szRightChild;

		//gini(numClasses , histograms + idxLeftChild, tot0, tot, W);
		gini(numClasses , histograms + idxLeftChild, szLeftChild + eps, tot);
		//shannon(numClasses , histograms + idxRightChild, szLeftChild + eps, tot);		
		const float Il =  tot;
		//-----------------------------

		const float I = nodes[nodeIdx].entropy; // For the very first node in the whole tree this initial value is irrelevant except that it be the same across all devices.
		const float deltaI = I - (szLeftChild * Il + szRightChild * Ir) / (szParent + eps);


		// Update current best parameters for the split
		if (deltaI > nodes[nodeIdx].bestDelta) // weird stuff happens if you make this >= instead of >  : the nodes dont sort correctly for folded depths...
		{
			nodes[nodeIdx].bestDelta = deltaI;
			nodes[nodeIdx].bestFeat = fId;
			nodes[nodeIdx].bestThresh = thresh;						
			nodes[nodeIdx].didUpdate = 1;

			if (updateNextLevel) 
			{
				nodes[nodeIdx*2 + 1].entropy = Il;
				nodes[nodeIdx*2 + 2].entropy = Ir;
			}
		}
	}
}

void computeEntropies(const bool updateNextLevel, const int fId, float thresh, GPUDevice *gpuDevice, const uint nodeBegin, const uint nodeEnd, cudaStream_t stream)
{
	volatile const int nodeCount = nodeEnd - nodeBegin + 1;
	const float THREADS = 8; // we use here a much smaller set of threads because we dont expect to need many until the later levels
	const float BLOCKS = ceil(nodeCount/THREADS);
	volatile uint sqrtVal = ceil(sqrt(BLOCKS));        
	dim3 numBlocks(sqrtVal,sqrtVal);
	dim3 numThreads(THREADS);
	volatile cudaError ce = cudaGetLastError();
	_computeEntropies<<< numBlocks, numThreads , 0, stream >>>(
		fId,
		thresh,
		gpuDevice->nodes(),
		gpuDevice->histograms(),
		gpuDevice->W(),
		gpuDevice->numSamples(),
		gpuDevice->numClasses(),
		gpuDevice->histWords()/2,
		nodeCount,
		nodeBegin,
		updateNextLevel
		);

	cudaStreamSynchronize(stream);	
	if (cudaGetLastError()!=cudaSuccess)
	{ 
		FILE_LOG(logERROR) << "Error occurred in computeEntropies";
		FILE_LOG(logERROR) << "Error: " << errCESTRing(cudaGetLastError());
		discardSystemResult("pause");
	}

}
