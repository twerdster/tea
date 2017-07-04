#ifndef KERNELS
#define KERNELS

#include "defines.h"
#include "log.h"
class GPUDevice;

// Used when pushing and pulling data onto the device under the index mapping
#define MAKE_SCATTER_CALL(Ftype) \
void gpuScatteredWrite(Ftype* dst, const Ftype *src, const uint *index, const uint dataCount, cudaStream_t stream);\
void gpuScatteredRead(Ftype* dst, const Ftype *src, const uint *index, const uint dataCount, cudaStream_t stream);

MAKE_SCATTER_CALL(char);
MAKE_SCATTER_CALL(short);
MAKE_SCATTER_CALL(int);
MAKE_SCATTER_CALL(float);

// Builds a histogram for each node by sending data to left or right leaves depending on thresh <= feature
#define MAKE_BUILD_HISTOGRAM_CALL(ftype) \
	double buildHistogram(NodeType ntype, HistType htype, float thresh, const ftype *d_feature, GPUDevice *gpuDevice, \
	const RFTrainNode *nodes, const uint nodeBegin, const uint nodeEnd, cudaStream_t stream);

MAKE_BUILD_HISTOGRAM_CALL(char);
MAKE_BUILD_HISTOGRAM_CALL(short);
MAKE_BUILD_HISTOGRAM_CALL(int);
MAKE_BUILD_HISTOGRAM_CALL(float);

//template <class Ftype>
//double buildHistogram(NodeType ntype, HistType htype, float thresh, const Ftype *d_feature, GPUDevice *gpuDevice, const RFTrainNode *nodes, const uint nodeBegin, const uint nodeEnd, cudaStream_t stream);

// Computes the entropies and replaces those nodes with the best entropy
void computeEntropies(const bool updateNextLevel, const int fId, float thresh, GPUDevice *gpuDevice, const uint nodeBegin, const uint nodeEnd, cudaStream_t stream);

// Update each samples bestTraj with the current best classification
template <typename Ftype> 
void updateSamples(const Ftype* d_feature, GPUDevice *gpuDevice, const uint nodeBegin, const uint nodeEnd, cudaStream_t stream);

// Reset the flag for all the nodes in the defined level
void resetUpdateFlag(RFTrainNode *nodes, const uint numNodes, cudaStream_t stream);

const char * errCESTRing(int errCode);

#endif //KERNELS