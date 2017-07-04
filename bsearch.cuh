#include "kernels.h"

inline __device__ 
	int binarySearch(const int idx, const RFTrainNode* d_nodes, const uint nodeBegin, const uint nodeEnd, const int idxOffset)
{
	int imin = nodeBegin;
	int imax = nodeEnd, imid;

	//Could add a guess of the initial imid. Might speedup the search.
	while (imin<imax)
	{
		imid = (int)floor((imin+imax)/2.0);
		if ( (d_nodes[imid].idxEnd - idxOffset) < idx)
			imin = imid + 1;
		else
			imax = imid;
	}
	return imin;
}