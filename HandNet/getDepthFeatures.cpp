#include <omp.h>
#include <algorithm>
#include <math.h>
#include "mex.h"

using namespace std;

#define MAXMIN(x,lower,upper) ceil(max(min((float)(x),(float)(upper)),(float)(lower)))
#define OFFLIMITS(x,lower,upper) ((x)<(lower) || (x)>(upper))
#define OFFLIMITS2(x,lx,ux,y,ly,uy) ( OFFLIMITS(x,lx,ux) || OFFLIMITS(y,ly,uy) )

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
{ 
	int* i = (int*)mxGetData(prhs[0]);
	int* j = (int*)mxGetData(prhs[1]);
	float* depthImage = (float*)mxGetData(prhs[2]);
	int numRows = (int)mxGetM(prhs[2]);
	int numCols = (int)mxGetN(prhs[2]);
	
	//Contains the offset coords for a feature from the central pixel
	float* coords = (float*)mxGetData(prhs[3]);    
    //Maximum depth between two points on hand
	float maxDiff = (float)mxGetScalar(prhs[4]);
	//Maximum value this particular data type can attain
	float fMAX = (float)mxGetScalar(prhs[5]);
	
	int nF = (int)mxGetN(prhs[3]);
	int nS = (int)mxGetM(prhs[0]);
	plhs[0] = mxCreateNumericMatrix(nS, nF, mxSINGLE_CLASS, mxREAL);
	float *point_features = (float*)mxGetData(plhs[0]);
	
    // An outer loop omp parfor is 10 times faster because less threads are created
    //omp_set_num_threads(6);
    #pragma omp parallel for 
	for (int s = 0; s < nS; s++)
	{
		int row = i[s]-1;
		int col = j[s]-1;
		float z = (float)depthImage[row + col*numRows];
		
		for (int f = 0; f < nF; f++)
		{
			int row1 = (row + coords[0 + f*4]/z);             
			int col1 = (col + coords[1 + f*4]/z);
			int row2 = (row + coords[2 + f*4]/z);
			int col2 = (col + coords[3 + f*4]/z);
                      
            float d1, d2;
            if (OFFLIMITS2(row1, 0, numRows-1, col1, 0, numCols-1)) 
                d1 = 0;
            else 
                d1 = depthImage[(row1 + col1*numRows)];
            
            if (OFFLIMITS2(row2, 0, numRows-1, col2, 0, numCols-1)) 
                d2 = 0;
            else 
                d2 = depthImage[(row2 + col2*numRows)];
            
			float feat;
			if (d1==0 && d2==0) feat = 0;
			else
			if (d1==0) feat = fMAX;
			else
			if (d2==0) feat = -fMAX;
			else
            { 
                 // this maps a value from [-1,1] to [-fMAX, fMAX]
                feat = (d1 - d2) / maxDiff * fMAX;            
                feat = MAXMIN(feat,-fMAX,fMAX);
            }
			point_features[s + f*nS] = feat;
		}
	}
	
    return;
}