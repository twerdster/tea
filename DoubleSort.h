#include "defines.h"

//This performs a double swap for whatever index array is here
void swap(Sample *a, uint *b, int i, int j)
{
	Sample tmpa = a[i];
	a[i] = a[j];
	a[j] = tmpa;

	int tmpb = b[i];
	b[i] = b[j];
	b[j] = tmpb;
}

//This function sorts a subset of samples according to 0 on the left and 1 on the right
int partition(Sample *a, uint *b, int start, int end)
{
	int i = start; // index of left-to-right scan
	int k = end;   // index of right-to-left scan

	if (end - start < 1)                  
		return i;

	uint pivot = 0; // We choose 0 to be our pivot. We could obviously have used 1 instead. 				

	while (k > i) 
	{
		while (i <= end && k > i && a[i].bestTraj <= pivot) i++;                                    
		while (k >= start && k >= i && a[k].bestTraj > pivot) k--;
		if (k > i) swap(a,b, i, k);
	}
	if (k >= start) 
		swap(a,b, start, k);  

	return k+1;
}