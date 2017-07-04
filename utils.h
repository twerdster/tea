
#ifndef UTILS
#define UTILS

#include <assert.h>
#include <queue>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
//#include <tchar.h>
#include "defines.h"
#include "log.h"

//This should definitely go inside the tree struct
//In fact the whole tree thing needs to be better thought out and organized but this
//is last minute design/build rubbish!
void loadTree(Tree &tree)
{
	tree.th =				(TreeHeader*)	(tree.treeAlloc);
	tree.nodes =			(RFNode*)		((char*)tree.th				+ sizeof(TreeHeader));
	tree.compactLeaves =	(CompactLeaf*)	((char*)tree.nodes			+ sizeof(RFNode)* tree.th->totalNodes);
	tree.histograms =		(uchar*)		((char*)tree.compactLeaves	+ sizeof(CompactLeaf)* tree.th->totalHists);	
}

//tree.alloc needs to be released by deleteTree outside this function
void loadTree(std::string treeName, Tree &tree)
{	
	std::ifstream ifs(treeName.c_str(), std::ios::binary);

	TreeHeader th;
	ifs.read((char*)&th, sizeof(TreeHeader));
	ifs.seekg(0,std::ios_base::beg);

	int allocSize = sizeof(TreeHeader) + sizeof(RFNode)*th.totalNodes + sizeof(CompactLeaf)*th.totalHists + sizeof(uchar)*th.totalHists*th.numClasses;
	tree.treeAlloc = new char[allocSize];
	ifs.read((char*)tree.treeAlloc,allocSize);

	loadTree(tree);

	ifs.close();
}

void deleteTree(Tree &tree)
{
	delete[] tree.treeAlloc;
}

void readPairs(std::vector<int> &weights, std::string fileName)
{
	int _numClasses = weights.size();
	std::ifstream ifs(fileName.c_str());
	while (!ifs.eof())
	{
		int i, weight;
		ifs >> i;				
		ifs >> weight;
		if (i >= 0 && i < _numClasses)
		weights[i] = weight;
	}
	ifs.close();
}

template <class T>
void readList(T * dst, int numElements, std::string strFileName)
{
	std::ifstream ifs(strFileName.c_str(), std::ios::binary);

	if (!ifs.fail()) {
		long i=0;

		for (i = 0; i < numElements*sizeof(T); i+=BLOCKSIZE_64) 
		{
			long dataCount = std::min<int>(BLOCKSIZE_64, numElements*sizeof(T) - i);
			ifs.read(reinterpret_cast<char*>(dst) + i, dataCount);
			if (ifs.fail())
			{
				FILE_LOG(logERROR) << "Less than "<< numElements <<" elements read from " << strFileName.c_str();
				break;
			} else
				if (ifs.eof())
				{
					FILE_LOG(LOG1) << "File " << strFileName.c_str()<< " read.";
					break;
				}
		}					
	}
	else
		FILE_LOG(logERROR) << "File: " << strFileName.c_str() <<" failed to open";
	ifs.close();
}

#endif // UTILS