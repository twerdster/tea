#ifndef _FEATURE
#define _FEATURE

#include <fstream>
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <omp.h>
#include "defines.h"
#include "Clock.h"
#include "log.h"

template <class Ftype>
class Feature 
{
private:
	bool _processed;
	int _id;
	Ftype *_h_data; // This needs to be replaced by a proper constructor
	std::string _strFeatureBaseName;
	long _numSamples;
	//const uint *_index; //Under no circumstances may Feature alter the contents of the index

public:
	Feature():
	  _id(-1),_processed(false),_h_data(0),
		  _numSamples(0){}//,_index(0){}

	  //Here we call a features init method which is like a constructor.
	  //Im not sure what the best design pattern here is in general but I like the init method.
	  //Each data array has to be allocated using cudaHostAlloc
	  void init(std::string strFeatureBaseName, long numSamples) //, const uint *index) 
	  {
		  _strFeatureBaseName = strFeatureBaseName;
		  _numSamples = numSamples;
		  //_index = index;
		  
		  cudaError ce = cudaHostAlloc(&_h_data, _numSamples*sizeof(Ftype),cudaHostAllocPortable ); 
	  }

	  //We could make this threaded but it might not be worth it because disk requests are serialized anyway ..?
	  //unless  some kind of raid system is used...?
	  void load(int fId)
	  {
		  //load a file using featureBaseName
		  Clock clk;
		  char fileName[256];
		  sprintf(fileName, "%s%.4i.feat", _strFeatureBaseName.c_str(), fId);
		  std::ifstream ifs(fileName, std::ios::binary);
		  _id = fId;		

		  //If everything has been setup and the file has been found and opened
		  //then read from it and put it in the data slot
		  if (!ifs.fail()) {
			  clk.tic();
			  for (int i = 0; i < _numSamples*sizeof(Ftype); i+=BLOCKSIZE_64) //if we use 2b examples and int i (not long i) then this would overflow at i+=BLOCKSIZE_64 and the loop would get stuck
			  {
				  int dataCount = std::min<int>(BLOCKSIZE_64, _numSamples*sizeof(Ftype) - i);
				  ifs.read(reinterpret_cast<char*>(_h_data) + i, dataCount);
				  //Should add some kind of message if the file is too short.
			  }				  
			  FILE_LOG(LOG2) <<"Load time for Feature " << fId << " = " << clk.toc() << "s";
		  }
		  else 
		  {
			  _id = -1;
			  FILE_LOG(logERROR) << "A problem occured while loading " << fileName;

		  }

		  ifs.close();
	  }

	  // const uint* getIndex()	{ return _index; }

	  const Ftype* getDataPtr() { return _h_data; }
	  int	getId()				{ return _id; }

	  ~Feature()
	  {		  
		  if (_h_data)
			  cudaFreeHost(_h_data);
	  }
};



#endif
