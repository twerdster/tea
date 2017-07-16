

#include <assert.h>
#include <boost/thread.hpp>
#include <queue>
#include <iostream>
#include <string>
#include "AtomicQueue.h"
#include "Feature.h"
#include "GPUWorker.h"
#include "FeaturePool.h"
#include "GPUWorkerPool.h"
#include "DoubleSort.h"
#include "utils.h"
#include "Clock.h"
#include <omp.h>
#include "log.h"
#include "errors.h"

boost::mutex broadcastMutex;
boost::condition_variable broadcastConditionVar;
int broadcastCount;

//I chose against abstracting the classes to base classes with pure virtual coz im lazy but everything needs to be properly cleaned up eventually
void* featurePool;
void* gpuWorkerPool;

std::vector<GPUDevice> gpuDevice;
std::vector<RFTrainNode> nodes;
std::vector<int> scoreWeights;

int treeNum;
int maxNodes;
int numClasses;
int numSamples;
int numFeatures;
int poolSize;
int maxDepth;	
int foldingDepth;
int numThresh;
int startDevice, endDevice, numDevices; 
int numStreams;
FeatureType featureType;
WeightType weightType;
std::string baseDir;
std::string treePrefix;

uint *fwdIndexList, *revIndexList;

std::vector<float> threshList;
Sample* sampleList; // Labels and nodetraj together

//Save tree build state
void saveTreeBuildState()
{
	//write nodes to a file with depth;
	//write samples to a file;
	//write  reverseindex to file; fwd can be gnerated from rev
}

//Load tree build state
void loadTreeBuildState()
{
	//load up data from above;
}

//This function saves the tree leaf number every sample has fallen into for the current depth
void generateSampleTrajectoriesFromBuildState(int depth, std::string sampleLabelsName)
{
	int totalNodes = pow(2.0f, depth + 1) - 1; // All the nodes sum to this number
	
	FILE_LOG(LOG1) << "Saving Samples traj list ...";

	std::ofstream ofs(sampleLabelsName.c_str(), std::ios::binary);
	uint * tmpTrajs = new uint[numSamples];	
	memset(tmpTrajs,0,numSamples*sizeof(uint));
	
	Clock clk;

	for (int nIdx = totalNodes/2; nIdx < totalNodes;  nIdx++) 
	{
		for (int s = nodes[nIdx].idxBegin; s <= nodes[nIdx].idxEnd; s++)
		{
			uint traj =  nIdx*2 + sampleList[s].bestTraj + 1;
			tmpTrajs[revIndexList[s]] = traj;		
		}
	}

	ofs.write((char*)tmpTrajs, sizeof(uint)*numSamples);
	ofs.close();

	delete[] tmpTrajs;	
}

//Generate tree from a build state
//The tree generated can be used to test trees at different depths because
//the histogram provided is complete. One simple way of reducing histogram memory by half is to not store intermediate histograms.
void generateTreeFromBuildState(int depth, std::string treeName)
{	
	Clock clk;
	clk.tic();
	int totalNodes = pow(2.0f, depth + 1) - 1; // All the nodes sum to this number
	int totalHists = 2*totalNodes + 1; // There are 2 histograms under every node and one above the root node

	//allocate char* array to size of whole tree
	//set each field to its respective position in array
	Tree tree;
	int allocSize = sizeof(TreeHeader) + sizeof(RFNode)*totalNodes + sizeof(CompactLeaf)*totalHists + sizeof(uchar)*totalHists*numClasses;
	tree.treeAlloc = new char[allocSize];
	tree.th =				(TreeHeader*)	(tree.treeAlloc);
	tree.nodes =			(RFNode*)		((char*)tree.th				+ sizeof(TreeHeader));
	tree.compactLeaves =	(CompactLeaf*)	((char*)tree.nodes			+ sizeof(RFNode)*totalNodes);
	tree.histograms =		(uchar*)		((char*)tree.compactLeaves	+ sizeof(CompactLeaf)*totalHists);
	uint * fullHistograms = new uint[totalHists*numClasses];
	memset(fullHistograms,0,sizeof(uint)*totalHists*numClasses);
	
	for (int nIdx = totalNodes/2; nIdx < totalNodes;  nIdx++) 
	{
		for (int s = nodes[nIdx].idxBegin; s <= nodes[nIdx].idxEnd; s++)
		{
			uint traj =  nIdx*2 + sampleList[s].bestTraj + 1;
			uint label = sampleList[s].label;
			//printf("Node, Traj, Label : %i,%i,%i\n",nIdx,traj,label);
			for (int d = 0; d <= depth + 1; d++ ) 
			{			
				uint hist = (traj >> (depth +1 - d))*numClasses;
				fullHistograms[hist + label]++;
			}
		}
	}

	tree.th->depth = depth;
	tree.th->numClasses = numClasses;
	tree.th->totalNodes = totalNodes;
	tree.th->totalHists = totalHists;

	for (int n = 0; n < tree.th->totalNodes; n++)
	{
		RFNode &node = *(tree.nodes + n);
		node.F = nodes[n].bestFeat;		
		node.thr = nodes[n].bestThresh;
		//printf("Tree Node:%i  F:%i  thr:%f\n",n,node.F,node.thr);
	}

	for (int n = 0; n < tree.th->totalHists; n++)
	{	
		float maxVal=-1;
		int distVal=-1;
		
		for (int j = 0; j < tree.th->numClasses; j++)
		{
			float f = (1.0f/((float)scoreWeights[j]))*(float)fullHistograms[n*tree.th->numClasses + j]; 
			//float f = (float)fullHistograms[n*tree.th->numClasses + j]; 
			//printf("hist[%i + %i] = %f \n",n,j,f);
			if (f>maxVal)
			{
				distVal = j;
				maxVal = f;
			}
		}

		CompactLeaf &cl = tree.compactLeaves[n];
		// NOTE: the following line may be an error because it doesnt take into account the scoreWeights.
		//        should probably correct this. Currently its only a minor error because the label is still correct and
		//        this is whats used at runtime. But in general scoreWeights should definitely be taken into account.
		//        Fixed line should probably be: cl.maxVal = (1.0f/((float)scoreWeights[j]))*(float)fullHistograms[n*tree.th->numClasses + distVal];
		cl.maxVal = fullHistograms[n*tree.th->numClasses + distVal];
		cl.label = distVal;		
		//printf("[%i,%i]\n",(int)cl.label,(int)cl.maxVal);
		
		//uint* itrBegin = fullHistograms + n * tree.th->numClasses;
		//uint* itrEnd   = fullHistograms + (n+1) * tree.th->numClasses;	
		//uint* itrMax = std::max_element(itrBegin,itrEnd);
		
		//cl.maxVal = *itrMax;
		//cl.label = std::distance(itrBegin,itrMax);		
		//printf("(%i,%i),(%i,%i)\n",(int)distVal,(int)maxVal,(int)cl.label,(int)cl.maxVal);
	}

	for (int n = 0; n < tree.th->totalHists; n++)
	{
		CompactLeaf &cl = tree.compactLeaves[n];
		for (int c = 0 ; c < tree.th->numClasses; c++)
		{
			//We compress the histogram into a byte so that the histograms dont take up too much space otherwise we would just do a memcpy here
			// NOTE: this is almost certainly now an error if we are not taking into account scoreWeights.
			//        the result will be an overflow for all the histogram bins which are larger than cl.maxVal
			//        Must fix this this. However can still stay inside a byte. Does not need to be larger.
			//        fixed line should probably be: uchar val = (1.0f/((float)scoreWeights[j]))*((float)(fullHistograms[n*tree.th->numClasses + c]))*255.0f/(float)cl.maxVal;
			uchar val = ((float)(fullHistograms[n*tree.th->numClasses + c]))*255.0f/(float)cl.maxVal;
			tree.histograms[n*tree.th->numClasses + c] = val;
		}
	}


	//Output label values
	ushort * tmpLabels = new ushort[numSamples];
	memset(tmpLabels,0,numSamples*sizeof(ushort));  //SampleType -> This needs to be changed to matc th eoutput label type

	//Determine how many samples fall into a leaf where their label is maximum
	uint numCorrect = 0;
	uint *correct = new uint[numClasses];
	uint *confusion = new uint[numClasses*numClasses];
	uint *total = new uint[numClasses];
	for (int i=0; i< numClasses; i++) 
		for (int j=0; j <numClasses; j++)
			confusion[i + j*numClasses] = correct[i] = total[i] = 0;

	FILE_LOG(LOG1) << "Calculating trees self-success ...";
	for (int nIdx = totalNodes/2; nIdx < totalNodes;  nIdx++) 
	{
		//printf("SNode:%i \n",nIdx);
		for (int s = nodes[nIdx].idxBegin; s <= nodes[nIdx].idxEnd; s++)
		{
			uint traj =  nIdx*2 + sampleList[s].bestTraj + 1;
			uint label = sampleList[s].label;
			tmpLabels[revIndexList[s]] = tree.compactLeaves[traj].label;

			//printf("(%i %i)\n",label, tree.compactLeaves[traj].label);
			if (label == tree.compactLeaves[traj].label) 
			{				
				numCorrect++;	
				correct[label]++;
			}
			total[label]++;
			confusion[tree.compactLeaves[traj].label + label*numClasses]++; //Standard format from wiki: Each row is the true label, each column along a row is the prediction 
		}
	}

	std::ofstream ofsLbl((treeName+".labels").c_str(), std::ios::binary);
	ofsLbl.write((char*)tmpLabels, numSamples*sizeof(ushort));
	ofsLbl.close();

	FILE_LOG(LOG1) << "Tree self-success:";
	FILE_LOG(LOG1) << "------ Per class -------";

	for (int i=0; i<numClasses; i++)
		FILE_LOG(LOG1) << std::setprecision(2) << std::setw(3) << i << "(" << std::setw(6) << std::fixed << 100.0f*(float)total[i]/(float)numSamples << "%):" << std::fixed << 100.0f*(float)correct[i]/(float)total[i];

	FILE_LOG(LOG3) << "----Confusion data------";
	
	std::stringstream ss;	
	ss.str("");
	ss << std::setw(9) << "Tru\\Pred";
	for (int i=0;i<numClasses; i++) 
		ss << std::setw(10) << i;
	FILE_LOG(LOG3) << ss.str().c_str();

	for (int i=0; i<numClasses; i++)
	{
		ss.str("");
		ss << std::setw(8) << i << ":";
		for (int j=0; j<numClasses; j++)
			ss << std::setw(10) << confusion[j + i*numClasses];
		FILE_LOG(LOG3) << ss.str().c_str();
	}


	FILE_LOG(LOG1) << "-------Total -------";
	FILE_LOG(LOG0) << "Depth "<< depth << ": "<< std::fixed << 100.0f*(float)numCorrect/(float)numSamples <<"%";

	std::ofstream ofs(treeName.c_str(), std::ios::binary);
	ofs.write((char*)tree.treeAlloc, allocSize);
	ofs.close();
	delete[] correct;
	delete[] total;
	delete[] fullHistograms;
	delete[] tree.treeAlloc;
	FILE_LOG(LOG1) << "Full tree snapshot and self success timing in " << std::fixed << clk.toc() << "s";
}

void pushMapTasksData(int deviceId)
{
	//Send the full set of labels, nodetraj and other info to the device.
	// The following calls are blocking on this thread
	gpuDevice[deviceId].pushSamples( &sampleList[0]);
	gpuDevice[deviceId].pushNodes( &nodes[0], 0, maxNodes);

	// Now notify the waiting thread if the correct number of calls have been made
	boost::mutex::scoped_lock lock(broadcastMutex);
	broadcastCount++;
	if (broadcastCount == numDevices)
		broadcastConditionVar.notify_one();
}

void broadcastMapTasksData()
{
	Clock clk;
	clk.tic();
	//We want to start out with no broadcasts
	broadcastCount = 0;

	// Broadcast all the data onto the different devices. 
	for (int deviceId = 0; deviceId < numDevices; deviceId++)
		boost::thread task(&pushMapTasksData, deviceId);
		//pushMapTasksData(deviceId);
		
	//And now we want to check how many broadcasts occurred. If the right number occured then we have finished
	boost::mutex::scoped_lock lock(broadcastMutex);
	while (broadcastCount < numDevices)
		broadcastConditionVar.wait(lock);

	FILE_LOG(LOG1) << "Broadcast map task data: "<< clk.toc() << "s";
}

void reduceMapTasksData(int depth)
{
	Clock clk;
	clk.tic();
	// depth starts from 0 
	int numNodes = pow(2.0f, depth); // This is the number of the nodes at a particular depth level
	int idxDepth = numNodes - 1; //this is the offset index of the first node on the depth level

	//create enough memory for all the devices to pull for this depth level and the next
	RFTrainNode *gpuNodes;
	cudaHostAlloc(&gpuNodes, numDevices * (1 /*This level*/ + 2 /*next */) * numNodes *sizeof(RFTrainNode),cudaHostAllocPortable); 

	//We now pull all the current device trees. For an MPI implementation this could be done
	//one node at a time instead of storing the whole set in memory but here we really dont care
	// because its less than 300mb for 4 devices at 2m nodes
	for (int deviceId = 0; deviceId < numDevices; deviceId++)
		gpuDevice[deviceId].pullNodes(gpuNodes + deviceId*(1 + 2)*numNodes, idxDepth, (2*(int)(depth<maxDepth) + 1)*numNodes);

	//NOTE: In above, the gpu and cpu stores the entire tree of nodes in memory 
	//However here we only pull one or two levels and then update that level in the cpu node array

	for (int node = 0; node < numNodes; node++ )
	{
		//Find the device with the best result for this node
		int bestDeviceId = 0;
		for (int deviceId = 0; deviceId < numDevices; deviceId++)
		{
			if (gpuNodes[node + deviceId*(2 + 1)*numNodes].bestDelta > gpuNodes[node + bestDeviceId*(2 + 1)*numNodes].bestDelta)
				bestDeviceId = deviceId;

			FILE_LOG(LOG3) << "Node: "<<node+idxDepth<<" Best delta:"<< gpuNodes[node + deviceId*(2 + 1)*numNodes].bestDelta <<"\n";
		}

		//Set the global node to be the best node from all the devices for this node index
		//and also update the entropies of the next node level
		nodes[node + idxDepth] = gpuNodes[node + bestDeviceId*(2 + 1)*numNodes];

		//getting the corresponding best set of samples relating to that node
		gpuDevice[bestDeviceId].pullSamples( &sampleList[0], nodes[node + idxDepth].idxBegin, nodes[node + idxDepth].idxEnd);

		//now reorder the index based on these samples for this node
		int pIdx = partition(&sampleList[0], &revIndexList[0], nodes[node + idxDepth].idxBegin, nodes[node + idxDepth].idxEnd);
		
		pIdx = iSnapDown(pIdx,sizeof(SampleType4)); 
		
		//__PROBLEM__^__. This will pose a problem for deeper nodes and leaves.
		// Im not exactly sure why I used it. I think by not doing so it caused some kind of alignment bug.
		// However the problem it generates is as such: Suppose we have a node that looks like this after sorting 000111. pIdx will then start as 3.
		// The above line however will turn it into pIdx = 0 which will make the node think it has 0 elements on the left and 6 elements on the right. 
		// Im not sure yet how much of an effect this has. But it does mean that when there are very few examples in a node some will go the wrong way which
		// means they will be incorrectly classified at runtime...

		if (depth < maxDepth)
		{
			// and set the start and end indices of the next levels node
			// What happens when one of these has no occupants? Then the children receive the same boundaries as the parents						
			nodes[(node + idxDepth)*2 + 1].idxBegin = nodes[node + idxDepth].idxBegin;
			nodes[(node + idxDepth)*2 + 1].idxEnd = pIdx - 1;
			nodes[(node + idxDepth)*2 + 1].entropy = gpuNodes[numNodes + (node*2) + bestDeviceId*(2 + 1)*numNodes].entropy;

			nodes[(node + idxDepth)*2 + 2].idxBegin = pIdx;
			nodes[(node + idxDepth)*2 + 2].idxEnd = nodes[node + idxDepth].idxEnd;
			nodes[(node + idxDepth)*2 + 2].entropy = gpuNodes[numNodes + (node*2 + 1) + bestDeviceId*(2 + 1)*numNodes].entropy;
		}
	}
	FILE_LOG(LOG1) << "Reduced map task data in " << std::fixed << clk.toc() << "s";
	clk.tic();

	// Here we update all the indicies and samples in preparation for next level
#pragma omp parallel for
	for (int sampleId = 0; sampleId < numSamples; sampleId++)
	{
		sampleList[sampleId].processed = 0;

		//The +1 below ensures that the trajectory represents the actual node number
		//so .traj is not relative to the current level and instead represents the true node

		//sampleList[sampleId].traj = sampleList[sampleId].traj*2 + sampleList[sampleId].bestTraj + 1;
		fwdIndexList[ revIndexList[ sampleId ] ] = sampleId;
	}
	FILE_LOG(LOG1) << "Updated index arrays in " << std::fixed << clk.toc() << "s";

	cudaFreeHost(gpuNodes);
}

template<class Ftype>
void launchMapTask(int depth, int workerId, int featureToken)
{
	Clock clk;
	clk.tic();
	//Get the gpu worker. One task is run per one CPU thread. 
	GPUWorker<Ftype> *gpuWorker = ((GPUWorkerPool<Ftype>*)gpuWorkerPool)->getWorker(workerId);		

	//Get the cpu based feature data
	//const uint * index = featurePool->getFeature(featureToken)->getIndex();
	const Ftype* data_host = (Ftype*)((FeaturePool<Ftype>*)featurePool)->getFeature(featureToken)->getDataPtr();
	int fId = ((FeaturePool<Ftype>*)featurePool)->getFeature(featureToken)->getId();	

	gpuWorker->setWorkerHeading(workerId,fId);

	//Perform a synchronous copy to the GPU on the worker's stream. Synchronous is meant to illustrate 
	// that this CPU thread wont continue until after the copy is finished.
	gpuWorker->sendData(data_host, fwdIndexList);

	//CPU memory has been copied to GPU so we can release it on the CPU
	((FeaturePool<Ftype>*)featurePool)->releaseFeatureToken(featureToken);	

	//Lock this particular worker's device and compute whatever we need to compute
	// Uses whatever is currently in the workers memory slot to perform a map from data to <node,score> pairs
	gpuWorker->map(depth, fId, numThresh, threshList, &nodes[0]);	

	((GPUWorkerPool<Ftype>*)gpuWorkerPool)->releaseGPUWorker(workerId);

	FILE_LOG(LOG1) << "Worker: " << workerId << ", Feature: " << fId<< " completed in " << std::fixed << clk.toc() << "s";
}


template <class Ftype>
int runBuilder()
{

	char tn[10];
	sprintf(tn, "%.4i", treeNum);
	std::string treeName(tn);

	FILE_LOG(LOG1) << "Loading data ...";
	cudaError_t err;

	// ------------------ Initially pure linear index
	err = cudaHostAlloc(&fwdIndexList, numSamples*sizeof(uint),cudaHostAllocPortable);	
	err = cudaHostAlloc(&revIndexList, numSamples*sizeof(uint),cudaHostAllocPortable);	

    if (err!=cudaSuccess)
    	FILE_LOG(LOG0) << "Error allocating memory: " << errCESTRing(err) << ".\n\n\n";

	for (int idx = 0; idx < numSamples; idx++)
	{
		fwdIndexList[idx] = idx;
		revIndexList[idx] = idx;
	}

	// ------------------Read in the label list and load up the sample list 
	ushort* labelList;
	labelList =		0;
	sampleList =	0;

	cudaHostAlloc(&labelList, numSamples*sizeof(ushort),cudaHostAllocPortable);
	cudaHostAlloc(&sampleList, numSamples*sizeof(Sample),cudaHostAllocPortable);

	readList<ushort>(labelList, numSamples, (baseDir + "Labels.lbl"));
	numClasses = *std::max_element(labelList, labelList + numSamples) + 1; // Classes start from 0 
	FILE_LOG(LOG0) << "Number of classes detected in Labels.lbl : " << numClasses;

	for (int i = 0; i < numSamples; i++)
	{
		sampleList[i].bestTraj =	0;
		sampleList[i].processed =	0;
		sampleList[i].label =		labelList[i];		
	}

	cudaFreeHost(labelList);

	// ------------------Read in the threshold list
	threshList.resize(numFeatures*3);
	readList<float>(&threshList[0], numFeatures*3, (baseDir + "Threshholds.thr"));

	// ------------------Setup the tree nodes
	nodes.resize(maxNodes);
	for (int i = 0 ; i < maxNodes; i++) 
	{
		nodes[i].idxBegin =		 0;
		nodes[i].idxEnd =		 numSamples - 1;
		nodes[i].entropy =		 0;		// Started as 0. This will be the initial entropy of the first node despite it not being correct. This is because we advance according to the DIFFERENCE in entropies and the first node entropy is the same across all nodes 
		nodes[i].bestDelta =	-1e20f;	// Best delta is as low as possible to start with
		nodes[i].bestFeat =		9999;		// Initiated with invalid index
		nodes[i].bestThresh =	 0;		// Difficult to define what an invalid threshold is so we leave it at 0
	}


	// ------------------Setup the devices and allocate memory
	// Setup the worker pool
	gpuDevice.resize(numDevices);			
	for (int deviceId = 0; deviceId < numDevices; deviceId++)
	{
		gpuDevice[deviceId].init(deviceId + startDevice, numSamples, maxNodes, maxDepth, numClasses, foldingDepth);
		gpuDevice[deviceId].pushSamples( &sampleList[0]);
		gpuDevice[deviceId].setWeights(weightType, (baseDir + "Weights.txt"));
	}


	// ------------------Setup the workers
	gpuWorkerPool =	new GPUWorkerPool<Ftype>(&gpuDevice, numStreams); 


	// ------------------Setup and start the featurepool manager
	featurePool	=	new FeaturePool<Ftype>(numFeatures, poolSize, (baseDir + "F_"),  numSamples);
	//__PROBLEM__: should make the base i.e. F_ be a user input value

	// ------------------Read score weights from file
	scoreWeights.resize(numClasses);
	for (int i = 0 ; i < scoreWeights.size(); i++)
		scoreWeights[i] = 1;
	//readPairs(scoreWeights, (baseDir + "WeightsScore.txt"));
	//__PROBLEM__: score weights should be returned? They are there to weight the errors


	Clock totalclk,tmpclock;
	totalclk.tic();
	// ------------------Start running the main loop

	FILE_LOG(LOG0) << "Generating Tree: " << treeNum;
	for (int d = 0; d <= maxDepth; d++) 
	{
		FILE_LOG(LOG0) << "Generating Tree level: " << d;

		//This copies nodes and sampleList (i.e. node traj and labels) to all the devices (no index mapping)
		broadcastMapTasksData(); 

		int n=0;
		Clock clk;
		clk.tic();
		while (1) 
		{
			int featureToken = ((FeaturePool<Ftype>*)featurePool)->acquireFeatureToken(); 

			if (featureToken<0) // then the current set of tasks has been fully dispatched. The feature loader will reset after the previous line
				break;

			int workerId = ((GPUWorkerPool<Ftype>*)gpuWorkerPool)->acquireGPUWorker();

			boost::thread task(&launchMapTask<Ftype>,d,workerId,featureToken);
			//launchMapTask<Ftype>(d,workerId,featureToken);

			n++;
		}

		((GPUWorkerPool<Ftype>*)gpuWorkerPool)->waitForGPUWorkers();

		//We now take all the devices which have held the intermediate reductions
		//and reduce them all onto the CPU and update the tree 
		reduceMapTasksData(d);

		// Extract histograms for this level
		//saveCurrentTree(); including coords, nodes and extracted histograms from samples;

		generateTreeFromBuildState(d,(baseDir + "Tree_" + treeName + ".tree"));
		//generateSampleTrajectoriesFromBuildState(d, (baseDir + "Leaves_" + treeName + ".tlbl"));
		
		FILE_LOG(LOG0) << "Processed " << n << " tasks in " << std::fixed << clk.toc() << "s\n\n";
	}

	FILE_LOG(LOG0) << "Processed Tree "<< treeNum << " in " << std::fixed << totalclk.toc() << "s";

	cudaFreeHost(fwdIndexList);
	cudaFreeHost(revIndexList);
	cudaFreeHost(sampleList);
	delete (FeaturePool<Ftype>*)featurePool;
	delete (GPUWorkerPool<Ftype>*)gpuWorkerPool;
	FILE_LOG(LOG0) << "Tree construction complete.\n\n\n";

	//getchar();

	return 0;
} 

#define stringify( name ) # name

const char* featureTypeName[]=
{
	stringify( F_CHAR ),
	stringify( F_SHORT ),
	stringify( F_INT ),
	stringify( F_FLOAT )
};

const char* weightTypeName[]=
{
	stringify( W_ONES ),
	stringify( W_APRIORI ),
	stringify( W_FILE )
};

int main (int argc, char ** argv) 
{
	// ------------------Set up the parameters. -----------------
	//printf("%i\n",sizeof(Sample));getchar();return 0;
	FILELog::ReportingLevel() = FILELog::FromString(argc>=15 ? argv[14] : "LOG1");

	if (argc<14)
	{
		FILE_LOG(logINFO) << " TreeBuilder " << VERSION << ". Trees built with this version are not compatible with earlier versions.";
		FILE_LOG(logINFO) << " They do not include coords so they are cleaner and require less space but they need a new reader.";
		FILE_LOG(logINFO) << argc-1 << " parameters supplied. ";
		FILE_LOG(logINFO) << "Usage:\n";
		FILE_LOG(logINFO) << "1. numFeatures  : How many features to use";
		FILE_LOG(logINFO) << "2. poolSize     : Size of the preloaded pool of features";
		FILE_LOG(logINFO) << "3. maxDepth     : Tree depth the be built to";
		FILE_LOG(logINFO) << "4. foldingDepth : Depth at which to fold the tree when sending it to the cards";	
		FILE_LOG(logINFO) << "5. featureType  : F_CHAR, F_SHORT, F_INT, F_FLOAT";
		FILE_LOG(logINFO) << "6. numSamples   : Number of samples to use from feature files";
		FILE_LOG(logINFO) << "7. numThresh    : Number of thresholds to test";
		FILE_LOG(logINFO) << "8. startDevice  : Start device in device range";
		FILE_LOG(logINFO) << "9. endDevice    : End device in device range";
		FILE_LOG(logINFO) << "10. weightType  : W_ONES, W_APRIORI, W_FILE";
		FILE_LOG(logINFO) << "11. baseDir     : Directory where all files exist and will be built";
		FILE_LOG(logINFO) << "12. treePrefix  : The prefix to be prepended to a tree";
		FILE_LOG(logINFO) << "13. treeNum     : The treenum to be appended to a tree";
		FILE_LOG(logINFO) << "14. Log type    : LOG0, LOG1, LOG2, LOG3, LOG4";
		FILE_LOG(logINFO) << "15. Comment     : A comment enclosed by inverted commas";

		return 0;
	}
	else
	{
		FILE_LOG(logINFO) << "Using cmdline parameters:";		
		numFeatures =	atoi(argv[1]);
		poolSize =		std::min(numFeatures,atoi(argv[2]));
		maxDepth =		atoi(argv[3]);
		foldingDepth =  atoi(argv[4]);
		maxNodes =		pow(2.0f, maxDepth + 1) - 1; 

		if (std::string(argv[5]) == "F_CHAR")	featureType =	F_CHAR; else
			if (std::string(argv[5]) == "F_SHORT")	featureType =	F_SHORT; else
				if (std::string(argv[5]) == "F_INT")	featureType =	F_INT; else
					if (std::string(argv[5]) == "F_FLOAT")	featureType =	F_FLOAT;

		numSamples =	atoi(argv[6]);
		numThresh =		atoi(argv[7]);		
		startDevice =	atoi(argv[8]);
		endDevice =		atoi(argv[9]);

		numDevices =	endDevice - startDevice + 1;

		if (std::string(argv[10]) == "W_ONES")	weightType = W_ONES; else
			if (std::string(argv[10]) == "W_APRIORI")	weightType = W_APRIORI; else
				if (std::string(argv[10]) == "W_FILE")	weightType = W_FILE; 
		
		numStreams =	1; //There seems to be no real advantage of using 2 streams and it seems to have a bug anyway. Could probably join gpuworker and gpudevice

		baseDir		= std::string(argv[11]);
		treePrefix	= std::string(argv[12]); 
		treeNum  	= atoi(argv[13]); 
	}


	char tn[100];
	sprintf(tn, "%s_%.4i", treePrefix.c_str(), treeNum);
	std::string logFile = baseDir + "LogFile_" + std::string(tn) + ".log"; 
	FILE* pFile = fopen(logFile.c_str(),"a");
	Output2FILE::Stream() = pFile;

	for (int i = 0; i < argc; i++)
		FILE_LOG(logINFO) << "[" << std::setw(2) << i << "] : " << argv[i];

	FILE_LOG(logINFO) << "---------------------";
	FILE_LOG(logINFO) << "numFeatures : " << numFeatures;
	FILE_LOG(logINFO) << "poolSize    : " << poolSize;
	FILE_LOG(logINFO) << "maxDepth    : " << maxDepth;
	FILE_LOG(logINFO) << "foldingDepth: " << foldingDepth;
	FILE_LOG(logINFO) << "maxNodes    : " << maxNodes;
	FILE_LOG(logINFO) << "featureType : " << featureTypeName[featureType];
	FILE_LOG(logINFO) << "numSamples  : " << numSamples;
	FILE_LOG(logINFO) << "numThresh   : " << numThresh;
	FILE_LOG(logINFO) << "firstDevice : " << startDevice;
	FILE_LOG(logINFO) << "lastDevice  : " << endDevice;
	FILE_LOG(logINFO) << "numDevices  : " << numDevices;
	FILE_LOG(logINFO) << "weightType  : " << weightTypeName[weightType];
	FILE_LOG(logINFO) << "baseDir     : " << baseDir.c_str();
	FILE_LOG(logINFO) << "logFile     : " << logFile.c_str();
	FILE_LOG(logINFO) << "treePrefix  : " << treePrefix.c_str();
	FILE_LOG(logINFO) << "treeNum     : " << treeNum;
	FILE_LOG(logINFO) << "---------------------";

	switch (featureType)
	{
	case F_CHAR:	runBuilder<char>(); break;
	case F_SHORT:	runBuilder<short>(); break;
	case F_INT:		runBuilder<int>(); break;
	case F_FLOAT:	runBuilder<float>(); break;
	}

	return 0;
}