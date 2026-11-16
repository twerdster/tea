

#include <assert.h>
#include <condition_variable>
#include <iomanip>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <limits>
#include <stdexcept>
#include "AtomicQueue.h"
#include "Feature.h"
#include "GPUWorker.h"
#include "FeaturePool.h"
#include "GPUWorkerPool.h"
#include "DoubleSort.h"
#include "utils.h"
#include "Clock.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "log.h"
#include "errors.h"

std::mutex broadcastMutex;
std::condition_variable broadcastConditionVar;
int broadcastCount;

//I chose against abstracting the classes to base classes with pure virtual coz im lazy but everything needs to be properly cleaned up eventually
void* featurePool;
void* gpuWorkerPool;

std::vector<GPUDevice> gpuDevice;
std::vector<RFTrainNode> nodes;

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

// Publish a complete snapshot with one same-directory rename. A monitor or
// evaluator can open the current file while the next depth is being trained.
void writeAtomicBinary(const std::string& path, const void* data, size_t bytes)
{
	const std::string temporary = path + ".tmp";
	std::ofstream output(temporary.c_str(), std::ios::binary | std::ios::trunc);
	output.write(static_cast<const char*>(data), bytes);
	output.close();
	if (!output || std::rename(temporary.c_str(), path.c_str()) != 0)
		throw std::runtime_error("Failed publishing snapshot: " + path);
}

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
	size_t allocSize = sizeof(TreeHeader) + sizeof(RFNode)*totalNodes + sizeof(CompactLeaf)*totalHists + sizeof(uchar)*totalHists*numClasses;
	tree.treeAlloc = new char[allocSize](); // deterministic native struct padding
	tree.th =				(TreeHeader*)	(tree.treeAlloc);
	tree.nodes =			(RFNode*)		((char*)tree.th				+ sizeof(TreeHeader));
	tree.compactLeaves =	(CompactLeaf*)	((char*)tree.nodes			+ sizeof(RFNode)*totalNodes);
	tree.histograms =		(uchar*)		((char*)tree.compactLeaves	+ sizeof(CompactLeaf)*totalHists);
	uint * fullHistograms = new uint[size_t(totalHists)*numClasses];
	memset(fullHistograms,0,sizeof(uint)*size_t(totalHists)*numClasses);

	for (int nIdx = totalNodes/2; nIdx < totalNodes;  nIdx++)
	{
		for (int s = nodes[nIdx].idxBegin; s <= nodes[nIdx].idxEnd; s++)
		{
			uint traj =  nIdx*2 + sampleList[s].bestTraj + 1;
			uint label = sampleList[s].label;
			//printf("Node, Traj, Label : %i,%i,%i\n",nIdx,traj,label);
			// Heap indices are zero based: parent(i)=(i-1)/2, not i>>1.
			for (uint hist = traj; ; hist = (hist - 1) / 2) {
				fullHistograms[size_t(hist)*numClasses + label]++;
				if (hist == 0) break;
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
		// Empty leaves inherit the nearest populated ancestor's posterior.
		bool occupied = false;
		for (int c = 0; c < numClasses; ++c)
			occupied = occupied || fullHistograms[size_t(n)*numClasses + c] != 0;
		if (!occupied && n > 0)
			std::copy(fullHistograms + size_t((n-1)/2)*numClasses,
			          fullHistograms + size_t((n-1)/2 + 1)*numClasses,
			          fullHistograms + size_t(n)*numClasses);
		// Counts can exceed float's exact integer range on large datasets.
        // Keep the raw argmax exact; ties select the lowest class index.
        uint maximum = 0;
        int label = 0;
        for (int c = 0; c < numClasses; ++c) {
            const uint count = fullHistograms[size_t(n)*numClasses + c];
            if (count > maximum) { maximum = count; label = c; }
        }
        CompactLeaf &cl = tree.compactLeaves[n];
        cl.maxVal = maximum;
        cl.label = label;
	}

	for (int n = 0; n < tree.th->totalHists; n++)
	{
		CompactLeaf &cl = tree.compactLeaves[n];
		for (int c = 0 ; c < tree.th->numClasses; c++)
		{
			// Max scaling preserves the argmax; inference normalizes the sum.
			uchar val = cl.maxVal ? static_cast<uchar>(std::min(255.0,
				double(fullHistograms[size_t(n)*tree.th->numClasses + c])*255.0/double(cl.maxVal))) : 0;
			tree.histograms[size_t(n)*tree.th->numClasses + c] = val;
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

	writeAtomicBinary(treeName + ".labels", tmpLabels, size_t(numSamples)*sizeof(ushort));

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

	char depthSuffix[32];
	std::snprintf(depthSuffix, sizeof(depthSuffix), ".depth%02d.tree", depth);
	std::string depthName = treeName.substr(0, treeName.size()-5) + depthSuffix;
	if (std::ifstream(depthName.c_str()).good())
		throw std::runtime_error("Refusing to overwrite completed depth checkpoint: " + depthName);
	writeAtomicBinary(depthName, tree.treeAlloc, allocSize);
	writeAtomicBinary(treeName, tree.treeAlloc, allocSize);
	delete[] correct;
	delete[] total;
	delete[] fullHistograms;
	delete[] tree.treeAlloc;
	delete[] confusion;
	delete[] tmpLabels;
	FILE_LOG(LOG1) << "Full tree snapshot and self success timing in " << std::fixed << clk.toc() << "s";
}

void pushMapTasksData(int deviceId)
{
	//Send the full set of labels, nodetraj and other info to the device.
	// The following calls are blocking on this thread
	gpuDevice[deviceId].pushSamples( &sampleList[0]);
	gpuDevice[deviceId].pushNodes( &nodes[0], 0, maxNodes);

	// Now notify the waiting thread if the correct number of calls have been made
	std::lock_guard<std::mutex> lock(broadcastMutex);
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
	std::vector<std::thread> tasks;
	tasks.reserve(numDevices);
	for (int deviceId = 0; deviceId < numDevices; deviceId++)
		tasks.emplace_back(&pushMapTasksData, deviceId);
		//pushMapTasksData(deviceId);

	//And now we want to check how many broadcasts occurred. If the right number occured then we have finished
	std::unique_lock<std::mutex> lock(broadcastMutex);
	while (broadcastCount < numDevices)
		broadcastConditionVar.wait(lock);
	lock.unlock();

	for (std::thread &task : tasks)
		task.join();

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
	teaCudaCheck(cudaHostAlloc(&gpuNodes, size_t(numDevices) * 3 * numNodes *sizeof(RFTrainNode),cudaHostAllocPortable), "reduction host allocation");

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

		// Keep exact boundaries, including odd and singleton children.

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

	char tn[32];
	sprintf(tn, "%.4i", treeNum);
	std::string treeName(tn);

	FILE_LOG(LOG1) << "Loading data ...";


	// ------------------ Initially pure linear index
		teaCudaCheck(cudaHostAlloc(&fwdIndexList, size_t(numSamples)*sizeof(uint),cudaHostAllocPortable), "forward index allocation");
		teaCudaCheck(cudaHostAlloc(&revIndexList, size_t(numSamples)*sizeof(uint),cudaHostAllocPortable), "reverse index allocation");


	for (int idx = 0; idx < numSamples; idx++)
	{
		fwdIndexList[idx] = idx;
		revIndexList[idx] = idx;
	}

	// ------------------Read in the label list and load up the sample list
	ushort* labelList;
	labelList =		0;
	sampleList =	0;

	teaCudaCheck(cudaHostAlloc(&labelList, size_t(numSamples)*sizeof(ushort),cudaHostAllocPortable), "labels allocation");
	teaCudaCheck(cudaHostAlloc(&sampleList, size_t(numSamples)*sizeof(Sample),cudaHostAllocPortable), "sample host allocation");

	readList<ushort>(labelList, numSamples, (baseDir + "Labels.lbl"));
	numClasses = *std::max_element(labelList, labelList + numSamples) + 1; // Classes start from 0
	if (numClasses > (1 << 14)) throw std::invalid_argument("Labels exceed 14-bit Sample label storage");
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
		nodes[i].bestFeat =		0;		// Valid deterministic split for empty nodes
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
		std::vector<std::thread> tasks;
		while (1)
		{
			int featureToken = ((FeaturePool<Ftype>*)featurePool)->acquireFeatureToken();

			if (featureToken<0) // then the current set of tasks has been fully dispatched. The feature loader will reset after the previous line
				break;

			int workerId = ((GPUWorkerPool<Ftype>*)gpuWorkerPool)->acquireGPUWorker();

			tasks.emplace_back(&launchMapTask<Ftype>,d,workerId,featureToken);
			//launchMapTask<Ftype>(d,workerId,featureToken);

			n++;
		}

		((GPUWorkerPool<Ftype>*)gpuWorkerPool)->waitForGPUWorkers();
		for (std::thread &task : tasks)
			task.join();

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
		if (numFeatures <= 0 || poolSize <= 0 || maxDepth < 0 || maxDepth > 20 ||
		    foldingDepth < 0 || foldingDepth > maxDepth)
			throw std::invalid_argument("Require features/pool>0 and 0<=foldingDepth<=maxDepth<=20; CLI depth d means d+1 decisions");
		maxNodes = (1u << (maxDepth + 1)) - 1;

		if (std::string(argv[5]) == "F_CHAR")	featureType =	F_CHAR; else
			if (std::string(argv[5]) == "F_SHORT")	featureType =	F_SHORT; else
				if (std::string(argv[5]) == "F_INT")	featureType =	F_INT; else
					if (std::string(argv[5]) == "F_FLOAT")	featureType =	F_FLOAT;
					else throw std::invalid_argument("Unknown feature type");

		numSamples =	atoi(argv[6]);
		numThresh =		atoi(argv[7]);
		startDevice =	atoi(argv[8]);
		endDevice =		atoi(argv[9]);
		if (numSamples <= 0 || numThresh < 1 || startDevice < 0 || endDevice < startDevice)
			throw std::invalid_argument("Invalid sample, threshold or GPU range");

		numDevices =	endDevice - startDevice + 1;

		if (std::string(argv[10]) == "W_ONES")	weightType = W_ONES; else
			if (std::string(argv[10]) == "W_APRIORI")	weightType = W_APRIORI; else
				if (std::string(argv[10]) == "W_FILE")	weightType = W_FILE;
				else throw std::invalid_argument("Unknown weight type");

		numStreams =	1; //There seems to be no real advantage of using 2 streams and it seems to have a bug anyway. Could probably join gpuworker and gpudevice

		baseDir		= std::string(argv[11]);
		if (baseDir.empty()) throw std::invalid_argument("Dataset directory is empty");
		if (baseDir.back() != '/') baseDir += '/';
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
