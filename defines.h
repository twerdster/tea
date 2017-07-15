
#ifndef DEFINES
#define DEFINES

#define VERSION "0.9"

#define BLOCKSIZE_64  ( 64 * 1024 * 1024 )
#define BLOCKSIZE_8  ( 8 * 1024 * 1024 )
#define BLOCKSIZE_1   ( 1 * 1024 * 1024 )

#define MAXMIN(x,lower,upper) std::ceil(std::max(std::min((float)(x),(float)(upper)),(float)(lower)))

#define LABEL(s)	( 0x3FFF & (int)((s).label) ) 

#define CHILD(f,t)	( (int) (( float)((f)) > ( float)((t))) ) // 0 is left child, 1 is right child

#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

#define DEBUG_LEVEL 40 
#define DEBUG(level, ...) {if (level<=DEBUG_LEVEL) printf(__VA_ARGS__);}

typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned char ntype;

///#define LABEL(s)	( 0x3F & (int)((s).label) ) 
//typedef uchar SampleType;    
//typedef uchar4 SampleType4; 

// Currently to change the program to work efficiently (half the memory) with 64 classes then uncomment the above lines and remove the other label(s) line

typedef ushort SampleType;
typedef ushort4 SampleType4;// should this be short or char?

#define HISTOGRAM64_BIN_COUNT				64
#define SHARED_MEMORY_BANKS					32
#define HISTOGRAM64_THREADBLOCK_SIZE		(4 * SHARED_MEMORY_BANKS)
#define MERGE_THREADBLOCK_SIZE				256
#define LOG2_WARP_SIZE						5U
#define WARP_SIZE							(1U << LOG2_WARP_SIZE )
#define HISTOGRAM256_BIN_COUNT				256
#define WARP_COUNT							6 //subhistograms per block ?
#define PARTIAL_HISTOGRAM256_COUNT			240
#define HISTOGRAM256_THREADBLOCK_SIZE		( WARP_COUNT * WARP_SIZE )
#define HISTOGRAM256_THREADBLOCK_MEMORY		( WARP_COUNT * HISTOGRAM256_BIN_COUNT )
#define MAX_PARTIAL_HISTOGRAM64_COUNT		(2*32768)

enum NodeType { CHILD_NODE, PARENT_NODE };
enum HistType { HIST_64, HIST_256, HIST_REGULAR};
enum FeatureType { F_CHAR=0, F_SHORT, F_INT, F_FLOAT};
enum WeightType { W_ONES=0, W_APRIORI, W_FILE};

// NOTE: we really need to add pointer type and other better defines so we can use sizeof and not have to get what is being used.
//       There should be no native types in the main program code. i.e. no char, uchar etc.

//Round a / b to nearest higher integer value
inline uint iDivUp(uint a, uint b){ return (a % b != 0) ? (a / b + 1) : (a / b); }

//Snap a to nearest lower multiple of b
inline uint iSnapDown(uint a, uint b){ return a - a % b; }

struct RFTrainNode 
{
	int idxBegin;			// Start index of the samples relating to this node 
	int idxEnd;				// End index of the samples relating to this node
	float entropy;			// entropy of this node
	float bestDelta;		// current best increase in entropy resulting from using bestFeat and bestThresh
	float bestThresh;		// current best threshold
	uint bestFeat : 31;		// current best feature
	uint didUpdate : 1;		// set if node updates 
}; // = 48 bytes -> Wont be properly aligned to any cache lines (128 bit)

struct RFNode
{
	int F;					// Index of best feature (not sure what this will be used for yet)
	float thr;
};

struct CompactLeaf
{
	uint maxVal;
	ushort label; // NOTE: This is now ushort. It will break compatibility with trees before this version.
};

struct TreeHeader
{
	int depth;
	int numClasses;
	int totalNodes;
	int totalHists;
};

struct Tree
{
	char * treeAlloc;  // NOTE: should probably be some nicer type which represents a byte  pointer. char is distracting.
	TreeHeader* th;
	RFNode* nodes;
	CompactLeaf* compactLeaves;
	uchar* histograms; // NOTE: should probably be char* or some nicer type which represents a byte  pointer. uchar is distracting.
};

struct ForrestHeader
{
	int numTrees;
	int treeAllocSize;
};

struct Sample
{
	SampleType label : (6+8); 
	SampleType bestTraj : 1;
	SampleType processed : 1;	
}; 


#endif //DEFINES