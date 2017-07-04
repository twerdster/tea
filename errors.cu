#include "kernels.h"
#include <cuda.h>
//#include <cuda_runtime.h>

#define CESTR(code) case code: return #code

const char * errCESTRing(int errCode)
{
	switch (errCode)
	{
		CESTR(cudaSuccess);
		CESTR(cudaErrorMissingConfiguration);
		CESTR(cudaErrorMemoryAllocation);
		CESTR(cudaErrorInitializationError);
		CESTR(cudaErrorLaunchFailure);
		CESTR(cudaErrorPriorLaunchFailure);
		CESTR(cudaErrorLaunchTimeout);
		CESTR(cudaErrorLaunchOutOfResources);
		CESTR(cudaErrorInvalidDeviceFunction);
		CESTR(cudaErrorInvalidConfiguration);
		CESTR(cudaErrorInvalidDevice);
		CESTR(cudaErrorInvalidValue);
		CESTR(cudaErrorInvalidPitchValue);
		CESTR(cudaErrorInvalidSymbol);
		CESTR(cudaErrorMapBufferObjectFailed);
		CESTR(cudaErrorUnmapBufferObjectFailed);
		CESTR(cudaErrorInvalidHostPointer);
		CESTR(cudaErrorInvalidDevicePointer);
		CESTR(cudaErrorInvalidTexture);
		CESTR(cudaErrorInvalidTextureBinding);
		CESTR(cudaErrorInvalidChannelDescriptor);
		CESTR(cudaErrorInvalidMemcpyDirection);
		CESTR(cudaErrorAddressOfConstant);
		CESTR(cudaErrorTextureFetchFailed);
		CESTR(cudaErrorTextureNotBound);
		CESTR(cudaErrorSynchronizationError);
		CESTR(cudaErrorInvalidFilterSetting);
		CESTR(cudaErrorInvalidNormSetting);
		CESTR(cudaErrorMixedDeviceExecution);
		CESTR(cudaErrorCudartUnloading);
		CESTR(cudaErrorUnknown);
		CESTR(cudaErrorNotYetImplemented);
		CESTR(cudaErrorMemoryValueTooLarge);
		CESTR(cudaErrorInvalidResourceHandle);
		CESTR(cudaErrorNotReady);
		CESTR(cudaErrorInsufficientDriver);
		CESTR(cudaErrorSetOnActiveProcess);
		CESTR(cudaErrorInvalidSurface);
		CESTR(cudaErrorNoDevice);
		CESTR(cudaErrorECCUncorrectable);
		CESTR(cudaErrorSharedObjectSymbolNotFound);
		CESTR(cudaErrorSharedObjectInitFailed);
		CESTR(cudaErrorUnsupportedLimit);
		CESTR(cudaErrorDuplicateVariableName);
		CESTR(cudaErrorDuplicateTextureName);
		CESTR(cudaErrorDuplicateSurfaceName);
		CESTR(cudaErrorDevicesUnavailable);
		CESTR(cudaErrorInvalidKernelImage);
		CESTR(cudaErrorNoKernelImageForDevice);
		CESTR(cudaErrorIncompatibleDriverContext);
		CESTR(cudaErrorPeerAccessAlreadyEnabled);
		CESTR(cudaErrorPeerAccessNotEnabled);
		CESTR(cudaErrorDeviceAlreadyInUse);
		CESTR(cudaErrorProfilerDisabled);
		CESTR(cudaErrorProfilerNotInitialized);
		CESTR(cudaErrorProfilerAlreadyStarted);
		CESTR(cudaErrorProfilerAlreadyStopped);
		CESTR(cudaErrorAssert);
		CESTR(cudaErrorTooManyPeers);
		CESTR(cudaErrorHostMemoryAlreadyRegistered);
		CESTR(cudaErrorHostMemoryNotRegistered);
	default:
		return "Unknown error code";
	}
}

