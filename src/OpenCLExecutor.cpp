#include "OpenCLExecutor.h"
#include <math.h>
#ifndef WIN32
#include <unistd.h>
#else
#include <windows.h>
#endif
#include <sstream>

OpenCLExecutor* OpenCLExecutor::internalExec = NULL;
#ifdef WIN32
  MUTEXTYPE OpenCLExecutor::CL_LOCK;
#else
  MUTEXTYPE OpenCLExecutor::CL_LOCK;
#endif

  void OpenCLExecutor::resolveLockThreadExec()
  {
#ifndef WIN32
	struct timespec time;
	time.tv_sec = 0;
	time.tv_nsec = 500000000L;
#endif

	  while (true)
	  {
#ifdef WIN32
		Sleep(500);
#else
		nanosleep(time);
#endif
		resolveLocks();
	  }
  }

  OpenCLExecutor::OpenCLExecutor()
{
	  bIsInitialized = false;
}

OpenCLExecutor::~OpenCLExecutor()
{
	ACQUIRE_MUTEX(CL_LOCK);

	/*for (int i = 0; i < workingGroups.size(); i++)
	{
		workingGroups[i]->CleanUpDevice();
	}*/

	RELEASE_MUTEX(CL_LOCK);
	DESTROYMUTEX(CL_LOCK);
}

void OpenCLExecutor::resolveLocks()
{
	throw OCLException("Undefined method 'resolveLocks' called!");
}

bool OpenCLExecutor::InitPlatform(int platformIdx, int deviceIdx)
{
	ACQUIRE_MUTEX(CL_LOCK);

	if (bIsInitialized)
	{
		RELEASE_MUTEX(CL_LOCK);
		return true;
	}

	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		throw OCLException(" No platforms found. Check OpenCL installation!\n");
		RELEASE_MUTEX(CL_LOCK);
		return false;
	}
	//std::printf("Found platforms: %zi\n", all_platforms.size());

	std::vector<cl::Device> all_devices;
	//Search for GPU first
	all_platforms[platformIdx].getDevices(CL_DEVICE_TYPE_GPU, &all_devices);

	device = all_devices[deviceIdx];
	platform = all_platforms[platformIdx];

	//Get all relevant infos
	deviceInfos = FOCLDeviceInfos(device);

	std::stringstream s2;
	s2 << platform.getInfo<CL_PLATFORM_NAME>();
	std::printf("Using platform: %s\n", s2.str().c_str());
	std::printf("Using device: %s | using %i cores at %i MHz \n", deviceInfos.deviceName.c_str(), device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(), device.getInfo <CL_DEVICE_MAX_CLOCK_FREQUENCY>());
	std::printf("OpenCL version: %s\n", deviceInfos.clVersion.c_str());

	internalExec->getContext();

	bIsInitialized = (internalExec->device.getInfo<CL_DEVICE_AVAILABLE>() == CL_TRUE);

	RELEASE_MUTEX(CL_LOCK);
	return bIsInitialized;
}

std::vector<std::string> OpenCLExecutor::GetDevices(int platformIdx)
{
	std::vector<std::string> devices;

	ACQUIRE_MUTEX(CL_LOCK);

	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	std::vector<cl::Device> all_devices;
	all_platforms[platformIdx].getDevices(CL_DEVICE_TYPE_GPU, &all_devices);

	for (int i = 0; i < all_devices.size(); i++)
	{
		std::stringstream s;
		s << all_devices[i].getInfo<CL_DEVICE_NAME>();
		devices.push_back(s.str());
	}
	RELEASE_MUTEX(CL_LOCK);

	return devices;
}

std::vector<std::string> OpenCLExecutor::GetPlatforms()
{
	std::vector<std::string> platforms;
	ACQUIRE_MUTEX(CL_LOCK);

	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		throw OCLException(" No platforms found. Check OpenCL installation!\n");
		RELEASE_MUTEX(CL_LOCK);
		return platforms;
	}

	for (int i = 0; i < all_platforms.size(); i++)
	{
		std::stringstream s;
		s << all_platforms[i].getInfo<CL_PLATFORM_NAME>();
		platforms.push_back(s.str());
	}
	RELEASE_MUTEX(CL_LOCK);

	return platforms;
}

OpenCLExecutor & OpenCLExecutor::getExecutor()
{
	if(!IS_MUTEX_VALID(CL_LOCK))
		CREATEMUTEX(CL_LOCK);

	ACQUIRE_MUTEX(CL_LOCK);
	if (internalExec == NULL)
	{
		internalExec = new OpenCLExecutor();
	}
	RELEASE_MUTEX(CL_LOCK);

	return *internalExec;
}

void OpenCLExecutor::DeinitPlatform()
{
	ACQUIRE_MUTEX(CL_LOCK);
	if (!bIsInitialized)
	{
		RELEASE_MUTEX(CL_LOCK);
		return;
	}

	if (OpenCLExecutor::internalExec != NULL)
		delete OpenCLExecutor::internalExec;

	OpenCLExecutor::internalExec = NULL;
	OpenCLExecutor::workingGroups.clear();
	bIsInitialized = false;

	RELEASE_MUTEX(CL_LOCK);
}

void OpenCLExecutor::StopKernel(FOCLKernel & kernel)
{
	throw OCLException("Undefined method 'StopKernel' called!");
}

bool OpenCLExecutor::WaitForKernel(FOCLKernel & kernel)
{
	FOCLKernelGroup* g = getWorkingGroupOfKernel(kernel);
	if (g == NULL)
		return false;

	g->WaitForGroup(&kernel);
	return true;
}

cl::Device OpenCLExecutor::getDefaultDevice()
{
	return device;
}

std::string OpenCLExecutor::decodeErrorCode(cl_int c)
{
	return clDecodeErrorCode(c);
}

bool OpenCLExecutor::runsKernel(FOCLKernel& kernel)
{

	for (int i = 0; i < workingGroups.size(); i++)
	{
		if (*workingGroups[i]->kernel == kernel)
		{
			return true;
		}
	}

	return false;
}

bool OpenCLExecutor::RunKernel(FOCLKernel & kernel, bool shouldBlockVariables, const VECTOR_CLASS<cl::Event>* events, cl::Event* event)
{
	if (deviceInfos.maxWorkGroupDimensions < kernel.localThreadCount.dimensions())
	{
		throw OCLException("kernel dimensions too high!");
		return false;
	}

	for (int i = 0; i < kernel.localThreadCount.dimensions(); i++)
	{
		if (kernel.localThreadCount[i] > deviceInfos.maxWorkItemsPerDimension[i])
		{
			throw OCLException("Workgroupdimension too big!");
			return false;
		}
	}

	if (!InitKernel(kernel))
	{
		throw OCLException("Could not initialize given Kernel!");
		return false;
	}

	return RunInitializedKernel(kernel, shouldBlockVariables, events, event);
}

void OpenCLExecutor::appendKernelToQueueOf(FOCLKernel & parent, FOCLKernel & child)
{
	ACQUIRE_MUTEX(CL_LOCK);
	FOCLKernelGroup* g = getWorkingGroupOfKernel(parent);
	if (g != NULL)
	{
		FOCLKernelGroup* c = getWorkingGroupOfKernel(child);
		if (c != NULL)
		{
			ReleaseKernel(child);
			child.context = context;
		}
		else
			InitKernel(child);
		
		workingGroups.push_back(new FOCLKernelGroup(child, g->queue));
	}
	RELEASE_MUTEX(CL_LOCK);
}

bool OpenCLExecutor::InitKernel(FOCLKernel & kernel)
{
	ACQUIRE_MUTEX(CL_LOCK);
	if (kernel.context != NULL)
	{
		RELEASE_MUTEX(CL_LOCK);
		return true;
	}

	kernel.context = context;
	kernel.device = &device;

	cl::Program::Sources sources({{ kernel.source.c_str(),kernel.source.length() }});

	try {
		kernel.program = cl::Program(*kernel.context, sources);
		if (kernel.program.build({ *kernel.device }) != CL_SUCCESS) {
			try {
				std::stringstream s;
				s << kernel.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*kernel.device);
				std::printf("Error building: %s\n", s.str().c_str());
				throw OCLException(" Error building:" + s.str());
			}
			catch (...)
			{
				std::printf("Unknown critical error found!\n");
				throw OCLException(" Error building!");
			}
			RELEASE_MUTEX(CL_LOCK);
			return false;
		}
	}
	catch (...)
	{
		std::printf("Unknown uncritical error found\n");
	}
	kernel.clKernel = cl::Kernel(kernel.program, kernel.mainMethodName.c_str());
	kernel.kernelID = std::hash<std::string>{}(kernel.source + kernel.mainMethodName);
	for(int i = 0; i < kernel.Arguments.size(); i++)
		kernel.kernelID += std::hash<int>{}(*(int*)kernel.Arguments[i]);

	RELEASE_MUTEX(CL_LOCK);

	return true;
}

bool OpenCLExecutor::RunInitializedKernel(FOCLKernel & kernel, bool shouldBlockVariables, const VECTOR_CLASS<cl::Event>* events, cl::Event* event)
{
	ACQUIRE_MUTEX(CL_LOCK);
	FOCLKernelGroup* g = getWorkingGroupOfKernel(kernel);

	if (NULL != g)
	{
		g->WaitForGroup(&kernel);

		//exec
		g->Run(events, event, &kernel);

		g->WaitForGroup(&kernel);
	}
	else
	{
		workingGroups.push_back(new FOCLKernelGroup(kernel, shouldBlockVariables));
		size_t pos = workingGroups.size() - 1;
		//exec
		workingGroups[pos]->Run(events, event, &kernel);

		workingGroups[pos]->WaitForGroup(&kernel);
	}
	RELEASE_MUTEX(CL_LOCK);

	return true;
}

void OpenCLExecutor::createWorkgroup(FOCLKernel & kernel)
{
	InitKernel(kernel);
	FOCLKernelGroup* g = getWorkingGroupOfKernel(kernel);
	if (g == NULL)
		workingGroups.push_back(new FOCLKernelGroup(kernel));
}

std::vector<OCLVariable*> OpenCLExecutor::GetAllResultsOf(FOCLKernel & kernel, bool waitForKernelToFinish)
{
	FOCLKernelGroup* group = getWorkingGroupOfKernel(kernel);

	if (group == NULL)
	{
		return std::vector<OCLVariable*>();
	}

	//ACQUIRE_MUTEX(CL_LOCK);
	if (waitForKernelToFinish)
	{
		group->WaitForGroup();
	}

	group->DownloadResults();
	//RELEASE_MUTEX(CL_LOCK);

	return group->kernel->Arguments;
}

FOCLKernelGroup* OpenCLExecutor::getWorkingGroupOfKernel(FOCLKernel& kernel)
{
	for (size_t i = 0; i < workingGroups.size(); i++)
	{
		if (*workingGroups[i]->kernel == kernel)
		{
			return workingGroups[i];
		}
	}

	return NULL;
}

void OpenCLExecutor::ReleaseKernel(FOCLKernel & kernel)
{
	for (int i = 0; i < workingGroups.size(); i++)
	{
		if (*workingGroups[i]->kernel == kernel)
		{
			//workingGroups[i]->CleanUpDevice();
			workingGroups[i]->bIsRunning = false;
			kernel.context = NULL;
			delete workingGroups[i];
			workingGroups.erase(workingGroups.begin() + i);
			break;
		}
	}
}

void OpenCLExecutor::InitOCLVariable(OCLVariable* var, void* data, FOCLKernel* kernel, size_t size)
{
	FOCLKernelGroup* group = NULL;
	if (kernel != NULL)
	{
		group = getWorkingGroupOfKernel(*kernel);
	}
	else
	{
		if (workingGroups.size() == 0)
		{
			cl::CommandQueue q = createQueue();
			if (CL_SUCCESS != var->initWithValue(&q, data, size))
				throw OCLException("Could not ini Variable");

			return;
		}

		group = workingGroups[0];
	}

	if (CL_SUCCESS != var->initWithValue(group->queue, data, size))
		throw OCLException("Could not ini Variable");
}

size_t ggT(size_t a, size_t b) {
	if (b == 0)
		return a;
	else return ggT(b, a % b);
}

cl::NDRange OpenCLExecutor::getMaxLocalNDRange(cl::NDRange globalRange, cl::NDRange rangeRatio)
{
	size_t maxSize = deviceInfos.maxWorkGroupSize;
	cl::NDRange localRange;
	if (rangeRatio.dimensions() == 0)
	{
		if (globalRange.dimensions() == 1)
		{
			localRange = cl::NDRange(maxSize);
			return localRange;
		}

		if (globalRange.dimensions() == 2 && globalRange[0]/globalRange[1] == 1)
		{
			size_t s = (size_t)sqrt(maxSize);
			localRange = cl::NDRange(s, s);
			return localRange;
		}

		if (globalRange.dimensions() == 3)
		{
			size_t s = (size_t)cbrt(maxSize);
			localRange = cl::NDRange(s, s, s);
			return localRange;
		}
	}

	if (rangeRatio.dimensions() == 2 && rangeRatio[0] == 1 && rangeRatio[1] == 0)
	{
		if (globalRange[1] <= maxSize)
		{
			localRange = cl::NDRange(1, globalRange[1]);
			return localRange;
		}
		else
		{
			for (size_t i = maxSize; i >= 2; i--)
			{
				if (globalRange[1] % i == 0)
				{
					localRange = cl::NDRange(1, i);
					return localRange;
				}
			}
		}
	}

	localRange = cl::NDRange(1, 1, 1);
	return localRange;
}

bool OpenCLExecutor::GetResultOf(FOCLKernel & kernel, OCLVariable * var, bool waitForKernelToFinish)
{
	FOCLKernelGroup* group = getWorkingGroupOfKernel(kernel);

	if (group == NULL)
	{
		return false;
	}

	//ACQUIRE_MUTEX(CL_LOCK);
	if (waitForKernelToFinish)
	{
		group->queue->enqueueBarrierWithWaitList();
		group->WaitForGroup();
	}

	group->DownloadResults({ var });

	//RELEASE_MUTEX(CL_LOCK);
	return true;
}

cl::Context OpenCLExecutor::getContext()
{
	ACQUIRE_MUTEX(CL_LOCK);
	if (!context)
	{
		context = new cl::Context(device);
	}
	RELEASE_MUTEX(CL_LOCK);

	return *context;
}

cl::CommandQueue OpenCLExecutor::createQueue()
{
	//ACQUIRE_MUTEX(CL_LOCK);
	cl::CommandQueue ret = cl::CommandQueue(getContext(), device);
	//RELEASE_MUTEX(CL_LOCK);
	return ret;
}
