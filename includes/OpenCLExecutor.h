#pragma once
#include "MultiplattformTypes.h"
#include "OpenCLTypes.h"
#include <vector>

class OpenCLExecutor
{
private:
	void resolveLockThreadExec();
protected:
	OpenCLExecutor();
	virtual ~OpenCLExecutor();
	typedef struct FOCLHandledVariable
	{
		OCLVariable* var;
		std::vector<FOCLKernelGroup*> associatedGroups;

		void addGroup(FOCLKernelGroup& g)
		{
			associatedGroups.push_back(&g);
		}

		void removeGroup(FOCLKernelGroup& g)
		{
			for (int i = 0; i < associatedGroups.size(); i++)
			{
				if (&g == associatedGroups[i])
				{
					associatedGroups.erase(associatedGroups.begin() + i);
					return;
				}
			}
		}
	}FOCLHandledVariable;
	void resolveLocks();

public:
	bool InitPlatform(int platformIdx = 0, int deviceIdx = 0);
	static std::vector<std::string> GetDevices(int platform);
	static std::vector<std::string> GetPlatforms();
	/** Gets the OpenCLExecutor only! Implement other Executors in higher classes */
	static OpenCLExecutor& getExecutor();
	virtual void DeinitPlatform(); 
	virtual bool RunKernel(FOCLKernel& kernel, bool shouldBlockVariables = true, const cl::vector<cl::Event>* events = NULL, cl::Event* event = NULL);
	virtual void appendKernelToQueueOf(FOCLKernel& parent, FOCLKernel& child);
	virtual bool InitKernel(FOCLKernel& kernel);
	virtual bool RunInitializedKernel(FOCLKernel& kernel, bool shouldBlockVariables = true, const cl::vector<cl::Event>* events = NULL, cl::Event* event = NULL);
	virtual void createWorkgroup(FOCLKernel& kernel);
	virtual void StopKernel(FOCLKernel& kernel);
	virtual bool WaitForKernel(FOCLKernel& kernel);
	virtual std::vector<OCLVariable*> GetAllResultsOf(FOCLKernel& kernel, bool waitForKernelToFinish = false);
	virtual bool GetResultOf(FOCLKernel& kernel, OCLVariable* var, bool waitForKernelToFinish = false);
	virtual cl::Context getContext();
	virtual cl::Device getDefaultDevice();
	virtual cl::CommandQueue createQueue();
	static std::string decodeErrorCode(cl_int c);
	virtual bool runsKernel(FOCLKernel& kernel);
	virtual FOCLKernelGroup* getWorkingGroupOfKernel(FOCLKernel& kernel);
	virtual void ReleaseKernel(FOCLKernel& kernel);
	virtual void InitOCLVariable(OCLVariable* var, void* data, FOCLKernel* kernel = NULL, size_t size = 0);

	/**
	* Automatically select the max size of local workgroup for the selected device
	* @Param rangeRatio defines how the local workgroup extention should be defined
	* if rangeRatio is NullRange -> local NDRange should be equal if rangeRatio dim is 0 -> fill with max avaliable 
	*/
	virtual cl::NDRange getMaxLocalNDRange(cl::NDRange globalRange, cl::NDRange rangeRatio = cl::NullRange);
	bool IsInitialized() { return bIsInitialized; };
	
protected:
	static OpenCLExecutor* internalExec;
	cl::Context* context = NULL;
	cl::Device device;
	cl::Platform platform;
	std::vector<FOCLKernelGroup*> workingGroups;
	bool bIsInitialized = false;
	static MUTEXTYPE CL_LOCK;
	FOCLDeviceInfos deviceInfos;
};
