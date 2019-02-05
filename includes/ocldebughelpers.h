#include "OpenCLTypes.h"
#include "OpenCLExecutor.h"
#include <opencv2/opencv.hpp>
#include "MultiplattformTypes.h"

class OCLDebugHelpers
{
public:
	operator OCLDebugHelpers&() const
	{
		if (OCLDebugHelpers::instance == NULL)
			OCLDebugHelpers::instance = new OCLDebugHelpers();

		return *OCLDebugHelpers::instance;
	}

	static void setDefaultOutput(cv::Mat& out);
	static void safeExec(std::function<void()> functor);
	static void saveImgToFile(cv::Mat& img);

protected:
	FOCLKernel shadowDrawingKernel;
	FOCLKernel calcShadowsFromFilterSetup;

private:
	OCLDebugHelpers();
	~OCLDebugHelpers();
	static OCLDebugHelpers* instance;
	static void doInit();
	static cv::Mat* defaultOutput;
	static MUTEXTYPE lock;
	int debugOutCount = 0;
};
