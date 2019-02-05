#include "OCLDebugHelpers.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

OCLDebugHelpers* OCLDebugHelpers::instance = NULL;
cv::Mat* OCLDebugHelpers::defaultOutput = NULL;
MUTEXTYPE CREATEMUTEX(OCLDebugHelpers::lock);

void OCLDebugHelpers::setDefaultOutput(cv::Mat & out)
{
	defaultOutput = &out;
}

void OCLDebugHelpers::safeExec(std::function<void()> functor)
{
	ACQUIRE_MUTEX(lock);
	functor();
	RELEASE_MUTEX(lock);
}

void OCLDebugHelpers::saveImgToFile(cv::Mat & img)
{
	doInit();
	cv::Mat colored(img.rows, img.cols, CV_32FC3);
	switch (img.channels())
	{
		case 1: cv::cvtColor(img, colored, cv::COLOR_GRAY2BGR);
		case 3: colored.data = img.data;
		case 4: cv::cvtColor(img, colored, cv::COLOR_BGRA2BGR);
	}
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	if (!cv::imwrite("dbgOut_" + std::to_string(instance->debugOutCount++) + ".png", colored, compression_params))
		std::printf("could not store cv::Mat!");
}

OCLDebugHelpers::OCLDebugHelpers()
{
	shadowDrawingKernel = loadOCLKernel(DebugColoring, "draw_shadows");
	OpenCLExecutor::getExecutor().InitKernel(shadowDrawingKernel);
}


OCLDebugHelpers::~OCLDebugHelpers()
{
}

void OCLDebugHelpers::doInit()
{
	if (OCLDebugHelpers::instance == NULL)
		OCLDebugHelpers::instance = new OCLDebugHelpers();
}
