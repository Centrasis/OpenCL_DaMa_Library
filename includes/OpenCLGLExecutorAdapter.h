#pragma once
#ifdef __USE_OPENGL__
#include "OpenCLExecutor.h"
#include <CL/cl2.hpp>
#include "GL/glew.h"
#include <gl/GL.h>
#include "opencv2/core.hpp"

class OpenCLGLExecutorAdapter :
	public OpenCLExecutor
{
public:
	virtual cl::Context getContext() override;

//protected:
public:
	virtual ~OpenCLGLExecutorAdapter();

	struct FOpenCLGLImageBound
	{
		GLuint GLtexture;
		int width, height;
		cl::ImageGL CLHandle;
		//cl::Image2D CLHandle;
	};

	struct FOpenCLGLRenderbufferBound
	{
		GLuint GLRenderBuffer;
		cl::BufferRenderGL CLHandle;
	};

	enum EOGLTextureColorMode 
	{
		TCM_Gray = GL_RED,
		TCM_RGB = GL_RGB,
		TCM_BGR = GL_BGR_EXT,
		TCM_RGBA = GL_RGBA
	};

	std::string getFragmentShaderForPixelIntegration(GLuint inBuffer, GLuint outBuffer);
	std::string getVertexShaderForPixelIntegration(GLuint inBuffer, GLuint outBuffer);
	FOpenCLGLImageBound addCLGLImageToKernel(FOCLKernel& kernel, int width, int height, EOGLTextureColorMode mode = EOGLTextureColorMode::TCM_Gray);
	FOpenCLGLImageBound addCLGLImageToKernel(FOCLKernel& kernel, cv::Mat& InitImg, EOGLTextureColorMode mode = EOGLTextureColorMode::TCM_Gray);
	void WriteTextureData(FOpenCLGLImageBound & imgBound, cv::Mat* data, EOGLTextureColorMode mode = EOGLTextureColorMode::TCM_Gray);
	FOpenCLGLRenderbufferBound createCLGLRenderBuffer(int width, int height, EOGLTextureColorMode mode = EOGLTextureColorMode::TCM_Gray);

private:
	std::vector<FOpenCLGLRenderbufferBound> RenderBuffers;
	std::vector<FOpenCLGLImageBound> GPUImages;
};
#endif