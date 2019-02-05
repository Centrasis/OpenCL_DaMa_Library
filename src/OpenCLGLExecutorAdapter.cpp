#ifdef __USE_OPENGL__
#include "OpenCLGLExecutorAdapter.h"
#include "GLFW/glfw3.h"

void glfwErrorCallBack(int n, const char* str)
{
	//std::printf("GLFW ERROR: %s [%i]\n", str, n);
}

cl::Context OpenCLGLExecutorAdapter::getContext()
{
	if (!context)
	{
		if (!glfwInit())
			//std::printf("Init OpenGL failed!\n");

		glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 4.0
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL

		GLFWwindow* window; // (In the accompanying source code, this variable is global for simplicity)
		window = glfwCreateWindow(1024, 768, "OpenGL output", NULL, NULL);
		glfwSetErrorCallback(&glfwErrorCallBack);
		glfwMakeContextCurrent(window);

		if(glewInit() != GLEW_OK)
			//std::printf("Init OpenGL failed!\n");


		cl_context_properties props[] =
		{
			CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
			CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
			CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),
			0
		};

		cl_int err;
		context = new cl::Context(device, props, NULL, NULL, &err);

		if (err != CL_SUCCESS)
		{
			//std::printf("ERROR: Context creation: [%s]\n", OpenCLExecutor::decodeErrorCode(err).c_str());
		}
		else
		{
			//std::printf("GL Context attached!\n");
		}
	}

	return *context;
}

OpenCLGLExecutorAdapter::~OpenCLGLExecutorAdapter()
{
	for (auto itr : RenderBuffers)
	{
		glDeleteRenderbuffers(1, &itr.GLRenderBuffer);
	}

	for (auto itr : GPUImages)
	{
		glDeleteTextures(1, &itr.GLtexture);
	}
}

std::string OpenCLGLExecutorAdapter::getFragmentShaderForPixelIntegration(GLuint inBuffer, GLuint outBuffer)
{
	return "\
				#version 400\
				#extension ARB_gpu_shader_int64​ : enable​\
				layout(location = " + std::to_string(inBuffer) + ") in vec3 inPixel;\
         		layout(location = " + std::to_string(outBuffer) + ") out float diffuseColor;\
				uniform UNSIGNED_INT64_ARB startTime;\
				uniform UNSIGNED_INT64_ARB endTime;\
				\
				void main()\
				{\
					if(inPixel[0] >= startTime && inPixel[0] <= endTime) {\
						\diffuseColor = 1.0;\
					}\
					else {\
						diffuseColor = 0.0;\
					}\
				}";
}

std::string OpenCLGLExecutorAdapter::getVertexShaderForPixelIntegration(GLuint inBuffer, GLuint outBuffer)
{
	//in vec4 [x,y,time,tot]
	return "\
				#version 400\
				#extension ARB_gpu_shader_int64​ : enable​\
				\
				layout(location = 0) in vec4 inPixelVec_vec;";
}



OpenCLGLExecutorAdapter::FOpenCLGLImageBound OpenCLGLExecutorAdapter::addCLGLImageToKernel(FOCLKernel& kernel, int width, int height, EOGLTextureColorMode mode)
{
	return addCLGLImageToKernel(kernel, cv::Mat(height, width, CV_8U), mode);
}

OpenCLGLExecutorAdapter::FOpenCLGLImageBound OpenCLGLExecutorAdapter::addCLGLImageToKernel(FOCLKernel & kernel, cv::Mat & InitImg, EOGLTextureColorMode mode)
{
	//chekc if images are supported by device
	cl_int supportsImages = device.getInfo<CL_DEVICE_IMAGE_SUPPORT>();
	if (CL_TRUE != supportsImages)
	{
		//std::printf("ERROR: NO IMAGES ARE SUPPORTED ON THIS DEVICE");
		exit(-1);
	}

	FOpenCLGLImageBound ret;
	glGenTextures(1, &ret.GLtexture);
	GLenum glerr = glGetError();
	if (0 != glerr)
	{
		//std::printf("GL error: %s\n", gluErrorString(glerr));
	}
	glBindTexture(GL_TEXTURE_2D, ret.GLtexture);
	glerr = glGetError();
	if (0 != glerr)
	{
		//std::printf("GL error: %s\n", gluErrorString(glerr));
	}
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glerr = glGetError();
	if (0 != glerr)
	{
		//std::printf("GL error: %s\n", gluErrorString(glerr));
	}
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, mode, InitImg.cols, InitImg.rows, 0, mode, GL_UNSIGNED_BYTE, InitImg.data);
	glerr = glGetError();
	if (0 != glerr)
	{
		//std::printf("GL error: %s\n", gluErrorString(glerr));
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	glerr = glGetError();
	if (0 != glerr)
	{
		//std::printf("GL error: %s\n", gluErrorString(glerr));
	}
	glFlush();
	glerr = glGetError();
	if (0 != glerr)
	{
		//std::printf("GL error: %s\n", gluErrorString(glerr));
	}


	ret.width = InitImg.cols;
	ret.height = InitImg.rows;

	cl_int result = 0;
	//FOCLKernelGroup* group = getWorkingGroupOfKernel(kernel);
	ret.CLHandle = cl::ImageGL(getContext(), EOCLAccessTypes::ATRead, GL_TEXTURE_2D, 0, ret.GLtexture, &result);

	//group->queue.enqueueAcquireGLObjects();
	if (result != CL_SUCCESS)
	{
		//std::printf("Texture sharing failed! [%s]\n", OpenCLExecutor::decodeErrorCode(result).c_str());
	};
	//group->queue.enqueueReleaseGLObjects();

	GPUImages.push_back(ret);

	return ret;
}

void OpenCLGLExecutorAdapter::WriteTextureData(FOpenCLGLImageBound & imgBound, cv::Mat * data, EOGLTextureColorMode mode)
{
	glBindTexture(GL_TEXTURE_2D, imgBound.GLtexture);
	glTexImage2D(GL_TEXTURE_2D, 0, mode, data->cols, data->rows, 0, mode, GL_UNSIGNED_BYTE, data->data);
	/*cl_int err;
	auto format = GL_TEXTURE_2D;
	imgBound.CLHandle = cl::Image2D(getContext(), EOCLAccessTypes::ATReadCopy, cl::ImageFormat(CL_R, CL_UNSIGNED_INT8), data->cols, data->rows, (::size_t) 0, (void*) data->data, &err);

	if (err != CL_SUCCESS)
	{
		//std::printf("Texture sharing failed! [%s]\n", OpenCLExecutor::decodeErrorCode(err).c_str());
	};*/
}

OpenCLGLExecutorAdapter::FOpenCLGLRenderbufferBound OpenCLGLExecutorAdapter::createCLGLRenderBuffer(int width, int height, EOGLTextureColorMode mode)
{
	OpenCLGLExecutorAdapter::FOpenCLGLRenderbufferBound ret;
	GLenum texTarget = mode;
	GLuint glwidth = width;
	GLuint glheight = height;

	cl_int result;

	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glGenRenderbuffers(1, &ret.GLRenderBuffer);

	glBindRenderbuffer(GL_RENDERBUFFER, ret.GLRenderBuffer);
	
	glRenderbufferStorage(GL_RENDERBUFFER, texTarget, glwidth, glheight);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, ret.GLRenderBuffer);
	

	ret.CLHandle = cl::BufferRenderGL(getContext(), EOCLAccessTypes::ATRead, ret.GLRenderBuffer, &result);

	if (result != CL_SUCCESS) {
		//std::printf("Renderbuffer attachment failed! [%s]\n", OpenCLExecutor::decodeErrorCode(result).c_str());
	};

	RenderBuffers.push_back(ret);
	
	return ret;
}
#endif