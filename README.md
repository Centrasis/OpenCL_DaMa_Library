# OpenCL DaMa Library
Simple data management library for OpenCL

This libray contains some simple classes for easy working with OpenCL.
The class OpenCLExecutor contains all necessary things in order to launch an OpenCL Kernel and create queues, contexts, .. .
The OpenCLGLAdapter should enable OpenCL/OpenGL object sharing. It will be enabled by selecting the cmake option "USE_OpenGL".

The OCLVariable classes are supposed to synchronize host and OpenCL data operations.
Every host data type can be wrapped by a OCLVariable class in order to work with them on a OpenCL device.
Besides normal data download and upload, the OCLRingBuffer offers to sychronize host data from a ring buffer to OpenCL memory.

The OCLMemoryVariable should be used for OpenCL Images. The method SetHostPointer() enables assigning other objects like cv::Mat classes to the OCLVariable. It can be used to load content to OpenCL or save it.
