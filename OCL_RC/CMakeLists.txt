cmake_minimum_required(VERSION 3.9)

PROJECT(OCL_Ressource_Compiler)

if(UNIX)
  set (CMAKE_CXX_FLAGS "-std=c++17") 
  set (CMAKE_C_FLAGS "-std=c17")
else()
  set (CMAKE_CXX_FLAGS "/std:c++17 /EHsc") 
  set (CMAKE_C_FLAGS "/std:c17 /EHsc")
endif()

add_executable(OCL_RC OCL_RC.cpp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ".")