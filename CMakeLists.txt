cmake_minimum_required(VERSION 2.8)

#PROJECT(NID LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -lstdc++fs")
add_definitions(-DCOMPILEDWITHC14)#add_definitions(-DCOMPILEDWITHC11)
message(STATUS "Using flag -std=c++14.")
add_definitions(-w) #not show warning

find_package(CUDA QUIET REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS -arch=compute_60)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(OpenCV QUIET)

SET(EIGEN3_INCLUDE_DIR ${G2O_EIGEN3_INCLUDE})
FIND_PACKAGE(Eigen3 REQUIRED)
IF(EIGEN3_FOUND)
  SET(G2O_EIGEN3_INCLUDE ${EIGEN3_INCLUDE_DIR} CACHE PATH "Directory of Eigen3")
ELSE(EIGEN3_FOUND)
  SET(G2O_EIGEN3_INCLUDE "" CACHE PATH "Directory of Eigen3")
ENDIF(EIGEN3_FOUND)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
)

SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(NID_standard_property NID_standard_property.cpp)
target_link_libraries(NID_standard_property ${OpenCV_LIBS})

cuda_add_executable(NID_pose_estimation NID_pose_estimation.cpp CudaComputeHref.cu CudaPoints3d.cu)
target_link_libraries(NID_pose_estimation ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/g2o/lib/libg2o.so)

