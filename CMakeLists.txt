cmake_minimum_required(VERSION 3.13)

project(tensorrt_sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

find_package(ONNX REQUIRED)
find_package(Threads REQUIRED)

find_package(CUDA 10.2 REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenCV REQUIRED)

add_executable(onnx_trt onnx_trt.cpp)
target_include_directories(onnx_trt PUBLIC 
    ${ONNX_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OPENCV_INCLUDE_DIRS}
    ${NVINFER_INCLUDES}
    ${TensorRT_INCLUDE_DIRS}
)

target_link_libraries(onnx_trt PUBLIC
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    nvinfer
    nvonnxparser
)