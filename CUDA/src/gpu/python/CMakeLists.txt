cmake_minimum_required(VERSION 3.15)
find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)
MESSAGE(STATUS "Building Python interface")

include_directories(
    ${GPUNUFFT_INC_DIR}
    ${PYBIND11_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    )
cuda_include_directories(${GPUNUFFT_INC_DIR})
# Add atomic and other libraries here so that we can build a single .so
set(GPU_CU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../atomic/atomic_gpuNUFFT.cu)
set(GPU_CU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../coarse/coarse_gpuNUFFT.cu)
cuda_add_library(gpuNUFFT  ${GPU_CU_SOURCES}  ${GPU_GPUNUFFT_SOURCES} ${GPUNUFFT_SOURCES} ${GPUNUFFT_INCLUDE} gpuNUFFT_operator_python_factory.cpp SHARED)
set_target_properties(gpuNUFFT PROPERTIES PREFIX "")
CUDA_ADD_CUFFT_TO_TARGET(gpuNUFFT)
CUDA_ADD_CUBLAS_TO_TARGET(gpuNUFFT)
message(STATUS ${GPU_CU_SOURCES} ${GPU_GPUNUFFT_SOURCES} ${GPUNUFFT_SOURCES} ${GPUNUFFT_INCLUDE})
TARGET_LINK_LIBRARIES(gpuNUFFT ${CUDA_LIBRARIES})
if(WIN32)
    MESSAGE("Pybind11 include dir ${PYBIND11_INCLUDE_DIR}")
    MESSAGE("Python include dir ${PYTHON_INCLUDE_DIR}")
    MESSAGE("Found ${PYTHON_LIBRARIES}")
    set_target_properties(gpuNUFFT PROPERTIES SUFFIX ".pyd")
endif(WIN32)
