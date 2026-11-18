cmake_minimum_required(VERSION 3.15)
find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

MESSAGE(STATUS "Building Python interface")
MESSAGE("Pybind11 include dir ${PYBIND11_INCLUDE_DIR}")
MESSAGE("Python include dir ${Python3_INCLUDE_DIRS}")
MESSAGE("Found ${Python3_LIBRARIES}")

include_directories(
    ${GPUNUFFT_INC_DIR}
    ${PYBIND11_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    )
cuda_include_directories(${GPUNUFFT_INC_DIR})
cuda_add_library(gpuNUFFT  ${GPU_CU_SOURCES}  ${GPU_GPUNUFFT_SOURCES} ${GPUNUFFT_SOURCES} ${GPUNUFFT_INCLUDE} gpuNUFFT_operator_python_factory.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../atomic/atomic_gpuNUFFT.cu SHARED)
set_target_properties(gpuNUFFT PROPERTIES PREFIX "")


if(WIN32)
    set_target_properties(gpuNUFFT PROPERTIES SUFFIX ".pyd")

    TARGET_LINK_LIBRARIES(gpuNUFFT ${CUDA_LIBRARIES} ${GRID_LIB_ATM_NAME} ${Python3_LIBRARIES})
elseif(UNIX)
    TARGET_LINK_LIBRARIES(gpuNUFFT ${CUDA_LIBRARIES})
endif(WIN32)
# Only cuFFT is needed (no cuBLAS / cuRAND, see vector_kernels.cu).
CUDA_ADD_CUFFT_TO_TARGET(gpuNUFFT)

# For binary wheels: look up the CUDA library (cuFFT) in the
# NVIDIA pip package installed next to the module in site-packages, e.g.
# GPUNUFFT_RPATH="$ORIGIN/nvidia/cu13/lib". The absolute toolkit path of the
# build machine is NOT embedded in that case.
SET(GPUNUFFT_RPATH "" CACHE STRING "RPATH to embed in the python module (wheel builds)")
if(GPUNUFFT_RPATH AND UNIX)
    message(STATUS "Python module RPATH: ${GPUNUFFT_RPATH}")
    set_target_properties(gpuNUFFT PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "${GPUNUFFT_RPATH}")
endif()

