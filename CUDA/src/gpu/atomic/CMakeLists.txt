MESSAGE("Atomic GPU GpuNUFFT Library")
#add exe to build from gpuNUFFT

include_directories(${GPUNUFFT_INC_DIR})
													
cuda_include_directories(${GPUNUFFT_INC_DIR})

set(GPU_CU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/atomic_gpuNUFFT.cu 	
					#${CMAKE_CURRENT_SOURCE_DIR}/atomic_gpuNUFFT_kernels.cu 
					#${CMAKE_CURRENT_SOURCE_DIR}/../std_gpuNUFFT_kernels.cu
									 )
if(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_ATM_NAME} ${GPU_CU_SOURCES}  ${GPU_GPUNUFFT_SOURCES} ${GPUNUFFT_SOURCES} ${GPUNUFFT_INCLUDE} )
else(WIN32)
    CUDA_ADD_LIBRARY(${GRID_LIB_ATM_NAME} ${GPU_CU_SOURCES}  ${GPU_GPUNUFFT_SOURCES} ${GPUNUFFT_SOURCES} ${GPUNUFFT_INCLUDE}  SHARED)
endif(WIN32)

CUDA_ADD_CUFFT_TO_TARGET(${GRID_LIB_ATM_NAME})
CUDA_ADD_CUBLAS_TO_TARGET(${GRID_LIB_ATM_NAME})
target_link_libraries(${GRID_LIB_ATM_NAME} ${CUDA_curand_LIBRARY})