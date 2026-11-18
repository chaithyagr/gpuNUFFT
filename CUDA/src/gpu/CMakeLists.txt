SET (GPU_GPUNUFFT_SOURCES ${GPUNUFFT_SOURCES} 
		 ${CMAKE_CURRENT_SOURCE_DIR}/precomp_kernels.cu
		 ${CMAKE_CURRENT_SOURCE_DIR}/vector_kernels.cu)

# The standalone gpuNUFFT_ATM library is not needed by the Python module on
# Linux (it compiles the sources itself), so wheel builds can skip it.
SET(GEN_ATM_LIB ON CACHE BOOL "Build the standalone gpuNUFFT_ATM shared library.")
if(GEN_ATM_LIB OR WIN32 OR GEN_MEX_FILES OR GEN_TESTS)
	ADD_SUBDIRECTORY(atomic)
endif()

if(GEN_MEX_FILES)
  ADD_SUBDIRECTORY(matlab)
endif(GEN_MEX_FILES)

if (GEN_PYTHON_FILES)
	ADD_SUBDIRECTORY(python)
endif(GEN_PYTHON_FILES)
