#ifndef VECTOR_KERNELS_H
#define VECTOR_KERNELS_H

#include "config.hpp"

/**
 * @file
 * \brief Small BLAS-like helpers on complex device vectors.
 *
 * They replace the few cuBLAS calls gpuNUFFT used, so that the library does
 * not need to depend on cuBLAS (several hundred MB in binary distributions).
 */
namespace gpuNUFFT
{
/** \brief Euclidean norm of a complex device vector of length n. */
DType computeComplexNorm(const DType2 *d_x, IndType n);

/** \brief In-place scaling d_x <- alpha * d_x of a complex device vector. */
void scaleComplexVector(DType2 *d_x, DType alpha, IndType n);
}

#endif  // VECTOR_KERNELS_H
