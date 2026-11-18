#include "vector_kernels.hpp"
#include "cuda_utils.hpp"

#include <cmath>
#include <vector>

namespace
{
const int kThreads = 256;
const int kMaxBlocks = 1024;

// Each block writes the partial sum of |x_i|^2 (accumulated in double for
// accuracy) for its grid-stride share of the vector.
__global__ void squaredNormPartialKernel(const DType2 *x, IndType n,
                                         double *partial)
{
  __shared__ double cache[kThreads];
  double sum = 0.0;
  for (IndType i = blockIdx.x * (IndType)blockDim.x + threadIdx.x; i < n;
       i += (IndType)blockDim.x * gridDim.x)
  {
    const double re = x[i].x;
    const double im = x[i].y;
    sum += re * re + im * im;
  }
  cache[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
      cache[threadIdx.x] += cache[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partial[blockIdx.x] = cache[0];
}

__global__ void scaleKernel(DType2 *x, DType alpha, IndType n)
{
  for (IndType i = blockIdx.x * (IndType)blockDim.x + threadIdx.x; i < n;
       i += (IndType)blockDim.x * gridDim.x)
  {
    x[i].x *= alpha;
    x[i].y *= alpha;
  }
}

int numBlocks(IndType n)
{
  IndType blocks = (n + kThreads - 1) / kThreads;
  if (blocks > (IndType)kMaxBlocks)
    blocks = kMaxBlocks;
  return blocks > 0 ? (int)blocks : 1;
}
}  // namespace

DType gpuNUFFT::computeComplexNorm(const DType2 *d_x, IndType n)
{
  const int blocks = numBlocks(n);
  double *d_partial = NULL;
  allocateDeviceMem(&d_partial, blocks);
  // Clear a stale (non-sticky) error left by an unrelated earlier unchecked
  // call, so that only a failure of this launch is reported below.
  cudaGetLastError();
  squaredNormPartialKernel<<<blocks, kThreads>>>(d_x, n, d_partial);
  // Note: HANDLE_ERROR evaluates its argument twice, so store the error first.
  cudaError_t err = cudaGetLastError();
  HANDLE_ERROR(err);

  std::vector<double> partial(blocks);
  copyFromDevice(d_partial, partial.data(), blocks);
  freeDeviceMem(d_partial);

  double sum = 0.0;
  for (int b = 0; b < blocks; b++)
    sum += partial[b];
  return (DType)std::sqrt(sum);
}

void gpuNUFFT::scaleComplexVector(DType2 *d_x, DType alpha, IndType n)
{
  cudaGetLastError();  // clear stale errors, see computeComplexNorm
  scaleKernel<<<numBlocks(n), kThreads>>>(d_x, alpha, n);
  cudaError_t err = cudaGetLastError();
  HANDLE_ERROR(err);
}
