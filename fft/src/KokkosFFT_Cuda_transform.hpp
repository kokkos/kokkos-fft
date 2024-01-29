#ifndef KOKKOSFFT_CUDA_TRANSFORM_HPP
#define KOKKOSFFT_CUDA_TRANSFORM_HPP

#include <cufft.h>

namespace KokkosFFT {
namespace Impl {
void _exec(cufftHandle& plan, cufftReal* idata, cufftComplex* odata,
           [[maybe_unused]] int direction) {
  cufftResult cufft_rt = cufftExecR2C(plan, idata, odata);
  if (cufft_rt != CUFFT_SUCCESS)
    throw std::runtime_error("cufftExecR2C failed");
}

void _exec(cufftHandle& plan, cufftDoubleReal* idata, cufftDoubleComplex* odata,
           [[maybe_unused]] int direction) {
  cufftResult cufft_rt = cufftExecD2Z(plan, idata, odata);
  if (cufft_rt != CUFFT_SUCCESS)
    throw std::runtime_error("cufftExecD2Z failed");
}

void _exec(cufftHandle& plan, cufftComplex* idata, cufftReal* odata,
           [[maybe_unused]] int direction) {
  cufftResult cufft_rt = cufftExecC2R(plan, idata, odata);
  if (cufft_rt != CUFFT_SUCCESS)
    throw std::runtime_error("cufftExecC2R failed");
}

void _exec(cufftHandle& plan, cufftDoubleComplex* idata, cufftDoubleReal* odata,
           [[maybe_unused]] int direction) {
  cufftResult cufft_rt = cufftExecZ2D(plan, idata, odata);
  if (cufft_rt != CUFFT_SUCCESS)
    throw std::runtime_error("cufftExecZ2D failed");
}

void _exec(cufftHandle& plan, cufftComplex* idata, cufftComplex* odata,
           int direction) {
  cufftResult cufft_rt = cufftExecC2C(plan, idata, odata, direction);
  if (cufft_rt != CUFFT_SUCCESS)
    throw std::runtime_error("cufftExecC2C failed");
}

void _exec(cufftHandle& plan, cufftDoubleComplex* idata,
           cufftDoubleComplex* odata, int direction) {
  cufftResult cufft_rt = cufftExecZ2Z(plan, idata, odata, direction);
  if (cufft_rt != CUFFT_SUCCESS)
    throw std::runtime_error("cufftExecZ2Z failed");
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif