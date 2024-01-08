#ifndef KOKKOSFFT_HIP_TRANSFORM_HPP
#define KOKKOSFFT_HIP_TRANSFORM_HPP

#include <hipfft/hipfft.h>

namespace KokkosFFT {
namespace Impl {
void _exec(hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata,
           [[maybe_unused]] int direction) {
  hipfftResult hipfft_rt = hipfftExecR2C(plan, idata, odata);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecR2C failed");
}

void _exec(hipfftHandle plan, hipfftDoubleReal* idata,
           hipfftDoubleComplex* odata, [[maybe_unused]] int direction) {
  hipfftResult hipfft_rt = hipfftExecD2Z(plan, idata, odata);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecD2Z failed");
}

void _exec(hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata,
           [[maybe_unused]] int direction) {
  hipfftResult hipfft_rt = hipfftExecC2R(plan, idata, odata);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecC2R failed");
}

void _exec(hipfftHandle plan, hipfftDoubleComplex* idata,
           hipfftDoubleReal* odata, [[maybe_unused]] int direction) {
  hipfftResult hipfft_rt = hipfftExecZ2D(plan, idata, odata);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecZ2D failed");
}

void _exec(hipfftHandle plan, hipfftComplex* idata, hipfftComplex* odata,
           int direction) {
  hipfftResult hipfft_rt = hipfftExecC2C(plan, idata, odata, direction);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecC2C failed");
}

void _exec(hipfftHandle plan, hipfftDoubleComplex* idata,
           hipfftDoubleComplex* odata, int direction) {
  hipfftResult hipfft_rt = hipfftExecZ2Z(plan, idata, odata, direction);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecZ2Z failed");
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif